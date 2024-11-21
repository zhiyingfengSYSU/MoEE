import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
# from models.densenet import DenseNet3
# from models.wrn_godin import WideResNet
from models.densenet_godin import DenseNet3
# from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from utilsood.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import utilsood.score_calculation as lib
from utilsood.run_knn import run_knn_func
from utilsood.vim import get_vim
# from sklearn.svm import OneClassSVM
# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def get_SVM_score(ood_loader, clf, in_score, out_as_pos):
    start = True
    for data, _ in ood_loader:
        x = data.numpy()
        if start:
            X = x
            start = False
        else:
            X = np.concatenate((X, x))
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    out_score = clf.predict(X)
    for i in range(len(out_score)):
        if out_score[i] == -1:
            out_score[i] = 1
        else:
            out_score[i] = 0
    if out_as_pos:  # OE's defines out samples as positive
        measures = get_measures(out_score, in_score)
    else:
        measures = get_measures(-in_score, -out_score)
    auroc = np.mean(measures[0])
    aupr = np.mean(measures[1])
    fpr = np.mean((measures[2]))
    print_measures(auroc, aupr)
    return auroc, aupr, fpr
def get_other_score(in_score, out_score, out_as_pos):
    print("inscore shape", in_score.shape)
    print("out_score shape", out_score.shape)
    if out_as_pos:  # OE's defines out samples as positive
        measures = get_measures(out_score, in_score)
    else:
        measures = get_measures(-in_score, -out_score)
    auroc = np.mean(measures[0])
    aupr = np.mean(measures[1])
    fpr = np.mean((measures[2]))
    print_measures(auroc, aupr)
    return auroc, aupr, fpr




def get_ood_scores(user_class, test_bs, use_xent, score, T, ood_num_examples, net, to_np,concat, loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []
    #print("ood_num_examples", ood_num_examples)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // test_bs and in_dist is False and batch_idx > 0:
                break

            data = data.cuda()

            output = net(data)
            #print("output shape", output.shape)
            output = output[:, user_class]
            smax = to_np(F.softmax(output, dim=1))

            if use_xent:
                _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                if score == 'energy':
                    _score.append(-to_np((T * torch.logsumexp(output / T, dim=1))))
                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                    _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                if use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def VOS_evaluate(args, out_as_pos, num_to_avg, use_xent, method_name, score, test_bs, T, noise, net, test_loader, train_loader, user_class, data_name=None, m_name=None, client_id=None):
    # mean and standard deviation ofevaluate channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    num_classes = 10
    net.eval()


    cudnn.benchmark = True  # fire on all cylinders

    # /////////////// Detection Prelims ///////////////
    ood_num_examples = len(test_loader)*test_bs // 5

    #expected_ap = ood_num_examples / (ood_num_examples + len(test_loader))

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    if score == 'odin':
        # separated because no grad is not applied
        in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, test_bs, ood_num_examples, T, noise, in_dist=True)
        sample_mean = 0
        precision = 0
        count = 0
        num_batches = 0
    elif score == 'M':
        from torch.autograd import Variable

        _, right_score, wrong_score = get_ood_scores(user_class, test_bs, use_xent, score, T, ood_num_examples, net, to_np, concat, test_loader, in_dist=True)

        num_batches = ood_num_examples // test_bs

        temp_x = torch.rand(2, 3, 32, 32)
        temp_x = Variable(temp_x)
        temp_x = temp_x.cuda()
        temp_list = net.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        print('get sample mean and covariance', count)
        sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
        in_score = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count - 1,
                                             noise,
                                             num_batches, in_dist=True)
        print(in_score[-3:], in_score[-103:-100])
    elif score == 'SVM':
        print("score", score)
        start = True
        #transform training data from numpy to tensor
        for data, _ in train_loader:
            x = data.numpy()
            if start:
                X = x
                start = False
            else:
                X = np.concatenate((X, x))
        X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
        clf = OneClassSVM(gamma='auto').fit(X)
        start = True
        # transform test data from numpy to tensor
        for data, _ in test_loader:
            x = data.numpy()
            if start:
                X = x
                start = False
            else:
                X = np.concatenate((X, x))
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        in_score = clf.predict(X)
        for i in range(len(in_score)):
            if in_score[i] == -1:
                in_score[i] = 1
            else:
                in_score[i] = 0
    else:
        in_score, right_score, wrong_score = get_ood_scores(user_class, test_bs, use_xent, score, T, ood_num_examples, net, to_np, concat, test_loader, in_dist=True)
        sample_mean = 0
        precision = 0
        count = 0
        num_batches = 0


    #num_right = len(right_score)
    #num_wrong = len(wrong_score)


    # /////////////// End Detection Prelims ///////////////



    # /////////////// Error Detection ///////////////

    #print('\n\nError Detection')
    #show_performance(wrong_score, right_score, method_name=method_name)

    # /////////////// OOD Detection ///////////////
    auroc_list, aupr_list = [], []
    auroc_dict, aupr_dict = {}, {}


    # /////////////// Texture ///////////////
    if data_name == 'stl':
        ood_data = dset.ImageFolder(root="dataset/dtd/images",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
    elif data_name in ['DomainNet', 'ImageNet']:
        ood_data = dset.ImageFolder(root="dataset/dtd/images",
                                    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(256),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
    else:
        ood_data = dset.ImageFolder(root="dataset/dtd/images",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=0, pin_memory=True)
    print('\n\nTexture Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        print("score", score)
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs, num_classes, data_name, ['Texture'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        print("score", score)
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['Texture'], args.batch, test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    else:
        auroc, aupr, _ = get_and_print_results(args, user_class, use_xent, method_name, score, test_bs, T, out_as_pos, noise, to_np, concat, net, ood_num_examples, num_classes, sample_mean, precision, count, num_batches, in_score, ood_loader, num_to_avg, data_name, 'Textures', client_id, args.use_external, args.loss_weight)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    auroc_dict['Texture'] = auroc
    aupr_dict['Texture'] = aupr


    # /////////////// SVHN /////////////// # cropped and no sampling of the test set
    # ood_data = dset.ImageFolder(root="dataset/svhn_t",
    #                     transform=trn.Compose(
    #                         [trn.Resize(32),
    #                         trn.ToTensor(), trn.Normalize(mean, std)]))
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
    #                                         num_workers=0, pin_memory=True)
    # print('\n\nSVHN Detection')
    # get_and_print_results(ood_loader)

    # /////////////// Places365 ///////////////
    # if data_name == 'stl':
    #
    #     ood_data = dset.ImageFolder(root="dataset/places365",
    #                                 transform=trn.Compose([trn.Resize(96), trn.CenterCrop(96),
    #                                                        trn.ToTensor(), trn.Normalize(mean, std)]))
    # elif data_name in ['DomainNet', 'ImageNet']:
    #
    #     ood_data = dset.ImageFolder(root="dataset/places365",
    #                                 transform=trn.Compose([trn.Resize(256), trn.CenterCrop(256),
    #                                                        trn.ToTensor(), trn.Normalize(mean, std)]))
    # else:
    #     ood_data = dset.ImageFolder(root="dataset/places365",
    #                             transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
    #                                                    trn.ToTensor(), trn.Normalize(mean, std)]))
    # ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
    #                                          num_workers=0, pin_memory=True)
    # print('\n\nPlaces365 Detection')
    # if score == 'SVM':
    #     auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    # elif score == 'KNN':
    #     scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
    #                                             num_classes, data_name, ['Places365'], m_name)
    #     auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    # elif score == 'vim':
    #     scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['Places365'], args.batch,
    #                                    test_bs, num_classes, m_name)
    #     auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    # else:
    #     auroc, aupr, _= get_and_print_results(args, user_class, use_xent, method_name, score, test_bs,T,out_as_pos, noise, to_np, concat,net,ood_num_examples,num_classes,sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg)
    # auroc_list.append(auroc);
    # aupr_list.append(aupr);
    # auroc_dict['SVHN'] = auroc
    # aupr_dict['SVHN'] = aupr


    # /////////////// LSUN-C ///////////////
    if data_name == 'stl':
        ood_data = dset.ImageFolder(root="dataset/LSUN_C/LSUN_C",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    elif data_name in ['DomainNet', 'ImageNet']:
        ood_data = dset.ImageFolder(root="dataset/LSUN_C/LSUN_C",
                                    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(256), trn.ToTensor(), trn.Normalize(mean, std)]))
    else:
        ood_data = dset.ImageFolder(root="dataset/LSUN_C/LSUN_C",
                                transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=0, pin_memory=True)
    print('\n\nLSUN_C Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
                                                num_classes, data_name, ['LSUN_C'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['LSUN_C'], args.batch,
                                       test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    else:
        auroc, aupr, _= get_and_print_results(args, user_class, use_xent, method_name, score, test_bs,T,out_as_pos, noise, to_np, concat,net,ood_num_examples,num_classes,sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg, data_name, 'LSUN-C', client_id, args.use_external, args.loss_weight)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    auroc_dict['LSUN_C'] = auroc
    aupr_dict['LSUN_C'] = aupr


    # /////////////// LSUN-R ///////////////
    if data_name == 'stl':
        ood_data = dset.ImageFolder(root="dataset/LSUN_resize",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    elif data_name in ['DomainNet', 'ImageNet']:
        ood_data = dset.ImageFolder(root="dataset/LSUN_resize",
                                    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(256), trn.ToTensor(), trn.Normalize(mean, std)]))
    else:
        ood_data = dset.ImageFolder(root="dataset/LSUN_resize",
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=0, pin_memory=True)
    print('\n\nLSUN_Resize Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
                                                num_classes, data_name, ['LSUN_R'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['LSUN_R'], args.batch,
                                       test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)

    else:
        auroc, aupr, _= get_and_print_results(args, user_class, use_xent, method_name, score, test_bs,T,out_as_pos, noise, to_np, concat,net,ood_num_examples,num_classes,sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg, data_name, 'LSUN-R', client_id, args.use_external, args.loss_weight)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    auroc_dict['LSUN_Resize'] = auroc
    aupr_dict['LSUN_Resize'] = aupr


    # /////////////// iSUN ///////////////
    if data_name == 'stl':
        ood_data = dset.ImageFolder(root="dataset/iSUN",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
    elif data_name in ['DomainNet', 'ImageNet']:
        ood_data = dset.ImageFolder(root="dataset/iSUN",
                                    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(256), trn.ToTensor(), trn.Normalize(mean, std)]))
    else:
        ood_data = dset.ImageFolder(root="dataset/iSUN",
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
                                             num_workers=0, pin_memory=True)
    print('\n\niSUN Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
                                                num_classes, data_name, ['iSUN'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['iSUN'], args.batch,
                                       test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    else:
        auroc, aupr, _= get_and_print_results(args, user_class, use_xent, method_name, score, test_bs,T,out_as_pos, noise, to_np, concat,net,ood_num_examples,num_classes,sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg, data_name, 'iSUN', client_id, args.use_external, args.loss_weight)
    auroc_list.append(auroc);
    aupr_list.append(aupr);
    auroc_dict['iSUN'] = auroc
    aupr_dict['iSUN'] = aupr


    # /////////////// MNIST ///////////////
    import torchvision
    import torchvision.transforms as transforms
    if data_name == 'stl':

        transform_mnist = transforms.Compose([transforms.Resize((32, 32)),
                    transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor()])

        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_mnist)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)

    elif data_name in ['DomainNet', 'ImageNet']:
        transform_mnist =  transforms.Compose([transforms.Resize((256, 256)),
                    transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor()])

        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_mnist)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)
    else:
        transform_mnist =  transforms.Compose([transforms.Resize((32, 32)),
                    transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_mnist)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)

    print('\n\nMNIST Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
                                                num_classes, data_name, ['MNIST'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['MNIST'], args.batch,
                                       test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    else:
        auroc, aupr, _= get_and_print_results(args, user_class, use_xent, method_name, score, test_bs, T, out_as_pos, noise, to_np, concat, net, ood_num_examples, num_classes, sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg, data_name, 'MNIST', client_id, args.use_external, args.loss_weight)
    if args.data != 'mnist':
        auroc_list.append(auroc);
        aupr_list.append(aupr);
    auroc_dict['MNIST'] = auroc
    aupr_dict['MNIST'] = aupr

    # /////////////// FASHION-MNIST ///////////////
    import torchvision
    import torchvision.transforms as transforms
    if data_name == 'stl':

        transform_Fashionmnist = transforms.Compose([transforms.Resize((32, 32)),
                    transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor()])

        trainset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform_Fashionmnist)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)

    elif data_name in ['DomainNet', 'ImageNet']:
        transform_Fashionmnist =  transforms.Compose([transforms.Resize((256, 256)),
                    transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor()])

        trainset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform_Fashionmnist)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)
    else:
        transform_Fashionmnist = transforms.Compose([transforms.Resize((32, 32)),
                    transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor()])
        trainset = torchvision.datasets.FashionMNIST(root='./dataset', train=True, download=True, transform=transform_Fashionmnist)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)

    print('\n\nFashionMNIST Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
                                                num_classes, data_name, ['FashionMNIST'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['FashionMNIST'], args.batch,
                                       test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    else:
        auroc, aupr, _= get_and_print_results(args, user_class, use_xent, method_name, score, test_bs, T, out_as_pos, noise, to_np, concat, net, ood_num_examples,num_classes,sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg, data_name, 'FASHION-MNIST', client_id, args.use_external, args.loss_weight)
    if args.data != 'FashionMNIST':
        auroc_list.append(auroc);
        aupr_list.append(aupr);
    auroc_dict['FashionMNIST'] = auroc
    aupr_dict['FashionMNIST'] = aupr

    # /////////////// CIFAR10 ///////////////
    import torchvision
    import torchvision.transforms as transforms
    if data_name == 'stl':

        transform_cifar10 = transforms.Compose([transforms.Resize((32, 32)),
                     transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_cifar10)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)

    elif data_name in ['DomainNet', 'ImageNet']:
        transform_cifar10 =  transforms.Compose([transforms.Resize((256, 256)),
                     transforms.ToTensor()])

        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_cifar10)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)
    else:
        transform_cifar10 =  transforms.Compose([transforms.Resize((32, 32)),
                     transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_cifar10)
        ood_loader = torch.utils.data.DataLoader(trainset, batch_size=test_bs, shuffle=True, num_workers=0, pin_memory=True)

    print('\n\nCIFAR10 Detection')
    if score == 'SVM':
        auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    elif score == 'KNN':
        scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
                                                num_classes, data_name, ['CIFAR10'], m_name)
        auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    elif score == 'vim':
        scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['CIFAR10'], args.batch,
                                       test_bs, num_classes, m_name)
        auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    else:
        auroc, aupr, _ = get_and_print_results(args, user_class, use_xent, method_name, score, test_bs, T, out_as_pos, noise, to_np, concat, net, ood_num_examples,num_classes,sample_mean, precision, count,num_batches,in_score, ood_loader, num_to_avg, data_name, 'CIFAR10', client_id, args.use_external, args.loss_weight)
    if args.data != 'cifar10':
        auroc_list.append(auroc)
        aupr_list.append(aupr)
    auroc_dict['CIFAR10'] = auroc
    aupr_dict['CIFAR10'] = aupr

    # #///////////CIFAR100//////////////////
    # #mean and standard deviation of channels of CIFAR-100 images
    # if args.data == 'cifar10':
    #     mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    #     std = [x / 255 for x in [63.0, 62.1, 66.7]]
    #
    #     test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    #     ood_data = dset.CIFAR100('dataset/cifar10', train=False, transform=test_transform)
    #     ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=test_bs, shuffle=True,
    #                                              num_workers=0, pin_memory=True)
    #     print('\n\nCIFAR100 Detection')
    #     if score == 'SVM':
    #         auroc, aupr, _ = get_SVM_score(ood_loader, clf, in_score, out_as_pos)
    #     elif score == 'KNN':
    #         scores_in, all_score_ood = run_knn_func(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, args.batch, test_bs,
    #                                                 num_classes, data_name, ['Cifar100'], m_name)
    #         auroc, aupr, _ = get_other_score(scores_in, all_score_ood, out_as_pos)
    #     elif score == 'vim':
    #         scores_in, score_out = get_vim(client_id, args.loss_weight, train_loader, test_loader, ood_loader, net, data_name, ['Cifar100'],
    #                                        args.batch,
    #                                        test_bs, num_classes, m_name)
    #         auroc, aupr, _ = get_other_score(scores_in, score_out, out_as_pos)
    #     else:
    #         auroc, aupr, _ = get_and_print_results(args, user_class, use_xent, method_name, score, test_bs, T, out_as_pos,
    #                                              noise, to_np, concat,
    #                                              net, ood_num_examples, num_classes, sample_mean, precision, count,
    #                                              num_batches, in_score, ood_loader, num_to_avg, data_name, 'CIFAR100', client_id, args.use_external, args.loss_weight)
    #     auroc_list.append(auroc);
    #     aupr_list.append(aupr);
    #     auroc_dict['CIFAR100'] = auroc
    #     aupr_dict['CIFAR100'] = aupr


    # /////////////// Mean Results ///////////////

    print('\n\nMean Test Results!!!!!')
    print_measures(np.mean(auroc_list), np.mean(aupr_list),  method_name=method_name)
    # return np.mean(auroc_list), np.mean(aupr_list), auroc_list, aupr_list
    return np.mean(auroc_list), np.mean(aupr_list), auroc_dict, aupr_dict


def get_and_print_results(args, user_class, use_xent, method_name, score, test_bs, T, out_as_pos, noise, to_np, concat, net, ood_num_examples, num_classes, sample_mean, precision, count, num_batches, in_score, ood_loader, num_to_avg, in_name, out_name, id_client, method, loss_weight):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if score == 'odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, test_bs, ood_num_examples, T, noise)
        elif score == 'M':
            out_score = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count - 1, noise, num_batches)
        else:
            out_score = get_ood_scores(user_class, test_bs, use_xent, score, T, ood_num_examples, net, to_np, concat, ood_loader)
        if out_as_pos:  # OE's defines out samples as positive
            measures = get_measures(out_score, in_score, in_name, out_name, id_client, method, score, loss_weight, args)
        else:
            measures = get_measures(-in_score, -out_score, in_name, out_name, id_client, method, score, loss_weight, args)
        aurocs.append(measures[0]);
        auprs.append(measures[1]);
        fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)


    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, method_name)
    else:
        print_measures(auroc, aupr, method_name)
    return auroc, aupr, fpr



