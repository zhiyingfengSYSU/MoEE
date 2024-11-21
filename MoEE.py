import os, argparse, copy, time
import numpy as np
import torch
from torch import nn, optim
from learning import train, test
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, \
    MultiStepLR, LocalMaskCrossEntropyLoss, str2bool
from utils.config import CHECKPOINT_ROOT
import torchvision.transforms as trn
# NOTE import desired federation
from core import _Federation as Federation
from core import AdversaryCreator
#models
from models.allconv import AllConvNet
from models.wrn_virtual import WideResNet, linear_classifier, WideResNet_Tin, WideResNet_stl, WideResNet_Domain, WideResNet_MNIST
from VOS_virtual import VOS_train, VOS_train2, VOS_train_prox, inversion_train, topk_inversion_train, topk_inversion_train_prox, visualization, visualization2, visualization_external, get_weights
from VOS_evaluate import VOS_evaluate
from torch.utils.data import Dataset
from oodgen import CentralGen, generator_update_InversGenerator, LocalGenerator_update_InversGenerator

from utils_heterogenous import feature_Eulermeasure, feature_COSmeasure, classifier_Eulermeasure, classifier_COSmeasure, calcCov, covMat, mean_calcualation, global_classifier_aggregation
from nets.models import InversGenerator
from core_HeterDataset import _Federation_HeterDataset, _Federation_HeterDataset_MoE
from MoE import FedAvg, MoE_training_2, MoE_inference_2
import copy
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from models.ViTransformer import VisionTransformer

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utilsood.tinyimages_80mn_loader import TinyImages

class SimpleDataSet(Dataset):
    """ load synthetic time series data"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
        )

def render_run_name(args, exp_folder):
    """Return a unique run_name from given args."""
    if args.model == 'default':
        args.model = {'Digits': 'digit', 'cifar10': 'preresnet18', 'DomainNet': 'alex'}[args.data]
    run_name = f'{args.model}'
    if args.width_scale != 1.: run_name += f'x{args.width_scale}'
    run_name += Federation.render_run_name(args)
    # log non-default args
    if args.seed != 1: run_name += f'__seed_{args.seed}'
    # opt
    if args.lr_sch != 'none': run_name += f'__lrs_{args.lr_sch}'
    if args.opt != 'sgd': run_name += f'__opt_{args.opt}'
    if args.batch != 32: run_name += f'__batch_{args.batch}'
    if args.wk_iters != 1: run_name += f'__wk_iters_{args.wk_iters}'
    # slimmable
    if args.no_track_stat: run_name += f"__nts"
    if args.no_mask_loss: run_name += f'__nml'
    # adv train
    if args.adv_lmbd > 0:
        run_name += f'__at{args.adv_lmbd}'
    run_name += f'__at{args.loss_weight}'
    run_name += f'__ex{args.use_external}'
    if args.select_generator != None:
        run_name += f'__ex{args.select_generator}'
    if args.method != 'OE':
        run_name += f'__m{args.method}'
    args.save_path = os.path.join(CHECKPOINT_ROOT, exp_folder)
    if args.score != 'OE':
        run_name += f'__score{args.method}'
    if args.sample_number != 1000:
        run_name += f'__sample{args.sample_number}'
    if args.soft != 0:
        run_name += f'__{args.soft}'
    if args.fl != 'fedavg':
        run_name += f'__m{args.fl}'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_FILE = os.path.join(args.save_path, run_name)
    return run_name, SAVE_FILE
    # return run_name


def get_model_fh(data, model, num_classes=10):
    if data == 'Digits':
        if model in ['digit']:
            from nets.models import DigitModel
            ModelClass = DigitModel
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data in ['DomainNet', 'ImageNet']:
        if model in ['alex']:
            from nets.models import AlexNet
            ModelClass = AlexNet
        elif model == 'wrn':
            ModelClass = WideResNet_Domain

        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'cifar10' or data == 'cifar100':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'tin':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet_Tin
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'stl':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet_stl
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'mnist':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet_MNIST
        elif model == 'vit':
            ModelClass = VisionTransformer
    elif data == 'heterogeneous':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.preresne import resnet18
            ModelClass = resnet18
        elif model == 'allconv':
            ModelClass = AllConvNet(num_classes)
        elif model == 'wrn':
            ModelClass = WideResNet
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError(f"Unknown dataset: {data}")
    return ModelClass


def fed_test(fed, Running_model, val_loaders, verbose, adversary=None):
    mark = 's' if adversary is None else 'r'
    val_acc_list = [None for _ in range(fed.client_num)]
    val_loss_mt = AverageMeter()
    for client_idx in range(fed.client_num):
        # fed.download(running_model, client_idx)
        # Test
        print('client', client_idx, 'validation')
        running_model = Running_model[client_idx]
        val_loss, val_acc = test(running_model, val_loaders[client_idx], loss_fun, device, adversary=adversary)

        # Log
        val_loss_mt.append(val_loss)
        val_acc_list[client_idx] = val_acc
        if verbose > 0:
            print(' {:<19s} Val {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                'User-'+fed.clients[client_idx], mark.upper(), val_loss, mark.upper(), val_acc))
        # wandb.log({
        #     f"{fed.clients[client_idx]} val_{mark}-acc": val_acc,
        # }, commit=False)
    return val_acc_list, val_loss_mt.avg


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    # basic problem setting
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--data', type=str, default='Digits', help='data name')
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--width_scale', type=float, default=1., help='model width scale')
    parser.add_argument('--no_track_stat', action='store_true', help='disable BN tracking')
    parser.add_argument('--no_mask_loss', action='store_true', help='disable masked loss for class'
                                                                    ' niid')
    parser.add_argument('--fl', choices=['fedavg', 'fedprox'], default='fedavg')
    parser.add_argument('--federated', type=int, default=0, help='verbose level: 0 or 1')
    # control
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--verbose', type=int, default=0, help='verbose level: 0 or 1')


    # federated
    Federation.add_argument(parser)
    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_sch', type=str, default='multi_step', help='learning rate schedule')
    parser.add_argument('--opt', type=str.lower, default='sgd', help='optimizer')
    parser.add_argument('--iters', type=int, default=300, help='#iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1, help='#epochs in local train')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # adversarial train
    parser.add_argument('--adv_lmbd', type=float, default=0., help='adv coefficient in [0,1]; default 0 for standard training.')
    parser.add_argument('--test_noise', choices=['none', 'LinfPGD'], default='none')
    # energy reg
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--sample_number', type=int, default=1000)
    parser.add_argument('--select', type=int, default=1)
    parser.add_argument('--select_generator', type=int, default=None)
    parser.add_argument('--sample_from', type=int, default=10000)
    parser.add_argument('--loss_weight', type=float, default=0.1)
    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen_factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Setup for OOD evaluation
    parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
    parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
    parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
    parser.add_argument('--method_name', '-m', type=str, default='cifar10_wrn_baseline_0.1_50_40_1_10000_0.08', help='Method name.')
    # EG and benchmark details
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
    parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
    parser.add_argument('--model_name', default='res', type=str)
    parser.add_argument('--use_external', type=str, default='None', help='None|class|dataset|gen_inverse')
    parser.add_argument('--oe_batch_size', type=int, default=1000, help='ood Batch size.')
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
    parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
    parser.add_argument('--score', type=str, default='energy', help='OE|energy|energy_VOS|crossentropy')
    parser.add_argument('--method', type=str, default='energy', help='OE|energy|crossentropy')
    parser.add_argument('--evaluation_score', type=str, default='energy', help='energy|msp|odin')
    parser.add_argument('--soft', type=float, default=0, help='If >0, use soft label for generator')
    parser.add_argument('--visualization', type=bool, default=False, help='If True, visualize')
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--hard',  action='store_true', help='whether to use hard label in MoE training')
    parser.add_argument('--MoE_centralized_training',  action='store_true', help='whether to use centralized training in MoE training')



    args = parser.parse_args()

    set_seed(args.seed)

    # set experiment files, wandb
    exp_folder = os.path.basename(os.path.splitext(__file__)[0]) + f'_{args.data}'
    run_name, SAVE_FILE = render_run_name(args, exp_folder)


    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    # /////////////////////////////////
    # ///// Fed Dataset and Model /////
    # /////////////////////////////////
    fed = Federation(args.data, args, 0, {})
    dataset_client = {}
    # Data
    if args.data != 'heterogeneous':
        # train_loaders, val_loaders, test_loaders, distributionRate_labels_users = fed.get_data()
        train_loaders, val_loaders, test_loaders, _ = fed.get_data()
        distributionRate_labels_users = [[0 for _ in range(10)] for _ in range(args.pd_nuser)]
        dataset_client = {id : args.data for id in range(args.pd_nuser)}
        print('dataset_client:', dataset_client)
    else:
        train_loaders = []
        val_loaders = []
        test_loaders = []
        # fed_stl = _Federation_HeterDataset('stl', int(args.pd_nuser/3), args)
        # train_loader_stl, val_loader_stl, test_loader_stl, _ = fed_stl.get_data()
        # for tr_loader in train_loader_stl:
        #     train_loaders.append(tr_loader)
        # for va_loader in val_loader_stl:
        #     val_loaders.append(va_loader)
        # for te_loader in test_loader_stl:
        #     test_loaders.append(te_loader)
        # dataset_client = {id: 'stl' for id in range(int(args.pd_nuser / 3))}

        fed_cifar = _Federation_HeterDataset('cifar10', int(args.pd_nuser/2), args)
        train_loader_cifar, val_loader_cifar, test_loader_cifar, _ = fed_cifar.get_data()
        for tr_loader in train_loader_cifar:
            train_loaders.append(tr_loader)
        for va_loader in val_loader_cifar:
            val_loaders.append(va_loader)
        for te_loader in test_loader_cifar:
            test_loaders.append(te_loader)
        for id in range(int(args.pd_nuser/ 2)):
            dataset_client[id] = 'cifar10'

        fed_mnist = _Federation_HeterDataset('mnist', args.pd_nuser-int(args.pd_nuser/2), args)
        train_loader_mnist, val_loader_mnist, test_loader_mnist, _ = fed_mnist.get_data()
        for tr_loader in train_loader_mnist:
            train_loaders.append(tr_loader)
        for va_loader in val_loader_mnist:
            val_loaders.append(va_loader)
        for te_loader in test_loader_mnist:
            test_loaders.append(te_loader)
        for id in range(int(args.pd_nuser / 2), args.pd_nuser):
            dataset_client[id] = 'mnist'

        dataset_value = list(set(dataset_client.values()))
        num_dataset = len(dataset_value)

        distributionRate_labels_users = [[0 for _ in range(10)] for _ in range(args.pd_nuser)]
        distributionRate_labels_users_all = [[0 for _ in range(10 * num_dataset)] for _ in range(args.pd_nuser)]
        print('dataset_client:', dataset_client)

        ################################################################################################################
        if args.data == 'heterogeneous':

            train_loaders_MoE_cifar = []
            val_loaders_MoE_cifar = []
            test_loaders_MoE_cifar = []
            fed_cifar = _Federation_HeterDataset_MoE('cifar10', args.pd_nuser, args)
            train_loader_cifar, val_loader_cifar, test_loader_cifar, _ = fed_cifar.get_data()
            for tr_loader in train_loader_cifar:
                train_loaders_MoE_cifar.append(tr_loader)
            for va_loader in val_loader_cifar:
                val_loaders_MoE_cifar.append(va_loader)
            for te_loader in test_loader_cifar:
                test_loaders_MoE_cifar.append(te_loader)

            # train_loaders_MoE_stl = []
            # val_loaders_MoE_stl = []
            # test_loaders_MoE_stl = []
            # fed_stl = _Federation_HeterDataset_MoE('stl', args.pd_nuser, args)
            # train_loader_stl, val_loader_stl, test_loader_stl, _ = fed_stl.get_data()
            # for tr_loader in train_loader_stl:
            #     train_loaders_MoE_stl.append(tr_loader)
            # for va_loader in val_loader_stl:
            #     val_loaders_MoE_stl.append(va_loader)
            # for te_loader in test_loader_stl:
            #     test_loaders_MoE_stl.append(te_loader)

            train_loaders_MoE_mnist = []
            val_loaders_MoE_mnist = []
            test_loaders_MoE_mnist = []
            fed_mnist = _Federation_HeterDataset_MoE('mnist', args.pd_nuser, args)
            train_loader_mnist, val_loader_mnist, test_loader_mnist, _ = fed_mnist.get_data()
            for tr_loader in train_loader_mnist:
                train_loaders_MoE_mnist.append(tr_loader)
            for va_loader in val_loader_mnist:
                val_loaders_MoE_mnist.append(va_loader)
            for te_loader in test_loader_mnist:
                test_loaders_MoE_mnist.append(te_loader)

            trainloader_MoE = []
            valloader_MoE = []
            testloader_MoE = []
            for id_client in range(args.pd_nuser):
                # trainloader_MoE.append(train_loaders_MoE_stl[id_client], [train_loaders_MoE_cifar[id_client], train_loaders_MoE_mnist[id_client]])
                # valloader_MoE.append(val_loaders_MoE_stl[id_client], val_loaders_MoE_cifar[id_client], val_loaders_MoE_mnist[id_client]])
                # testloader_MoE.append(test_loaders_MoE_stl[id_client], test_loaders_MoE_cifar[id_client], test_loaders_MoE_mnist[id_client]])

                trainloader_MoE.append([train_loaders_MoE_cifar[id_client], train_loaders_MoE_mnist[id_client]])
                valloader_MoE.append([val_loaders_MoE_cifar[id_client], val_loaders_MoE_mnist[id_client]])
                testloader_MoE.append([test_loaders_MoE_cifar[id_client], test_loaders_MoE_mnist[id_client]])

        ################################################################################################################

    mean_batch_iters = int(np.mean([len(tl) for tl in train_loaders]))
    print(f"  mean_batch_iters: {mean_batch_iters}")

    # Model

    if args.model != 'heterogeneous':
        if args.model == 'wrn' or args.model == 'allconv':
            ModelClass = get_model_fh(args.data, args.model)
            running_model = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
            global_model = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        elif args.model == 'vit':
            running_model_VisionTransformer = VisionTransformer().to(device)
            running_model = running_model_VisionTransformer
            global_model = running_model_VisionTransformer
        else:
            ModelClass = get_model_fh(args.data, args.model)
            running_model = ModelClass( track_running_stats=not args.no_track_stat, num_classes=fed.num_classes, width_scale=args.width_scale,).to(device)
            global_model = ModelClass( track_running_stats=not args.no_track_stat, num_classes=fed.num_classes, width_scale=args.width_scale,).to(device)

        # if args.model == 'wrn':
        #     user_classifier = linear_classifier(fed.num_classes, args.widen_factor).to(device)
        # elif args.model == 'preresnet18':
        #     user_classifier = linear_classifier(fed.num_classes, model=args.model).to(device)
        # elif args.model == 'alex':
        #     user_classifier = copy.deepcopy(running_model.get_fc())
    else:
        from nets.HeteFL.preresne import resnet18
        ModelClass = resnet18
        running_model_ResNet = ModelClass( track_running_stats=not args.no_track_stat, num_classes=fed.num_classes, width_scale=args.width_scale,).to(device)
        from nets.models import AlexNet
        ModelClass = AlexNet
        running_model_AlexNet = ModelClass( track_running_stats=not args.no_track_stat, num_classes=fed.num_classes, width_scale=args.width_scale,).to(device)

        from models.ViTransformer import VisionTransformer
        running_model_VisionTransformer = VisionTransformer().to(device)

        if args.data in ['DomainNet', 'ImageNet']:
            ModelClass = WideResNet_Domain
            running_model_WideResNet_Domain = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        elif args.data == 'cifar10' or args.data == 'cifar100' or args.data == 'SVHN' or args.data == 'stl':
            ModelClass = WideResNet
            running_model_WideResNet = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        elif args.data == 'mnist':
            ModelClass = WideResNet_MNIST
            running_model_WideResNet_MNIST = WideResNet_MNIST(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        elif args.data == 'tin':
            ModelClass = WideResNet_Tin
            running_model_WideResNet_Tin = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        elif args.data == 'stl':
            ModelClass = WideResNet_stl
            running_model_WideResNet_stl = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
        elif args.data == 'heterogeneous':
            running_model_WideResNet = WideResNet(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
            running_model_WideResNet_Domain = WideResNet_Domain(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate,track_running_stats=not args.no_track_stat).to(device)
            running_model_WideResNet_Tin = WideResNet_Tin(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
            running_model_WideResNet_stl = WideResNet_stl(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
            running_model_WideResNet_MNIST = WideResNet_MNIST(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
    if args.use_external == 'dataset':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
        dataset = datasets.ImageFolder(root='./dataset/coco/train2017', transform=transform)
        train_loader_out = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    user_class = {}
    userlogistic = {}
    weight_energy = {}
    privacy_engine = {}
    Running_model = {}
    num_features = {}
    user_classifier = {}
    # prepare external class set
    for client_idx in range(fed.client_num):
        ## get class for this client
        print('client_idx:', client_idx)
        len_train = len(train_loaders[client_idx])
        label = []
        count = 0
        for batch_id, (data, y) in enumerate(train_loaders[client_idx]):
            count += y.shape[0]
            for j in range(y.shape[0]):
                if y[j] not in label:
                    label.append(y[j])
        user_class[client_idx] = label
        userlogistic[client_idx] = torch.nn.Linear(1, 2)
        userlogistic[client_idx] = userlogistic[client_idx].cuda()
        weight_energy[client_idx] = torch.nn.Linear(len(user_class[client_idx]), 1).cuda()
        torch.nn.init.uniform_(weight_energy[client_idx].weight)
        if args.model != 'heterogeneous':
            if args.model == 'wrn' or args.model == 'allconv':
                Running_model[client_idx] = ModelClass(args.layers, fed.num_classes, args.widen_factor, dropRate=args.droprate, track_running_stats=not args.no_track_stat).to(device)
                if args.model == 'wrn':
                    user_classifier[client_idx] = linear_classifier(fed.num_classes, args.widen_factor).to(device)
                    num_features[client_idx] = 64 * args.widen_factor
            elif args.model == 'vit':
                Running_model[client_idx] = copy.deepcopy(running_model_VisionTransformer)
                user_classifier[client_idx] = copy.deepcopy(Running_model[client_idx].get_fc())
                num_features[client_idx] = 512
            else:
                Running_model[client_idx] = ModelClass(track_running_stats=not args.no_track_stat, num_classes=fed.num_classes,width_scale=args.width_scale,).to(device)
                if args.model == 'preresnet18':
                    user_classifier[client_idx] = linear_classifier(fed.num_classes, model=args.model).to(device)
                    num_features[client_idx] = 512
                elif args.model == 'alex':
                    user_classifier[client_idx] = copy.deepcopy(running_model.get_fc())
                    num_features[client_idx] = int(4096*args.width_scale)
        else:
            if client_idx < int(fed.client_num/3):
                Running_model[client_idx] = copy.deepcopy(running_model_ResNet)
                user_classifier[client_idx] = copy.deepcopy(Running_model[client_idx].get_fc())
                num_features[client_idx] = 512

            elif client_idx < int(fed.client_num - 3):
                if dataset_client[client_idx] == 'stl':
                    Running_model[client_idx] = copy.deepcopy(running_model_WideResNet)
                elif dataset_client[client_idx] == 'mnist':
                    Running_model[client_idx] = copy.deepcopy(running_model_WideResNet_MNIST)
                elif dataset_client[client_idx] == 'cifar10':
                    Running_model[client_idx] = copy.deepcopy(running_model_WideResNet)
                elif dataset_client[client_idx] == 'SVHN':
                    Running_model[client_idx] = copy.deepcopy(running_model_WideResNet)
                elif dataset_client[client_idx] == 'tin':
                    Running_model[client_idx] = copy.deepcopy(running_model_WideResNet_Tin)
                elif dataset_client[client_idx] in ['DomainNet', 'ImageNet']:
                    Running_model[client_idx] = copy.deepcopy(running_model_WideResNet_Domain)
                user_classifier[client_idx] = copy.deepcopy(Running_model[client_idx].get_fc())
                num_features[client_idx] = 64 * args.widen_factor

            # else:
            #     Running_model[client_idx] = copy.deepcopy(running_model_AlexNet)
            #     user_classifier[client_idx] = copy.deepcopy(Running_model[client_idx].get_fc())
            #     num_features[client_idx] = int(4096 * args.width_scale)

            else:
                Running_model[client_idx] = copy.deepcopy(running_model_VisionTransformer)
                user_classifier[client_idx] = copy.deepcopy(Running_model[client_idx].get_fc())
                num_features[client_idx] = 512

    # adversary
    if args.adv_lmbd > 0. or args.test:
        make_adv = AdversaryCreator(args.test_noise if args.test else 'LinfPGD')
        adversary = make_adv(running_model)
    else:
        adversary = None

    # Loss
    if args.pu_nclass > 0 and not args.no_mask_loss:  # niid
        loss_fun = LocalMaskCrossEntropyLoss(fed.num_classes)
    else:
        loss_fun = nn.CrossEntropyLoss()

    # Use running model to init a fed aggregator
    if args.federated == 1:
        fed.make_aggregator(running_model, local_fc=args.local_fc)

    # ////////////////
    # //// Train /////
    # ////////////////
    # LR scheduler
    if args.lr_sch == 'cos':
        lr_sch = CosineAnnealingLR(args.iters, eta_max=args.lr, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step':
        lr_sch = MultiStepLR(args.lr, milestones=[150, 250], gamma=0.1, last_epoch=start_epoch)
    else:
        assert args.lr_sch == 'none', f'Invalid lr_sch: {args.lr_sch}'
        lr_sch = None
    total_iter = torch.zeros((fed.client_num))

    # generator_model = InversGenerator(num_classes=args.num_classes, latent_dim=num_features[0])
    # generator_loss_set = []

    generator_loss_client = [[] for _ in range(args.pd_nuser)]
    generator_model_set = []
    for id in range(args.pd_nuser):
        generator_model = InversGenerator(num_classes=args.num_classes, latent_dim=num_features[id])
        generator_model_set.append(generator_model)

    for a_iter in range(start_epoch, args.iters):
        global_lr = args.lr if lr_sch is None else lr_sch.step()

        ##get global fc
        if args.federated == 1:
            global_fc = fed.get_global_fc()
            global_model.load_state_dict(fed.model_accum.server_state_dict)

        ##train central generator
        if args.use_external == 'gen_inverse':
            if args.federated == 1:
                for id in range(args.pd_nuser):
                    for k, v in user_classifier[id].state_dict().items():
                        if 'fc.weight' in k:
                            user_classifier[id].state_dict()[k].copy_(global_fc[0])
                        if 'fc.bias' in k:
                            user_classifier[id].state_dict()[k].copy_(global_fc[1])
                Central_gen.train_generator(args, user_classifier[0])


        ##train personalized generator
        if args.use_external == 'ours':
            if args.federated == 1:
                for id in range(args.pd_nuser):
                    for k, v in user_classifier[id].state_dict().items():
                        if 'fc.weight' in k:
                            user_classifier[id].state_dict()[k].copy_(global_fc[0])
                        if 'fc.bias' in k:
                            user_classifier[id].state_dict()[k].copy_(global_fc[1])
            if a_iter > 0:
                for id in range(args.pd_nuser):
                    generator_weight, generator_loss = LocalGenerator_update_InversGenerator(generator_model_set[id], mean_numpy[id], covariance_numpy[id], distributionRate_labels_users[id], global_features_vector_COSdistance, global_features_vector_COSdistance_min, global_features_vector_COSdistance_max, dataset_client[id])
                    generator_loss_client[id].append(generator_loss)
                    generator_model_set[id].load_state_dict(generator_weight)

        # ----------- Train Client ---------------
        train_loss_mt = AverageMeter()
        epsilon_mt = AverageMeter()
        best_alpha_mt = AverageMeter()
        print("============ Train epoch {} ============".format(a_iter))

        classifier_COSdiastance = []
        classifier_Eulerdiastance = []
        ouput_logit_client = []
        features_COSdiastance = []
        features_Eulerdiastance= []
        features_clients_set = []
        features_distribution_KLdivergence_set = []
        mean_numpy = [{} for _ in range(args.pd_nuser)]
        covariance_numpy = [{} for _ in range(args.pd_nuser)]

        # for client_idx in range(args.pd_nuser):
        for client_idx in fed.client_sampler.iter():
            start_time = time.process_time()
            if args.federated == 0:
                running_model = Running_model[client_idx]
            else:
                fed.download(running_model, client_idx)

            if args.model != 'heterogeneous':
                #prepare for VOS
                num_classes = fed.num_classes
                data_dim = 128
                if args.model == 'preresnet18':
                    data_dim = 512
                elif args.model == 'vit':
                    data_dim = 512
                elif args.model == 'alex':
                    data_dim = int(4096*args.width_scale)
                data_dict = torch.zeros(num_classes, args.sample_number, data_dim).cuda()
                number_dict = {}
                for i in range(num_classes):
                    number_dict[i] = 0
                eye_matrix = torch.eye(data_dim, device='cuda')
            else:
                num_classes = fed.num_classes
                data_dim = num_features[client_idx]
                data_dict = torch.zeros(num_classes, args.sample_number, data_dim).cuda()
                number_dict = {}
                for i in range(num_classes):
                    number_dict[i] = 0
                eye_matrix = torch.eye(data_dim, device='cuda')

            if isinstance(running_model, VisionTransformer):
                fc_head = running_model.get_fc()
                fc_para = list(fc_head.parameters())
                optimizer_fc = torch.optim.SGD(fc_para, 0.001, momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
                local_para = list(running_model.parameters())
                for i in range(len(fc_para)):
                    for j in range(len(local_para)):
                        if fc_para[i].equal(local_para[j]):
                            local_para.pop(j)
                            break
                optimizer_local = torch.optim.SGD(local_para + list(weight_energy[client_idx].parameters()) + list(userlogistic[client_idx].parameters()), 0.001, momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
                optimizer = torch.optim.SGD(list(running_model.parameters()) + list(weight_energy[client_idx].parameters()) + list(userlogistic[client_idx].parameters()), 0.001, momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
            else:
                fc_head = running_model.get_fc()
                fc_para = list(fc_head.parameters())
                optimizer_fc = torch.optim.SGD(fc_para, global_lr, momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
                local_para = list(running_model.parameters())
                for i in range(len(fc_para)):
                    for j in range(len(local_para)):
                        if fc_para[i].equal(local_para[j]):
                            local_para.pop(j)
                            break
                optimizer_local = torch.optim.SGD(local_para + list(weight_energy[client_idx].parameters()) + list(userlogistic[client_idx].parameters()), global_lr, momentum=state['momentum'],weight_decay=state['decay'], nesterov=True)
                optimizer = torch.optim.SGD(list(running_model.parameters()) + list(weight_energy[client_idx].parameters()) + list(userlogistic[client_idx].parameters()), global_lr, momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

            #train for VOS
            if args.partition_mode != 'uni':
                max_iter = mean_batch_iters * args.wk_iters
            else:
                max_iter = len(train_loaders[client_idx]) * args.wk_iters
            if args.use_external == 'None':
                if (args.fl == 'fedprox') and (a_iter > start_epoch):
                    train_loss, total_iter[client_idx] = VOS_train_prox(user_class[client_idx], args.model,
                                                                   total_iter[client_idx], state, max_iter, global_model,
                                                                   running_model, train_loaders[client_idx],
                                                                   fed.num_classes, number_dict, args.sample_number,
                                                                   args.start_iter, data_dict,
                                                                   eye_matrix, userlogistic[client_idx], optimizer,
                                                                   args.loss_weight, weight_energy[client_idx],
                                                                   args.sample_from, args.select, verbose=args.verbose)
                else:
                    print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
                    train_loss, total_iter[client_idx] = VOS_train(user_class[client_idx], args.model, total_iter[client_idx], state, max_iter, running_model, train_loaders[client_idx], fed.num_classes, number_dict, args.sample_number, args.start_iter, data_dict,
                      eye_matrix, userlogistic[client_idx], optimizer, args.loss_weight,  weight_energy[client_idx], args.sample_from, args.select, verbose=args.verbose)
            elif args.use_external == 'dataset':
                train_loss, total_iter[client_idx] = VOS_train2(args.model, total_iter[client_idx], state, max_iter, running_model, train_loaders[client_idx], train_loader_out, fed.num_classes, number_dict, args.sample_number, args.start_iter, data_dict,
                      eye_matrix, userlogistic[client_idx], optimizer, args.loss_weight,  weight_energy[client_idx], args.sample_from, args.select, verbose=args.verbose)
            elif args.use_external == 'class':
                train_loss, total_iter[client_idx] = VOS_train2(args.model, total_iter[client_idx], state, max_iter,
                                                                running_model, train_loaders[client_idx],
                                                                external_loader[client_idx], fed.num_classes, number_dict,
                                                                args.sample_number, args.start_iter, data_dict,
                                                                eye_matrix, userlogistic[client_idx], optimizer,
                                                                args.loss_weight, weight_energy[client_idx], args.sample_from,
                                                                args.select, verbose=args.verbose)

            elif args.use_external == 'gen_inverse':
                if args.select_generator != None:
                    if (args.fl == 'fedprox') and (a_iter > start_epoch):
                        train_loss, total_iter[client_idx] = topk_inversion_train_prox(num_classes, number_dict, data_dict,
                                                                                  args.sample_number, eye_matrix,
                                                                                  args.sample_from,
                                                                                  Central_gen.generative_model,
                                                                                  total_iter[client_idx],
                                                                                  user_class[client_idx], args.score,
                                                                                  args.m_in,
                                                                                  args.m_out, a_iter,
                                                                                  user_classifier[client_idx],
                                                                                  total_iter[client_idx], state,
                                                                                  max_iter, global_model,
                                                                                  running_model,
                                                                                  train_loaders[client_idx], optimizer,
                                                                                  verbose=args.verbose,
                                                                                  logistic_regression=userlogistic[client_idx],
                                                                                  weight_energy=weight_energy[client_idx],
                                                                                  select=args.select_generator,
                                                                                  soft=args.soft,
                                                                                  optimizer_fc=optimizer_fc,
                                                                                  optimizer_local=optimizer_local)
                    else:
                        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                        train_loss, total_iter[client_idx], client_logit, feature, features_set, running_model = topk_inversion_train(args, num_features[client_idx], num_classes, number_dict, data_dict,
                                                                                                                  args.sample_number, eye_matrix,
                                                                                                                  args.sample_from,
                                                                                                                  Central_gen.generative_model,
                                                                                                                  total_iter[client_idx],
                                                                                                                  user_class[client_idx], args.score,
                                                                                                                  args.m_in,
                                                                                                                  args.m_out, a_iter,
                                                                                                                  user_classifier[client_idx],
                                                                                                                  total_iter[client_idx], state, max_iter,
                                                                                                                  running_model,
                                                                                                                  train_loaders[client_idx], optimizer,
                                                                                                                  verbose=args.verbose,
                                                                                                                  logistic_regression=userlogistic[client_idx],
                                                                                                                  weight_energy=weight_energy[client_idx],
                                                                                                                  select=args.select_generator,
                                                                                                                  soft=args.soft, optimizer_fc=optimizer_fc, optimizer_local=optimizer_local)
                        Running_model[client_idx] = running_model
                        user_classifier[client_idx] = running_model.get_fc()
                        # print('student_features of', client_idx, ':', feature)
                        # classifier_COSdiastance.append(classifier_COSmeasure(local_weights[client_idx], distributionRate_labels_users[client_idx]))  # 上传所有classifier vector之间的角度（不管vector是否已知）
                        # classifier_Eulerdiastance.append(classifier_Eulermeasure(local_weights[client_idx],distributionRate_labels_users[client_idx]))  # 上传所有classifier vector之间的欧氏距离（不管vector是否已知）
                        # ouput_logit_client.append(client_logit)

                        for i_class in range(len(features_set)):
                            if features_set[i_class] != []:
                                covariance_numpy[client_idx][i_class] = covMat(features_set[i_class])  # 输出为numpy
                                mean_numpy[client_idx][i_class] = mean_calcualation(features_set[i_class])  # 输出为numpy
                        # print('mean_numpy', client_idx,':', mean_numpy[client_idx])

                        distributionRate_labels_users[client_idx] = [0 for _ in range(10)]
                        for key in mean_numpy[client_idx].keys():
                            distributionRate_labels_users[client_idx][key] = 1

                        features_COSdiastance.append(feature_COSmeasure(feature, distributionRate_labels_users[client_idx]))
                        # features_Eulerdiastance.append(feature_Eulermeasure(feature, distributionRate_labels_users[client_idx]))
                        features_clients_set.append(features_set)

                        # features_distribution_KLdivergence = KL_calculation(mean_numpy[client_idx], covariance_numpy[client_idx], feature_unprunSet[client_idx])
                        # features_distribution_KLdivergence_set.append(features_distribution_KLdivergence)


                if args.score == 'energy_VOS':
                    train_loss, total_iter[client_idx] = inversion_train(Central_gen.generative_model,
                                                                         total_iter[client_idx],
                                                                         user_class[client_idx], args.score,
                                                                         args.m_in,
                                                                         args.m_out, a_iter,
                                                                         user_classifier[client_idx], args.oe_batch_size,
                                                                         total_iter[client_idx], state, max_iter,
                                                                         running_model,
                                                                         train_loaders[client_idx], optimizer,
                                                                         verbose=args.verbose,
                                                                         logistic_regression=userlogistic[client_idx],
                                                                         weight_energy=weight_energy[client_idx])


            elif args.use_external == 'ours':
                if args.select_generator != None:
                    if (args.fl == 'fedprox') and (a_iter > start_epoch):
                        train_loss, total_iter[client_idx] = topk_inversion_train_prox(num_classes, number_dict, data_dict,
                                                                                  args.sample_number, eye_matrix,
                                                                                  args.sample_from,
                                                                                  generator_model_set[client_idx],
                                                                                  total_iter[client_idx],
                                                                                  user_class[client_idx], args.score,
                                                                                  args.m_in,
                                                                                  args.m_out, a_iter,
                                                                                  user_classifier[client_idx],
                                                                                  total_iter[client_idx], state,
                                                                                  max_iter, global_model,
                                                                                  running_model,
                                                                                  train_loaders[client_idx], optimizer,
                                                                                  verbose=args.verbose,
                                                                                  logistic_regression=userlogistic[client_idx],
                                                                                  weight_energy=weight_energy[client_idx],
                                                                                  select=args.select_generator,
                                                                                  soft=args.soft,
                                                                                  optimizer_fc=optimizer_fc,
                                                                                  optimizer_local=optimizer_local)
                    else:
                        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                        train_loss, total_iter[client_idx], client_logit, feature, features_set, running_model = topk_inversion_train(args, num_features[client_idx], num_classes, number_dict, data_dict,
                                                                                                                  args.sample_number, eye_matrix,
                                                                                                                  args.sample_from,
                                                                                                                  # Central_gen.generative_model,
                                                                                                                  # generator_model,
                                                                                                                  generator_model_set[client_idx],
                                                                                                                  total_iter[client_idx],
                                                                                                                  user_class[client_idx], args.score,
                                                                                                                  args.m_in,
                                                                                                                  args.m_out, a_iter,
                                                                                                                  user_classifier[client_idx],
                                                                                                                  total_iter[client_idx], state, max_iter,
                                                                                                                  running_model,
                                                                                                                  train_loaders[client_idx], optimizer,
                                                                                                                  verbose=args.verbose,
                                                                                                                  logistic_regression=userlogistic[client_idx],
                                                                                                                  weight_energy=weight_energy[client_idx],
                                                                                                                  select=args.select_generator,
                                                                                                                  soft=args.soft, optimizer_fc=optimizer_fc, optimizer_local=optimizer_local)
                        Running_model[client_idx] = running_model
                        user_classifier[client_idx] = running_model.get_fc()
                        # print('student_features of', client_idx, ':', feature)
                        # classifier_COSdiastance.append(classifier_COSmeasure(local_weights[client_idx], distributionRate_labels_users[client_idx]))  # 上传所有classifier vector之间的角度（不管vector是否已知）
                        # classifier_Eulerdiastance.append(classifier_Eulermeasure(local_weights[client_idx],distributionRate_labels_users[client_idx]))  # 上传所有classifier vector之间的欧氏距离（不管vector是否已知）
                        # ouput_logit_client.append(client_logit)

                        for i_class in range(len(features_set)):
                            if features_set[i_class] != []:
                                covariance_numpy[client_idx][i_class] = covMat(features_set[i_class])  # 输出为numpy
                                mean_numpy[client_idx][i_class] = mean_calcualation(features_set[i_class])  # 输出为numpy
                        # print('mean_numpy', client_idx,':', mean_numpy[client_idx])

                        distributionRate_labels_users[client_idx] = [0 for _ in range(10)]
                        for key in mean_numpy[client_idx].keys():
                            distributionRate_labels_users[client_idx][key] = 1

                        features_COSdiastance.append(feature_COSmeasure(feature, distributionRate_labels_users[client_idx]))
                        # features_Eulerdiastance.append(feature_Eulermeasure(feature, distributionRate_labels_users[client_idx]))
                        features_clients_set.append(features_set)

                        # features_distribution_KLdivergence = KL_calculation(mean_numpy[client_idx], covariance_numpy[client_idx], feature_unprunSet[client_idx])
                        # features_distribution_KLdivergence_set.append(features_distribution_KLdivergence)


                if args.score == 'energy_VOS':
                    train_loss, total_iter[client_idx] = inversion_train(# generator_model,
                                                                         generator_model_set[client_idx],
                                                                         total_iter[client_idx],
                                                                         user_class[client_idx], args.score,
                                                                         args.m_in,
                                                                         args.m_out, a_iter,
                                                                         user_classifier[client_idx], args.oe_batch_size,
                                                                         total_iter[client_idx], state, max_iter,
                                                                         running_model,
                                                                         train_loaders[client_idx], optimizer,
                                                                         verbose=args.verbose,
                                                                         logistic_regression=userlogistic[client_idx],
                                                                         weight_energy=weight_energy[client_idx])

            # Upload
            if args.federated == 1:
                fed.upload(running_model, client_idx)

            # Log
            client_name = fed.clients[client_idx]
            elapsed = time.process_time() - start_time
            # wandb.log({f'{client_name}_train_elapsed': elapsed}, commit=False)
            train_elapsed[client_idx].append(elapsed)

            train_loss_mt.append(train_loss)


            print(f' User-{client_name:<10s} Train | Loss: {train_loss:.4f} |'
                  f' Elapsed: {elapsed:.2f} s')

            # wandb.log({
            #     f"{client_name} train_loss": train_loss,
            # }, commit=False)
        if args.use_external == 'ours':
            # global_classifier_vector_COSdistance, global_classifier_vector_COSdistance_min, global_classifier_vector_COSdistance_max = global_classifier_aggregation(classifier_COSdiastance)  # 聚合计算每个用户的分类器的向量余弦相似度
            # global_classifier_vector_Eulerdistance, global_classifier_vector_Eulerdistance_min, global_classifier_vector_Eulerdistance_max = global_classifier_aggregation(classifier_Eulerdiastance)  # 聚合计算每个用户的分类器的向量距离
            # global_output_logit = global_logit_aggregation(ouput_logit_client)

            global_features_vector_COSdistance, global_features_vector_COSdistance_min, global_features_vector_COSdistance_max = global_classifier_aggregation(features_COSdiastance, dataset_client)  # 聚合计算每个用户的倒数第二层输出feature的向量余弦相似度
            # global_features_vector_Eulerdistance, global_features_vector_Eulerdistance_min, global_features_vector_Eulerdistance_max = global_classifier_aggregation(features_Eulerdiastance)  # 聚合计算每个用户的倒数第二层输出feature的向量距离

            # global_features_distribution_KLdivergence = global_classifier_aggregation(features_distribution_KLdivergence_set)  # 聚合计算每个用户的倒数第二层输出feature分布的KL散度

        # Use accumulated model to update server model
        if args.federated == 1:
            fed.aggregate()

        # ----------- Validation ---------------
        # val_acc_list, val_loss = fed_test(fed, running_model, val_loaders, args.verbose)
        # val_acc_list, val_loss = fed_test(fed, Running_model, val_loaders, args.verbose)
        val_acc_list, val_loss = fed_test(fed, Running_model, test_loaders, args.verbose)
        if args.adv_lmbd > 0:
            print(f' Avg Val SAcc {np.mean(val_acc_list) * 100:.2f}%')
            # wandb.log({'val_sacc': np.mean(val_acc_list)}, commit=False)
            val_racc_list, val_rloss = fed_test(fed, running_model, val_loaders, args.verbose, adversary=adversary)
            print(f' Avg Val RAcc {np.mean(val_racc_list) * 100:.2f}%')
            # wandb.log({'val_racc': np.mean(val_racc_list)}, commit=False)

            val_acc_list = [(1 - args.adv_lmbd) * sa_ + args.adv_lmbd * ra_
                            for sa_, ra_ in zip(val_acc_list, val_racc_list)]
            val_loss = (1 - args.adv_lmbd) * val_loss + args.adv_lmbd * val_rloss

        # Log averaged
        print(f' [Overall] Train Loss {train_loss_mt.avg:.4f} '
              f' | Val Acc {np.mean(val_acc_list) * 100:.2f}%')

        # # ----------- Save checkpoint -----------
        # if np.mean(val_acc_list) > np.mean(best_acc):
        #     best_epoch = a_iter
        #     for client_idx in range(fed.client_num):
        #         best_acc[client_idx] = val_acc_list[client_idx]
        #         if args.verbose > 0:
        #             print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
        #                 fed.clients[client_idx], best_epoch, best_acc[client_idx]))
        #     print(' [Best Val] Acc {:.4f}'.format(np.mean(val_acc_list)))
        #
        #     # Save
        #     print(f' Saving the local and server checkpoint to {SAVE_FILE}')
        #     save_dict = {
        #         'server_model': fed.model_accum.state_dict(),
        #         'best_epoch': best_epoch,
        #         'best_acc': best_acc,
        #         'a_iter': a_iter,
        #         'all_domains': fed.all_domains,
        #         'train_elapsed': train_elapsed,
        #     }
        #     print(f"  Test model: {args.model}x{args.width_scale}"
        #           + ('' if args.test_noise == 'none' else f'with {args.test_noise} noise'))

    ####################################################################################################################
    # Test on clients
    # Test on clients
    if args.data == 'cifar10':
        # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
        # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
        dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "MNIST", "FashionMNIST"]
    elif args.data == 'stl':
        # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
        # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
        dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "MNIST", "FashionMNIST"]
    elif args.data == 'mnist':
        # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
        # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
        dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "FashionMNIST", "CIFAR10"]
    elif args.data == 'FashionMNIST':
        # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
        # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
        dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "MNIST", "CIFAR10"]
    elif args.data == 'SVHN':
        dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "FashionMNIST", "CIFAR10"]
    else:
        # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN"]
        dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "FashionMNIST"]
    auroc_mt, aupr_mt, test_acc_mt = AverageMeter(), AverageMeter(), AverageMeter()
    auroc_detail, aupr_detail = {}, {}
    for i in range(len(dataset_name)):
        auroc_detail[dataset_name[i]] = AverageMeter()
        aupr_detail[dataset_name[i]] = AverageMeter()
    if args.data == 'tin':
        test_loaders = val_loaders
    for test_idx, test_loader in enumerate(test_loaders):
        if args.federated == 1:
            fed.download(running_model, test_idx)
        else:
            running_model = Running_model[test_idx]

        # _, test_acc = test(running_model, test_loader, loss_fun, device, adversary=adversary)
        _, test_acc = test(Running_model[test_idx], test_loader, loss_fun, device, adversary=adversary)
        print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[test_idx], test_acc))
        auroc, aupr, auroc_dict, aupr_dict = VOS_evaluate(args, args.out_as_pos, args.num_to_avg, args.use_xent,
                                                          args.method_name, args.evaluation_score,
                                                          args.test_batch, args.T, args.noise,
                                                          Running_model[test_idx],
                                                          test_loader, train_loaders[test_idx],
                                                          user_class[test_idx], data_name=args.data,
                                                          m_name=args.use_external, client_id=test_idx)

        test_acc_mt.append(test_acc)

        auroc_mt.append(auroc)
        aupr_mt.append(aupr)

        for i in range(len(dataset_name)):
            # auroc_detail[dataset_name[i]].append(auroc_list[i])
            # aupr_detail[dataset_name[i]].append(aupr_list[i])

            auroc_detail[dataset_name[i]].append(auroc_dict[dataset_name[i]])
            aupr_detail[dataset_name[i]].append(aupr_dict[dataset_name[i]])
    print(f"\n Test auroc: {auroc_mt}")
    print(f"\n Test aupr: {aupr_mt}")
    print(f"\n Test Acc: {test_acc_mt}")

    print(f"\n Average Test auroc: {auroc_mt.avg}")
    print(f"\n Average Test aupr: {aupr_mt.avg}")
    print(f"\n Average Test Acc: {test_acc_mt.avg}")

    print("Show detail:")
    for i in range(len(dataset_name)):
        print("{} detection".format(dataset_name[i]))
        print("auroc: {}, aupr: {}".format(auroc_detail[dataset_name[i]].avg, aupr_detail[dataset_name[i]].avg))
        # print("auroc: {}, aupr: {}".format(np.mean(auroc_detail[dataset_name[i]]), np.mean(aupr_detail[dataset_name[i]])))

