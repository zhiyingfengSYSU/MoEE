import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time

from models.wrn_virtual import oodGenerator, InversGenerator
MIN_SAMPLES_PER_LABEL=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#central generator using crossentropy
class CentralGen:
    def __init__(self, args, local_iter, num_class, model='wrn'):
        self.num_class = num_class
        self.local_iter = local_iter
        self.generative_model = InversGenerator(num_class, args.widen_factor, width_scale=args.width_scale, model=model).to(device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=1e-4, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=1e-2, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)

    def train_generator(self, args, user_classifier, epoches=1):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param net: local training model
        :param train_loader: local training loader (ID data)
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return angeneratorything.
        """
        #self.generative_regularizer.train()
        DIVERSITY_LOSS = 0

        def update_generator_(args, diversity_loss):
            self.generative_model.train()
            for i in range(self.local_iter):
                self.generative_optimizer.zero_grad()
                oodclass = [i for i in range(self.num_class)]
                oody = np.random.choice(oodclass, args.oe_batch_size)
                oody = torch.LongTensor(oody).cuda()
                oodz = self.generative_model(oody)
                logit_given_gen = user_classifier(oodz)
                if args.method == 'crossentropy':
                    diversity_loss = F.cross_entropy(logit_given_gen, oody)  # encourage different outputs

                diversity_loss.backward()
                self.generative_optimizer.step()
            return diversity_loss
        for i in range(epoches):
            DIVERSITY_LOSS=update_generator_(args, DIVERSITY_LOSS)
        info="Generator:  Diversity Loss = {:.4f}, ".format(DIVERSITY_LOSS)
        print(info)
        self.generative_lr_scheduler.step()



class CosineThresholdLoss(nn.Module):
    def __init__(self, threshold_target, threshold_min, threshold_max):
        super(CosineThresholdLoss, self).__init__()
        self.threshold_target = threshold_target
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def forward(self, output, target):
        cosine_sim = F.cosine_similarity(output, target)
        # print('cosine_sim:', cosine_sim)
        if cosine_sim > 0.9:
            loss = cosine_sim - 0.9
        elif cosine_sim > self.threshold_target:
            # loss = torch.Tensor([0]).to('cuda:0')
            loss = 0
        elif cosine_sim > self.threshold_min:
            # loss = torch.Tensor([0]).to('cuda:0')
            loss = 0
        else:
            loss = self.threshold_min - cosine_sim
        return loss

def generator_update_InversGenerator(args, model, mean_numpy, covariance_numpy, distributionRate_labels_users, global_features_vector_COSdistance, global_features_vector_COSdistance_min, global_features_vector_COSdistance_max, global_model):

    ###### 数据预处理，转化为矩阵计算 ######
    regular_con = 0.1   #在协方差矩阵的对角线元素加上一个小值，从而使协方差矩阵避免病态

    mean_matrix = copy.deepcopy(mean_numpy)
    for id in range(len(mean_numpy)):
        for i_class in mean_numpy[id].keys():
            mean_matrix[id][i_class] = np.matrix(mean_numpy[id][i_class])
    # print('mean_matrix:', mean_matrix)

    covariance_matrix = copy.deepcopy(covariance_numpy)
    for id in range(len(covariance_numpy)):
        for i_class in covariance_numpy[id].keys():
            covariance_matrix[id][i_class] = np.matrix(covariance_numpy[id][i_class])

    for id in range(len(covariance_numpy)):
        for i_class in covariance_numpy[id].keys():
            for i in range(len(covariance_matrix[id][i_class])):
                covariance_matrix[id][i_class][i, i] = covariance_matrix[id][i_class][i, i] + regular_con

    covariance_matrixInverse = copy.deepcopy(covariance_matrix)
    for id in range(len(covariance_matrix)):
        for i_class in covariance_matrix[id].keys():
            covariance_matrixInverse[id][i_class] = covariance_matrix[id][i_class].I

    covariance_matrixDeterminant = copy.deepcopy(covariance_matrix)
    for id in range(len(covariance_matrix)):
        for i_class in covariance_matrix[id].keys():
            covariance_matrixDeterminant[id][i_class] = np.linalg.det(covariance_matrix[id][i_class])

    # mean_matrix_Augmented = Augmented_MeanMatrix(mean_matrix, feature_unprunSet, len_feature)
    # covariance_matrixInverse_Augmented = Augmented_CovarianceMatrix(covariance_matrixInverse, feature_unprunSet, len_feature)

    model.train()
    model.to('cuda:0')
    global_model.to('cuda:0')

    epoch_loss = []
    cosine_similarity = nn.CosineSimilarity(dim=1)

    for iter in range(100):
        batch_loss = []
        hyper_lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_lr, weight_decay=1e-4)

        # oodclass = [i for i in range(10) if i not in user_class]
        oodclass = [i for i in range(10)]
        oody = np.random.choice(oodclass, 1)
        oody = torch.LongTensor(oody).cuda()
        virtual_feature = model(oody)
        # features_noise = copy.deepcopy(oodz)
        # virtual_feature = copy.deepcopy(oodz)
        loss = torch.Tensor([0])
        loss = loss.to('cuda:0')
        # for i in range(len(virtual_feature)):
        # outputs = F.log_softmax(global_model.linear(virtual_feature[i]))
        # criterion_NLLLoss = nn.NLLLoss().to('cuda:0')
        # true_label = torch.tensor(oody[i]).to('cuda:0')
        # loss = criterion_NLLLoss(outputs, true_label)

        # loss_fn = nn.MSELoss()
        for id in range(args.pd_nuser):
            for i_class in range(10):
                if distributionRate_labels_users[id][i_class] != 0 and global_features_vector_COSdistance[i_class, int(oody)] != -100:
                    criterion_cos = CosineThresholdLoss( threshold_target=global_features_vector_COSdistance[i_class, int(oody)], threshold_min=global_features_vector_COSdistance_min[i_class, int(oody)], threshold_max=global_features_vector_COSdistance_max[i_class, int(oody)] )
                    # print(virtual_feature[i])
                    # print(mean_matrix[id][i_class])
                    loss2 = criterion_cos(virtual_feature[i],  torch.Tensor(mean_matrix[id][i_class]).to('cuda:0'))
                    print('loss2:', loss2)
                    if loss2 != 0:
                        loss2.backward(retain_graph=True)
                        optimizer.step()
                    # if loss2 != 0:
                    #     loss = loss + loss2

            # if loss != 0:
            #     # loss = loss.to('cuda:0')
            #     # loss.backward(retain_graph=True)
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #     batch_loss.append(loss.item())
            # del input, virtual_feature, loss  # 内存释放
            # torch.cuda.empty_cache()
        print('generator_ep:', iter, 'epoch_loss:', sum(batch_loss) / (len(batch_loss) + 0.001) )
        epoch_loss.append( sum(batch_loss) / (len(batch_loss) + 0.001))

    return model.state_dict(), epoch_loss




def LocalGenerator_update_InversGenerator(model, mean_numpy, covariance_numpy, distributionRate_labels_users,  global_features_vector_COSdistance, global_features_vector_COSdistance_min, global_features_vector_COSdistance_max, dataset_client):
    #######################################################################
    class CosineThresholdLoss(nn.Module):
        def __init__(self, threshold_target, threshold_min, threshold_max):
            super(CosineThresholdLoss, self).__init__()
            self.threshold_target = threshold_target
            self.threshold_min = threshold_min
            self.threshold_max = threshold_max

        def forward(self, output, target):
            cosine_sim = F.cosine_similarity(output, target)
            if cosine_sim > 0.9:
                loss = cosine_sim - 0.9
            elif cosine_sim > self.threshold_target:
                loss = 0
            elif cosine_sim > self.threshold_min:
                loss = 0
            else:
                loss = self.threshold_min - cosine_sim
            return loss

    global_features_vector_COSdistance = global_features_vector_COSdistance[dataset_client]
    global_features_vector_COSdistance_min = global_features_vector_COSdistance_min[dataset_client]
    global_features_vector_COSdistance_max = global_features_vector_COSdistance_max[dataset_client]


    model.train()
    model.to('cuda:0')

    epoch_loss = []
    cosine_similarity = nn.CosineSimilarity(dim=1)

    user_class = []
    for i in range(len(distributionRate_labels_users)):
        if distributionRate_labels_users[i] != 0:
            user_class.append(i)

    for iter in range(100):
        batch_loss = []
        hyper_lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_lr, weight_decay=1e-4)

        oodclass = [i for i in range(10) if i not in user_class]
        oody = np.random.choice(oodclass, 2)
        oody = torch.LongTensor(oody).cuda()
        virtual_feature = model(oody)
        # virtual_feature = copy.deepcopy(oodz)


        # outputs = F.log_softmax(local_model.linear(virtual_feature[i]))
        # criterion_NLLLoss = nn.NLLLoss().to('cuda:0')
        # true_label = oody[i].clone().detach().to('cuda:0')
        # loss = criterion_NLLLoss(outputs, true_label)
        #
        # loss_MSELoss = nn.MSELoss().to('cuda:0')

        # if distributionRate_labels_users[int(oody[i])] != 0:
        #     loss1 = loss_MSELoss(virtual_feature[i], torch.Tensor(mean_numpy[int(oody[i].clone().detach())]).to('cuda:0'))
        #     loss = loss + loss1
        # else:
        loss = 0
        for i_class in range(10):
            if distributionRate_labels_users[i_class] != 0 and global_features_vector_COSdistance[i_class, int(oody[0])] != -100:
                # loss2 = (torch.cosine_similarity(virtual_feature[0], torch.Tensor(mean_numpy[i_class]).to('cuda:0'), dim=1) - global_features_vector_COSdistance[i_class, int(oody[0])]) ** 2
                # print('virtual_feature[0]:', virtual_feature[0])
                # print('torch.Tensor(mean_numpy[i_class]):', torch.Tensor(mean_numpy[i_class]))

                loss2 = (nn.functional.cosine_similarity(virtual_feature[0].unsqueeze(0), torch.Tensor(mean_numpy[i_class]).unsqueeze(0).to('cuda:0'), dim=1) - global_features_vector_COSdistance[i_class, int(oody[0])]) ** 2

                # loss2 = (cosine_similarity(virtual_feature[i], torch.Tensor(mean_numpy[i_class]).to('cuda:0')) - global_features_vector_COSdistance[i_class, int(oody[i])]) ** 2
                # loss2 = (cosine_similarity(virtual_feature[i], torch.Tensor(mean_numpy[i_class]).to('cuda:0')) - global_features_vector_COSdistance[i_class, int(oody[i])]).pow(2)

                # criterion_CosineThresholdLoss = CosineThresholdLoss(threshold_target = global_features_vector_COSdistance[i_class, int(oody[i])], threshold_min = global_features_vector_COSdistance_min[i_class, int(oody[i])], threshold_max = global_features_vector_COSdistance_max[i_class, int(oody[i])] )
                # loss2 = criterion_CosineThresholdLoss(virtual_feature[i].unsqueeze(0), torch.Tensor(mean_numpy[i_class]).unsqueeze(0).to('cuda:0'))
                loss = loss + loss2

                # criterion_MSELoss = nn.MSELoss().to('cuda:0')
                # loss3 = (criterion_MSELoss(virtual_feature[i], torch.Tensor(mean_numpy[i_class]).to('cuda:0')) - global_features_vector_Eulerdistance[i_class][int(oody[i])]) ** 2
                # print('loss2:', loss2)
                # print('loss3:', loss3)
                # loss = loss + loss2 + loss3

        if loss != 0:
            loss.backward(retain_graph=True)
            optimizer.step()
            batch_loss.append(loss.item())
            # del input, virtual_feature, loss  # 内存释放
            torch.cuda.empty_cache()
        # print('generator_ep:', iter, 'epoch_loss:', sum(batch_loss) / (len(batch_loss) + 0.001) )
        epoch_loss.append( sum(batch_loss) / (len(batch_loss) + 0.001) )

    return model.state_dict(), epoch_loss


