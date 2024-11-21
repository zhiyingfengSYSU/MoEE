import copy
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

def classifier_COSmeasure(args, local_weights, distributionRate_labels_users):
    key_model = []
    for key in local_weights.keys():
        key_model.append(key)
    if 'weight' in key_model[-1]:
        distance_classifier = torch.empty(len(local_weights[key_model[-1]]), len(local_weights[key_model[-1]]))
        distance_classifier.to(args.gpu)
        for i in range(len(local_weights[key_model[-1]])):
            for j in range(len(local_weights[key_model[-1]])):
                if distributionRate_labels_users[i] != 0 and distributionRate_labels_users[j] != 0:
                    distance_classifier[i, j] = F.cosine_similarity(local_weights[key_model[-1]][i], local_weights[key_model[-1]][j], dim=0)
                    #distance_classifier[i, j] = torch.dot(local_weights[key_model[-1]][i], local_weights[key_model[-1]][j])
                else:
                    distance_classifier[i, j] = -100
    else:
        distance_classifier = torch.empty(len(local_weights[key_model[-2]]), len(local_weights[key_model[-2]]))
        distance_classifier.to(args.gpu)
        for i in range(len(local_weights[key_model[-2]])):
            for j in range(len(local_weights[key_model[-2]])):
                if distributionRate_labels_users[i] !=0 and distributionRate_labels_users[j] !=0:
                    distance_classifier[i, j] = F.cosine_similarity(local_weights[key_model[-2]][i], local_weights[key_model[-2]][j], dim=0)
                    #distance_classifier[i, j] = torch.dot(local_weights[key_model[-2]][i],local_weights[key_model[-2]][j])
                else:
                    distance_classifier[i, j] = -100

    return distance_classifier

def classifier_Eulermeasure(args, local_weights, distributionRate_labels_users):
    key_model = []
    for key in local_weights.keys():
        key_model.append(key)
    if 'weight' in key_model[-1]:
        distance_classifier = torch.empty(len(local_weights[key_model[-1]]), len(local_weights[key_model[-1]]))
        distance_classifier.to(args.gpu)
        for i in range(len(local_weights[key_model[-1]])):
            for j in range(len(local_weights[key_model[-1]])):
                if distributionRate_labels_users[i] != 0 and distributionRate_labels_users[j] != 0:
                    distance_classifier[i, j] = F.pairwise_distance(local_weights[key_model[-1]][i], local_weights[key_model[-1]][j], p=2) / len(local_weights[key_model[-1]][i])
                else:
                    distance_classifier[i, j] = -100
    else:
        distance_classifier = torch.empty(len(local_weights[key_model[-2]]), len(local_weights[key_model[-2]]))
        distance_classifier.to(args.gpu)
        for i in range(len(local_weights[key_model[-2]])):
            for j in range(len(local_weights[key_model[-2]])):
                if distributionRate_labels_users[i] != 0 and distributionRate_labels_users[j] != 0:
                    distance_classifier[i, j] = F.pairwise_distance(local_weights[key_model[-2]][i], local_weights[key_model[-2]][j], p=2) / len(local_weights[key_model[-2]][i])
                else:
                    distance_classifier[i, j] = -100

    return distance_classifier


def feature_COSmeasure(client_feature, distributionRate_labels_users):

    distance_feature = torch.empty(len(client_feature), len(client_feature))

    distance_feature.to('cuda:0')
    for i in range(len(client_feature)):
        for j in range(len(client_feature)):
            if distributionRate_labels_users[i] != 0 and distributionRate_labels_users[j] != 0:
                distance_feature[i, j] = F.cosine_similarity(client_feature[i], client_feature[j], dim=0)
                # distance_classifier[i, j] = torch.dot(local_weights[key_model[-1]][i], local_weights[key_model[-1]][j])
            else:
                distance_feature[i, j] = -100
    return distance_feature


def feature_Eulermeasure(client_feature, distributionRate_labels_users):

    distance_feature= torch.empty(len(client_feature), len(client_feature))
    distance_feature.to('cuda:0')
    for i in range(len(client_feature)):
        for j in range(len(client_feature)):
            if distributionRate_labels_users[i] != 0 and distributionRate_labels_users[j] != 0:
                distance_feature[i, j] = F.pairwise_distance(client_feature[i], client_feature[j], p=2) / len(client_feature[i])
            else:
                distance_feature[i, j] = -100

    return distance_feature



def calcCov(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    n = len(x)

    return sum((x - mean_x) * (y - mean_y)) / (n-1)

# calculates the Covariance matrix
def covMat(data):
    # get the rows and cols
    rows = len(data)  # 样本个数
    cols = len(data[0])  # 属性维度

    data_array = np.random.rand(rows * cols).reshape(rows, cols)

    for i in range(rows):
        data_array[i] = data[i].to('cpu').numpy()

    # the covariance matroix has a shape of n_features x n_features
    # n_featurs  = cols - 1 (not including the target column)
    cov_mat = np.zeros((cols, cols))

    for i in range(cols):
        for j in range(cols):
            # store the value in the matrix
            cov_mat[i][j] = calcCov(data_array[:, i], data_array[:, j])
            # temp = [data_array[:, i], data_array[:, j]]
            # cov_mat[i][j] = np.cov(temp)

    return cov_mat


def mean_calcualation(feature_set):
    sum = copy.deepcopy(feature_set[0])
    for id in range(1, len(feature_set)):
        sum = sum + feature_set[id]
    sum = sum / len(feature_set)
    return sum.to('cpu').numpy()


def global_classifier_aggregation(classifier_diastance, dataset_client):

    distance_avg = {}
    distance_min = {}
    distance_max = {}
    dataset_value = list(set(dataset_client.values()))

    num_dataset = Counter(dataset_client.values())

    for key in dataset_value:
        distance_avg[key] = torch.zeros_like(classifier_diastance[0])
        distance_min[key] = torch.zeros_like(classifier_diastance[0])
        distance_max[key] = torch.zeros_like(classifier_diastance[0])

    for id in range(len(classifier_diastance)):
        dataset_key = dataset_client[id]
        distance_avg[dataset_key] = distance_avg[dataset_key] + classifier_diastance[id]
#
    num_invalid = {}
    for key in dataset_value:
        num_invalid[key] = [[0 for _ in range(10)] for _ in range(10)]
    for id in range(len(classifier_diastance)):
        dataset_key = dataset_client[id]
        for j in range(10):
            for k in range(10):
                if classifier_diastance[id][j][k] == -100:
                    num_invalid[dataset_key][j][k] = num_invalid[dataset_key][j][k] + 1
#
    for key in dataset_value:
        for i in range(10):
            for j in range(10):
                distance_avg[key][i][j] = distance_avg[key][i][j] + 100 * num_invalid[key][i][j]
                if num_dataset[key] - num_invalid[key][i][j] != 0:
                    distance_avg[key][i][j] = distance_avg[key][i][j] / (len(classifier_diastance) - num_invalid[key][i][j])
                else:
                    distance_avg[key][i][j] = -100

    for i in range(10):
        for j in range(10):
            Candidate_set = {}
            for key in dataset_value:
                Candidate_set[key] = []
            for id in range(len(classifier_diastance)):
                dataset_key = dataset_client[id]
                if classifier_diastance[id][i][j] != -100:
                    Candidate_set[dataset_key].append(classifier_diastance[id][i][j])

            for key in dataset_value:
                if Candidate_set[key] != []:
                    distance_min[key][i, j] = min(Candidate_set[key])
                    distance_max[key][i, j] = max(Candidate_set[key])
                else:
                    distance_min[key][i, j] = -100
                    distance_max[key][i, j] = -100

    return distance_avg, distance_min, distance_max

#