
import pulp
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pulp import PULP_CBC_CMD

def selection(distributionRate_labels_users, dataset_client):


    dataset_value = list(set(dataset_client.values()))
    num_users = len(distributionRate_labels_users)

    user_id_dict = {}
    for key in dataset_value:
        user_id_dict[key] = []
    for id in range(num_users):
        user_id_dict[dataset_client[id]].append(id)

    num_dataset = len(dataset_value)
    selected_client = {}
    temp = 0
    for key in dataset_value:
        x=[]
        for id in range(num_users):
            if dataset_client[id] == key:
                x.append(distributionRate_labels_users[id])

        num_client = len(x)

        prob = pulp.LpProblem("Minimize_n", pulp.LpMinimize)

        y = [pulp.LpVariable(f'y_{i}', cat='Binary') for i in range(num_client)]

        prob += pulp.lpSum(y)

        for j in range(10):
            prob += pulp.lpSum(x[i][j] * y[i] for i in range(num_client)) >= 2
        # 求解问题
        prob.solve(PULP_CBC_CMD(msg=False))

        selected_client[key] = [i for i in range(num_client) if pulp.value(y[i]) == 1]
    for key in selected_client.keys():
        temp = 0
        for id in selected_client[key]:
            selected_client[key][temp] = user_id_dict[key][id]
            temp = temp + 1
    for id in range(len(distributionRate_labels_users)):
        print('distributionRate_labels_users ', id, ':', distributionRate_labels_users[id])
    print('dataset_client:', dataset_client)
    return selected_client


def system_parameter(args):
    np.random.seed(42)
    t_up = np.random.rand(args.pd_nuser, args.pd_nuser)
    np.fill_diagonal(t_up, 0)

    t_down = t_up.T
    np.random.seed(32)
    t_comp = np.random.rand(args.pd_nuser)

    return t_up, t_down, t_comp



def selection_with_time(distributionRate_labels_users, dataset_client, args, id_client):
    dataset_value = list(set(dataset_client.values()))
    num_users = len(distributionRate_labels_users)
    t_up, t_down, t_comp = system_parameter(args)
    constants = [0 for _ in range(args.pd_nuser)]
    for id in range(args.pd_nuser):
        constants[id] = t_up[id_client][id] + t_down[id_client][id] + t_comp[id]
    user_id_dict = {}
    for key in dataset_value:
        user_id_dict[key] = []
    for id in range(num_users):
        user_id_dict[dataset_client[id]].append(id)
    num_dataset = len(dataset_value)
    selected_client = {}
    for key in dataset_value:
        x = []
        temp_id = []
        for id in range(num_users):
            if dataset_client[id] == key:
                temp_id.append(id)
                x.append(distributionRate_labels_users[id])
        num_client = len(x)
        # 定义问题
        # prob = pulp.LpProblem("Minimize_n", pulp.LpMinimize)
        prob = pulp.LpProblem("Minimize_Maximum_Time", pulp.LpMinimize)
        # 定义变量 y_i，表示是否选择第 i 个向量
        y = [pulp.LpVariable(f'y_{i}', cat='Binary') for i in range(num_client)]

        # Auxiliary variable for the maximum
        t_coll = pulp.LpVariable("t_coll", lowBound=0)

        prob += t_coll

        for n in range(num_client):
            prob += t_coll >= constants[temp_id[n]] * y[n]

        for c in range(10):
            prob += pulp.lpSum(y[i] * x[i][c] for i in range(num_client)) >= 2, f"Capacity_Constraint_{c}"


        prob.solve(PULP_CBC_CMD(msg=False))

        selected_client[key] = [i for i in range(num_client) if pulp.value(y[i]) == 1]
    for key in selected_client.keys():
        temp = 0
        for id in selected_client[key]:
            selected_client[key][temp] = user_id_dict[key][id]
            temp = temp + 1
    max_time = []
    for key in selected_client.keys():
        for id in selected_client[key]:
            max_time.append(t_up[id_client][id] + t_down[id_client][id] + t_comp[id])
    # print('max_time:', max_time)
    total_time = max(max_time)
    return selected_client, total_time, t_comp[id_client]

def selection_with_AllRouting(distributionRate_labels_users, dataset_client, args, id_client):

    dataset_value = list(set(dataset_client.values()))
    t_up, t_down, t_comp = system_parameter(args)

    constants = [0 for _ in range(args.pd_nuser)]
    for id in range(args.pd_nuser):
        constants[id] = t_up[id_client][id] + t_down[id_client][id] + t_comp[id]

    max_time = max(constants)
    selected_client = {}
    for key in dataset_value:
        selected_client[key] = [id for id in range(args.pd_nuser)]

    return selected_client, max_time

def selection_with_LocalInference(dataset_client, args, id_client):

    dataset_value = list(set(dataset_client.values()))
    t_up, t_down, t_comp = system_parameter(args)

    inference_time = t_comp[id_client]

    selected_client = {}
    for key in dataset_value:
        selected_client[key] = [id_client]

    return selected_client, inference_time


def selection_with_RandomRouting(num_select, dataset_client, args, id_client):
    import random
    dataset_value = list(set(dataset_client.values()))
    t_up, t_down, t_comp = system_parameter(args)

    random_numbers = random.sample(range(args.pd_nuser), num_select)

    constants = [0 for _ in range(len(random_numbers))]
    temp = 0
    for id in random_numbers:
        constants[temp] = t_up[id_client][id] + t_down[id_client][id] + t_comp[id]
        temp = temp + 1

    max_time = max(constants)

    selected_client = {}
    for key in dataset_value:
        selected_client[key] = random_numbers

    return selected_client, max_time




def selection_inference(testloader_MoE, distributionRate_labels_users, dataset_client):

    # num_users = len(distribution_set)
    optimal_client = selection(distributionRate_labels_users, dataset_client)
    num_dataset = len(testloader_MoE)

    # selected_client = [[] for _ in range(num_dataset)]


    dataset_value = list(set(dataset_client.values()))
    selected_client = {key: [] for key in dataset_value}

    for key in dataset_value:
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
            for id in range(len(labels)):
                selected_client[key].append(optimal_client[key])
                # print('selected_client[id_dataset]:', selected_client[id_dataset])

    return selected_client


def test_MoE_with_selection(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, id):
    """Run test single model.
    Returns: loss, acc
    """

    dataset_value = list(set(dataset_client.values()))


    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)
    # all_optimal_client = [item for sublist in optimal_client.values() for item in sublist]

    loss_all, total, correct = 0, 0, 0

    for key in dataset_value:
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
    # if 1 == 1:
    #     for images, labels in tqdm(testloader_MoE, file=sys.stdout, disable=True):
            images = images.cuda()
            batch_size = len(images[0])
            _score = []
            output_set = []
            # output_set = torch.empty(0, batch_size, 10)
            for id_client in optimal_client[key]:
            # for id_client in optimal_client[dataset_client[id]]:
            # for id_client in all_optimal_client:
                model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                    model.eval()
                    model.to(device)
                    output = model(images)
                    output_class = output[:, user_class[id_client]]
                    # output = F.softmax(output, dim=1)
                    # output_squeeze = output.squeeze(0)
                    output_set.append(output)
                    smax = to_np(F.softmax(output_class, dim=1))
                    if use_xent:
                        _score.append(to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                    else:
                        if evaluation_score == 'energy':
                            _score.append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                        else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                            _score.append(-np.max(smax, axis=1))

            score_tensor = torch.tensor(_score)*-1  # num_selected * batch size 取相反数表示数值越大，可信度越高

            score_tensor = score_tensor.unsqueeze(2)  # num_selected * batch size * 1
            score_tensor = F.softmax(score_tensor, dim=0)
            score_tensor = score_tensor.cuda()

            output_tensor = torch.stack(output_set)  # num_selected * batch size * 10

            multiplied_tensor = output_tensor * score_tensor
            # multiplied_tensor = output_tensor

            # print('multiplied_tensor', multiplied_tensor.size())
            collaborative_results = multiplied_tensor.sum(dim=0, keepdim=True)
            collaborative_results = collaborative_results.squeeze(0)

            # print('collaborative_results', collaborative_results.size())
            labels = labels.to(device)
            # loss = loss_fun(collaborative_results, labels)
            # loss_all += loss.item()

            # _, pred = torch.max(collaborative_results, dim = 1)
            pred = collaborative_results.data.max(1)[1]

            # print('pred:', pred)
            # print('labels.view(-1):', labels.view(-1))

            correct += pred.eq(labels.view(-1)).sum().item()

            # print('correct:', correct)
            total += labels.size(0)
            del images, labels, output_tensor, score_tensor, collaborative_results

    return loss_all / total, correct / total, total

def test_MoE_wo_dataset_selection(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, id):
    """Run test single model.
    Returns: loss, acc
    """

    dataset_value = list(set(dataset_client.values()))


    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)
    all_optimal_client = [item for sublist in optimal_client.values() for item in sublist]

    loss_all, total, correct = 0, 0, 0

    for key in dataset_value:
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
    # if 1 == 1:
    #     for images, labels in tqdm(testloader_MoE, file=sys.stdout, disable=True):
            images = images.cuda()
            batch_size = len(images[0])
            _score = []
            output_set = []
            # output_set = torch.empty(0, batch_size, 10)
            for id_client in all_optimal_client:
                model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                    model.eval()
                    model.to(device)
                    output = model(images)
                    output_class = output[:, user_class[id_client]]
                    # output = F.softmax(output, dim=1)
                    # output_squeeze = output.squeeze(0)
                    output_set.append(output)
                    smax = to_np(F.softmax(output_class, dim=1))
                    if use_xent:
                        _score.append(to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                    else:
                        if evaluation_score == 'energy':
                            _score.append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                        else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                            _score.append(-np.max(smax, axis=1))

            score_tensor = torch.tensor(_score)*-1  # num_selected * batch size 取相反数表示数值越大，可信度越高

            score_tensor = score_tensor.unsqueeze(2)  # num_selected * batch size * 1
            score_tensor = F.softmax(score_tensor, dim=0)
            score_tensor = score_tensor.cuda()

            output_tensor = torch.stack(output_set)  # num_selected * batch size * 10

            multiplied_tensor = output_tensor * score_tensor
            # multiplied_tensor = output_tensor

            # print('multiplied_tensor', multiplied_tensor.size())
            collaborative_results = multiplied_tensor.sum(dim=0, keepdim=True)
            collaborative_results = collaborative_results.squeeze(0)

            # print('collaborative_results', collaborative_results.size())
            labels = labels.to(device)
            # loss = loss_fun(collaborative_results, labels)
            # loss_all += loss.item()

            # _, pred = torch.max(collaborative_results, dim = 1)
            pred = collaborative_results.data.max(1)[1]

            # print('pred:', pred)
            # print('labels.view(-1):', labels.view(-1))

            correct += pred.eq(labels.view(-1)).sum().item()

            # print('correct:', correct)
            total += labels.size(0)
            del images, labels, output_tensor, score_tensor, collaborative_results

    return loss_all / total, correct / total, total

def test_MoE_without_selection(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, id):
    """Run test single model.
    Returns: loss, acc
    """

    # NUM_cifar10 = int(args.pd_nuser / 3)
    # NUM_fashionmnist = int(args.pd_nuser / 3)
    # NUM_SVHN = args.pd_nuser - NUM_cifar10 - NUM_fashionmnist
    #
    # optimal_client['cifar10'] = [i for i in range(NUM_cifar10)]
    # optimal_client['fashion_mnist'] = [i + NUM_cifar10 for i in range(NUM_fashionmnist)]
    # optimal_client['SVHN'] = [i + NUM_cifar10 + NUM_fashionmnist for i in range(NUM_SVHN)]

    # optimal_client = selection(distributionRate_labels_users, dataset_client)
    dataset_value = list(set(dataset_client.values()))

    # for key in dataset_value:
    #     optimal_client[key] = [i for i in range(20)]

    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)

    loss_all, total, correct = 0, 0, 0

    for key in dataset_value:
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
            images = images.cuda()
            batch_size = len(images[0])
            _score = []
            output_set = []

            for id_client in optimal_client[key]:
                model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                    model.eval()
                    model.to(device)
                    output = model(images)
                    output_class = output[:, user_class[id_client]]
                    # output = F.softmax(output, dim=1)
                    # output_squeeze = output.squeeze(0)
                    output_set.append(output)
                    smax = to_np(F.softmax(output_class, dim=1))
                    if use_xent:
                        _score.append(to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                    else:
                        if evaluation_score == 'energy':
                            _score.append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                        else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                            _score.append(-np.max(smax, axis=1))

            score_tensor = torch.tensor(_score)*-1  # num_selected * batch size 取相反数表示数值越大，可信度越高

            score_tensor = score_tensor.unsqueeze(2)  # num_selected * batch size * 1
            score_tensor = F.softmax(score_tensor, dim=0)
            score_tensor = score_tensor.cuda()

            output_tensor = torch.stack(output_set)  # num_selected * batch size * 10

            multiplied_tensor = output_tensor * score_tensor
            # multiplied_tensor = output_tensor

            # print('multiplied_tensor', multiplied_tensor.size())
            collaborative_results = multiplied_tensor.sum(dim=0, keepdim=True)
            collaborative_results = collaborative_results.squeeze(0)

            # print('collaborative_results', collaborative_results.size())
            labels = labels.to(device)
            # loss = loss_fun(collaborative_results, labels)
            # loss_all += loss.item()

            # _, pred = torch.max(collaborative_results, dim = 1)
            pred = collaborative_results.data.max(1)[1]

            # print('pred:', pred)
            # print('labels.view(-1):', labels.view(-1))

            correct += pred.eq(labels.view(-1)).sum().item()

            # print('correct:', correct)
            total += labels.size(0)
            del images, labels, output_tensor, score_tensor, collaborative_results

    return loss_all / total, correct / total, total


def test_MoE_with_selection_dataset(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, id):
    """Run test single model.
    Returns: loss, acc
    """

    # NUM_cifar10 = int(args.pd_nuser / 3)
    # NUM_fashionmnist = int(args.pd_nuser / 3)
    # NUM_SVHN = args.pd_nuser - NUM_cifar10 - NUM_fashionmnist
    #
    # optimal_client['cifar10'] = [i for i in range(NUM_cifar10)]
    # optimal_client['fashion_mnist'] = [i + NUM_cifar10 for i in range(NUM_fashionmnist)]
    # optimal_client['SVHN'] = [i + NUM_cifar10 + NUM_fashionmnist for i in range(NUM_SVHN)]

    # optimal_client = selection(distributionRate_labels_users, dataset_client)
    dataset_value = list(set(dataset_client.values()))

    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)
    all_optimal_client = [item for sublist in optimal_client.values() for item in sublist]

    loss_all, total, correct , correct_dataset = 0, 0, 0, 0

    for key in dataset_value:
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
    # if 1 == 1:
    #     for images, labels in tqdm(testloader_MoE, file=sys.stdout, disable=True):
            images = images.cuda()
            labels = labels.to(device)
            batch_size = len(images[0])
            _score = {}
            _score_max = {}
            output_set = {}
            score_tensor = {}
            # output_set = torch.empty(0, batch_size, 10)
            # for id_client in optimal_client[key]:
            confidence_level = {}
            collaborative_results = {}
            for dset in dataset_value:
                _score[dset] = []
                _score_max[dset] = []
                output_set[dset] = []
                for id_client in optimal_client[dset]:
                    model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                    with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                        model.eval()
                        model.to(device)
                        output = model(images)
                        output_class = output[:, user_class[id_client]]
                        # output = F.softmax(output, dim=1)
                        # output_squeeze = output.squeeze(0)
                        output_set[dset].append(output)
                        smax = to_np(F.softmax(output_class, dim=1))
                        if use_xent:
                            _score[dset].append(to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                        else:
                            if evaluation_score == 'energy':
                                _score[dset].append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                                _score_max[dset].append(-np.max(smax, axis=1))
                            else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                                _score[dset].append(-np.max(smax, axis=1))

                # score_tensor[dset] = torch.tensor(_score[dset]) * torch.tensor(_score_max[dset]) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
                score_tensor[dset] = torch.tensor(_score[dset]) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
                # print('size of score_tensor[dset]:', score_tensor[dset].size())
                confidence_level[dset] = score_tensor[dset].sum(dim=0, keepdim=True)  # 1 * batch size
                confidence_level[dset] = confidence_level[dset] / score_tensor[dset].size()[0]

                score_tensor[dset] = score_tensor[dset].unsqueeze(2)  # num_selected * batch size * 1
                score_tensor[dset] = F.softmax(score_tensor[dset], dim=0)
                score_tensor[dset] = score_tensor[dset].cuda()
                output_tensor = torch.stack(output_set[dset])  # num_selected * batch size * 10
                multiplied_tensor = output_tensor * score_tensor[dset]
                # multiplied_tensor = output_tensor
                collaborative_results[dset] = multiplied_tensor.sum(dim=0, keepdim=True)      #  torch.Size([1, 128, 10]
                collaborative_results[dset] = collaborative_results[dset].squeeze(0)     #  torch.Size([128, 10])

            ########################################################################
            stacked = torch.cat(list(confidence_level.values()), dim=0)
            # 获取最大值和对应的索引
            max_values, indices = torch.max(stacked, dim=0)
            # 通过索引找到对应的键
            keys_dataset = list(confidence_level.keys())
            result_keys = [keys_dataset[index] for index in indices]
            target_keys = [key for _ in range(len(result_keys))]
            # target_keys = [dataset_client[id] for _ in range(len(result_keys))]
            matching_count = sum(1 for x, y in zip(result_keys, target_keys) if x == y)
            correct_dataset = correct_dataset + matching_count
            ########################################################################
            for id_sample in range(len(result_keys)):
                target_dset = result_keys[id_sample]
                _, pred = torch.max(collaborative_results[target_dset][id_sample].unsqueeze(0), dim = 1)
                correct += pred.eq(labels[id_sample].unsqueeze(0).view(-1)).sum().item()
            total += labels.size(0)
            del images, labels, output_tensor, score_tensor, collaborative_results

    return correct_dataset / total, correct / total, total


def test_MoE_wo_OoD_selection(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, id):
    """Run test single model.
    Returns: loss, acc
    """

    # NUM_cifar10 = int(args.pd_nuser / 3)
    # NUM_fashionmnist = int(args.pd_nuser / 3)
    # NUM_SVHN = args.pd_nuser - NUM_cifar10 - NUM_fashionmnist
    #
    # optimal_client['cifar10'] = [i for i in range(NUM_cifar10)]
    # optimal_client['fashion_mnist'] = [i + NUM_cifar10 for i in range(NUM_fashionmnist)]
    # optimal_client['SVHN'] = [i + NUM_cifar10 + NUM_fashionmnist for i in range(NUM_SVHN)]

    # optimal_client = selection(distributionRate_labels_users, dataset_client)
    dataset_value = list(set(dataset_client.values()))

    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)
    all_optimal_client = [item for sublist in optimal_client.values() for item in sublist]

    loss_all, total, correct , correct_dataset = 0, 0, 0, 0

    for key in dataset_value:
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
    # if 1 == 1:
    #     for images, labels in tqdm(testloader_MoE, file=sys.stdout, disable=True):
            images = images.cuda()
            labels = labels.to(device)
            batch_size = len(images[0])
            _score = {}
            output_set = {}
            score_tensor = {}
            # output_set = torch.empty(0, batch_size, 10)
            # for id_client in optimal_client[key]:
            confidence_level = {}
            collaborative_results = {}
            for dset in dataset_value:
                _score[dset] = []
                output_set[dset] = []
                for id_client in optimal_client[dset]:
                    model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                    with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                        model.eval()
                        model.to(device)
                        output = model(images)
                        output_class = output[:, user_class[id_client]]
                        # output = F.softmax(output, dim=1)
                        # output_squeeze = output.squeeze(0)
                        output_set[dset].append(output)
                        smax = to_np(F.softmax(output_class, dim=1))
                        if use_xent:
                            _score[dset].append(to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                        else:
                            if evaluation_score == 'energy':
                                _score[dset].append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                            else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                                _score[dset].append(-np.max(smax, axis=1))

                score_tensor[dset] = torch.tensor(_score[dset]) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
                # print('size of score_tensor[dset]:', score_tensor[dset].size())
                confidence_level[dset] = score_tensor[dset].sum(dim=0, keepdim=True)  # 1 * batch size
                confidence_level[dset] = confidence_level[dset] / score_tensor[dset].size()[0]

                score_tensor[dset] = score_tensor[dset].unsqueeze(2)  # num_selected * batch size * 1
                score_tensor[dset] = F.softmax(score_tensor[dset], dim=0)
                score_tensor[dset] = score_tensor[dset].cuda()
                output_tensor = torch.stack(output_set[dset])  # num_selected * batch size * 10

                #######################################################################
                # multiplied_tensor = output_tensor * score_tensor[dset]
                multiplied_tensor = output_tensor
                #######################################################################

                collaborative_results[dset] = multiplied_tensor.sum(dim=0, keepdim=True)      #  torch.Size([1, 128, 10]
                collaborative_results[dset] = collaborative_results[dset].squeeze(0)     #  torch.Size([128, 10])

            ########################################################################
            stacked = torch.cat(list(confidence_level.values()), dim=0)
            # 获取最大值和对应的索引
            max_values, indices = torch.max(stacked, dim=0)
            # 通过索引找到对应的键
            keys_dataset = list(confidence_level.keys())
            result_keys = [keys_dataset[index] for index in indices]
            target_keys = [key for _ in range(len(result_keys))]
            # target_keys = [key for _ in range(len(result_keys))]
            matching_count = sum(1 for x, y in zip(result_keys, target_keys) if x == y)
            correct_dataset = correct_dataset + matching_count
            ########################################################################
            for id_sample in range(len(result_keys)):
                target_dset = result_keys[id_sample]
                _, pred = torch.max(collaborative_results[target_dset][id_sample].unsqueeze(0), dim = 1)
                correct += pred.eq(labels[id_sample].unsqueeze(0).view(-1)).sum().item()
            total += labels.size(0)
            del images, labels, output_tensor, score_tensor, collaborative_results

    return correct_dataset / total, correct / total, total


#
# def test_MoE_with_selection_with_threthod(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, OoD_threthod, self_id):
#     """Run test single model.
#     Returns: loss, acc
#     """
#
#     NUM_cifar10 = int(args.pd_nuser / 3)
#     NUM_fashionmnist = int(args.pd_nuser / 3)
#     NUM_SVHN = args.pd_nuser - NUM_cifar10 - NUM_fashionmnist
#
#     optimal_client['cifar10'] = [i for i in range(NUM_cifar10)]
#     optimal_client['fashion_mnist'] = [i + NUM_cifar10 for i in range(NUM_fashionmnist)]
#     optimal_client['SVHN'] = [i + NUM_cifar10 + NUM_fashionmnist for i in range(NUM_SVHN)]
#
#     # optimal_client = selection(distributionRate_labels_users, dataset_client)
#     dataset_value = list(set(dataset_client.values()))
#     model_self = model_set[int(self_id)]
#
#     # for key in dataset_value:
#     #     optimal_client[key] = [i for i in range(20)]
#
#     to_np = lambda x: x.data.cpu().numpy()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     num_dataset = len(testloader_MoE)
#
#     loss_all, total, correct = 0, 0, 0
#     # for id_dataset in range(num_dataset):
#     for key in dataset_value:
#         output_set = torch.Tensor([])
#         OoD_score_set = torch.Tensor([])
#         for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
#             data = images.cuda()
#             batch_size = len(images[0])
#             model_self = model_self.to(device)  # 确保模型只转移一次
#             with torch.no_grad():  # 禁用梯度计算，减少内存消耗
#                 model_self.eval()
#                 model_self.to(device)
#                 output = model_self(data)
#                 output_class = output[:, user_class[self_id]]
#                 output = F.softmax(output, dim=1)
#
#                 # output_set = torch.cat((output_set, output), 0)
#                 smax = to_np(F.softmax(output_class, dim=1))
#                 if use_xent:
#                     score = (to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
#                 else:
#                     if evaluation_score == 'energy':
#                         score = (-to_np((T * torch.logsumexp(output_class / T, dim=1))))
#                     else:
#                         score = (-np.max(smax, axis=1))
#                 model_self.to('cpu')
#                 positions_InD = (score <= OoD_threthod).nonzero(as_tuple=True)[0]
#                 labels_InD = labels[positions_InD]
#                 _, pred_InD = torch.max(output, dim=1)
#                 correct += pred_InD.eq(labels_InD.view(-1)).sum().item()
#
#                 positions_ood = (score > OoD_threthod).nonzero(as_tuple=True)[0]
#                 _score_ood = []
#                 output_set_ood = []
#                 for id_client in optimal_client[key]:
#                     data_ood = images[positions_ood]
#                     labels_ood = labels[positions_ood]
#                     model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
#                     model.eval()
#                     model.to(device)
#                     output_ood = model(data_ood)
#                     output_ood = output_ood[:, user_class[id_client]]
#                     # output_squeeze_ood = output_ood.squeeze(0)
#                     output_set_ood.append(output_ood)
#                     smax = to_np(F.softmax(output_ood, dim=1))
#                     if use_xent:
#                         _score_ood.append(to_np((output_ood.mean(1) - torch.logsumexp(output_ood, dim=1))))
#                     else:
#                         if evaluation_score == 'energy':
#                             _score_ood.append(-to_np((T * torch.logsumexp(output_ood / T, dim=1))))
#                         else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
#                             _score_ood.append(-np.max(smax, axis=1))
#                     model.to('cpu')
#                 score_tensor_ood = torch.tensor(_score_ood) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
#                 score_tensor_ood = score_tensor_ood.unsqueeze(2)
#                 score_tensor_ood = F.softmax(score_tensor_ood, dim=0)
#                 score_tensor_ood = score_tensor_ood.cuda()
#
#                 output_tensor_ood = torch.stack(output_set_ood)
#
#                 multiplied_tensor_ood = output_tensor_ood * score_tensor_ood
#                 # multiplied_tensor_ood = output_tensor_ood
#
#                 # print('multiplied_tensor', multiplied_tensor.size())
#                 collaborative_results_ood = multiplied_tensor_ood.sum(dim=0, keepdim=True)
#                 collaborative_results_ood = collaborative_results_ood.squeeze(0)
#                 labels_ood = labels_ood.to(device)
#                 # loss = loss_fun(collaborative_results_ood, labels_ood)
#                 # loss_all += loss.item()
#                 _, pred_ood = torch.max(collaborative_results_ood, dim=1)
#                 correct += pred_ood.eq(labels_ood.view(-1)).sum().item()
#
#                 total += labels.size(0)
#                 OoD_score_set = torch.cat((OoD_score_set, score), 0)
#
#     return loss_all / total, correct / total, total



def test_MoE_with_selection_with_threthod(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, OoD_threthod, self_id):
    """Run test single model.
    Returns: loss, acc
    """



    # optimal_client = selection(distributionRate_labels_users, dataset_client)
    dataset_value = list(set(dataset_client.values()))
    model_self = model_set[int(self_id)]

    # for key in dataset_value:
    #     optimal_client[key] = [i for i in range(20)]

    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)

    loss_all, total, correct = 0, 0, 0
    total_ID = 0
    total_OOD = 0
    correct_dataset = 0
    # for id_dataset in range(num_dataset):
    for key in dataset_value:
        output_set = torch.Tensor([])
        OoD_score_set = torch.Tensor([])
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
            data = images.cuda()
            batch_size = len(images[0])
            model_self = model_self.to(device)  # 确保模型只转移一次
            with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                model_self.eval()
                model_self.to(device)
                output = model_self(data)
                output_class = output[:, user_class[self_id]]
                output = F.softmax(output, dim=1)

                # output_set = torch.cat((output_set, output), 0)
                smax = to_np(F.softmax(output_class, dim=1))
                if use_xent:
                    score = (to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                else:
                    if evaluation_score == 'energy':
                        score = (-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                    else:
                        score = (-np.max(smax, axis=1))    #  表示数值越小，可信度越高
                score_tensor = torch.tensor(score) * -1  # 取相反数表示数值越大，可信度越高
                model_self.to('cpu')
                # print('score:',score)
                positions_InD = [index for index, value in enumerate(score_tensor) if value >= OoD_threthod]
                labels_InD = labels[positions_InD]
                labels_InD = labels_InD.cuda()
                total_ID += len(positions_InD)
                _, pred_InD = torch.max(output[positions_InD], dim=1)
                correct += pred_InD.eq(labels_InD.view(-1)).sum().item()

                # positions_ood = (score > OoD_threthod).nonzero(as_tuple=True)[0]
                positions_ood = [index for index, value in enumerate(score_tensor) if value < OoD_threthod]
                if len(positions_ood) != 0:
                    total_OOD += len(positions_ood)
                    _score_ood = []
                    output_set_ood = []
                    data_ood = images[positions_ood]
                    labels_ood = labels[positions_ood]
                    labels_ood = labels_ood.cuda()
                    data_ood = data_ood.cuda()
                    _score = {}
                    output_set = {}
                    score_tensor = {}
                    confidence_level = {}
                    collaborative_results = {}
                    for dset in dataset_value:
                        _score[dset] = []
                        output_set[dset] = []
                        for id_client in optimal_client[dset]:
                            model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                            with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                                model.eval()
                                model.to(device)
                                output = model(data_ood)
                                output_class = output[:, user_class[id_client]]
                                # output = F.softmax(output, dim=1)
                                # output_squeeze = output.squeeze(0)
                                output_set[dset].append(output)
                                smax = to_np(F.softmax(output_class, dim=1))
                                if use_xent:
                                    _score[dset].append(to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                                else:
                                    if evaluation_score == 'energy':
                                        _score[dset].append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                                    else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                                        _score[dset].append(-np.max(smax, axis=1))

                        score_tensor[dset] = torch.tensor(_score[dset]) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
                        # print('size of score_tensor[dset]:', score_tensor[dset].size())
                        confidence_level[dset] = score_tensor[dset].sum(dim=0, keepdim=True)  # 1 * batch size
                        confidence_level[dset] = confidence_level[dset] / score_tensor[dset].size()[0]
                        # print('score_tensor[dset]:', score_tensor[dset])
                        # print('score_tensor[dset].size():', score_tensor[dset].size())
                        score_tensor[dset] = score_tensor[dset].unsqueeze(2)  # num_selected * batch size * 1
                        score_tensor[dset] = F.softmax(score_tensor[dset], dim=0)
                        score_tensor[dset] = score_tensor[dset].cuda()
                        output_tensor = torch.stack(output_set[dset])  # num_selected * batch size * 10
                        multiplied_tensor = output_tensor * score_tensor[dset]

                        collaborative_results[dset] = multiplied_tensor.sum(dim=0, keepdim=True)  # torch.Size([1, 128, 10]
                        collaborative_results[dset] = collaborative_results[dset].squeeze(0)  # torch.Size([128, 10])
                    ########################################################################
                    stacked = torch.cat(list(confidence_level.values()), dim=0)
                    # 获取最大值和对应的索引
                    max_values, indices = torch.max(stacked, dim=0)
                    # 通过索引找到对应的键
                    keys_dataset = list(confidence_level.keys())
                    result_keys = [keys_dataset[index] for index in indices]
                    target_keys = [key for _ in range(len(result_keys))]
                    # target_keys = [key for _ in range(len(result_keys))]
                    matching_count = sum(1 for x, y in zip(result_keys, target_keys) if x == y)
                    correct_dataset = correct_dataset + matching_count
                    ########################################################################
                    for id_sample in range(len(result_keys)):
                        target_dset = result_keys[id_sample]
                        _, pred = torch.max(collaborative_results[target_dset][id_sample].unsqueeze(0), dim=1)
                        correct += pred.eq(labels_ood[id_sample].unsqueeze(0).view(-1)).sum().item()

            total += labels.size(0)
            # OoD_score_set = torch.cat((OoD_score_set, score), 0)

    return loss_all / total, correct / total, total, total_ID, total_OOD




def test_MoE_wo_dataset_selection_with_threthod(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, OoD_threthod, self_id):
    """Run test single model.
    Returns: loss, acc
    """

    # optimal_client = selection(distributionRate_labels_users, dataset_client)
    dataset_value = list(set(dataset_client.values()))
    model_self = model_set[int(self_id)]
    all_optimal_client = [item for sublist in optimal_client.values() for item in sublist]

    # for key in dataset_value:
    #     optimal_client[key] = [i for i in range(20)]

    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)

    loss_all, total, correct = 0, 0, 0
    total_ID = 0
    total_OOD = 0
    # for id_dataset in range(num_dataset):
    for key in dataset_value:
        output_set = torch.Tensor([])
        OoD_score_set = torch.Tensor([])
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
            data = images.cuda()
            batch_size = len(images[0])
            model_self = model_self.to(device)  # 确保模型只转移一次
            with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                model_self.eval()
                model_self.to(device)
                output = model_self(data)
                output_class = output[:, user_class[self_id]]
                output = F.softmax(output, dim=1)

                # output_set = torch.cat((output_set, output), 0)
                smax = to_np(F.softmax(output_class, dim=1))
                if use_xent:
                    score = (to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                else:
                    if evaluation_score == 'energy':
                        score = (-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                    else:
                        score = (-np.max(smax, axis=1))
                score_tensor = torch.tensor(score) * -1  # 取相反数表示数值越大，可信度越高
                model_self.to('cpu')
                # positions_InD = (score <= OoD_threthod).nonzero(as_tuple=True)[0]
                positions_InD = [index for index, value in enumerate(score_tensor) if value >= OoD_threthod]
                labels_InD = labels[positions_InD]
                labels_InD = labels_InD.cuda()
                total_ID += len(positions_InD)
                _, pred_InD = torch.max(output[positions_InD], dim=1)
                correct += pred_InD.eq(labels_InD.view(-1)).sum().item()

                # positions_ood = (score > OoD_threthod).nonzero(as_tuple=True)[0]
                positions_ood = [index for index, value in enumerate(score_tensor) if value < OoD_threthod]
                if len(positions_ood) != 0:
                    total_OOD += len(positions_ood)
                    _score_ood = []
                    output_set_ood = []
                    data_ood = images[positions_ood]
                    labels_ood = labels[positions_ood]
                    labels_ood = labels_ood.cuda()
                    data_ood = data_ood.cuda()
                    _score = []
                    output_set = []
                    for id_client in all_optimal_client:
                        model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                        with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                            model.eval()
                            model.to(device)
                            output = model(data_ood)
                            output_class = output[:, user_class[id_client]]
                            # output = F.softmax(output, dim=1)
                            # output_squeeze = output.squeeze(0)
                            output_set.append(output)
                            smax = to_np(F.softmax(output_class, dim=1))
                            if use_xent:
                                _score.append(
                                    to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                            else:
                                if evaluation_score == 'energy':
                                    _score.append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                                else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                                    _score.append(-np.max(smax, axis=1))

                    score_tensor = torch.tensor(_score) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
                    # print('size of score_tensor[dset]:', score_tensor[dset].size())
                    confidence_level = score_tensor.sum(dim=0, keepdim=True)  # 1 * batch size
                    confidence_level = confidence_level / score_tensor.size()[0]

                    score_tensor = score_tensor.unsqueeze(2)  # num_selected * batch size * 1
                    score_tensor = F.softmax(score_tensor, dim=0)
                    score_tensor = score_tensor.cuda()
                    output_tensor = torch.stack(output_set)  # num_selected * batch size * 10
                    multiplied_tensor = output_tensor * score_tensor
                    # multiplied_tensor = output_tensor
                    collaborative_results = multiplied_tensor.sum(dim=0, keepdim=True)  # torch.Size([1, 128, 10]
                    collaborative_results = collaborative_results.squeeze(0)  # torch.Size([128, 10])
                ########################################################################
                    pred = collaborative_results.data.max(1)[1]
                    correct += pred.eq(labels_ood.view(-1)).sum().item()

            total += labels.size(0)
            # OoD_score_set = torch.cat((OoD_score_set, score), 0)

    return loss_all / total, correct / total, total, total_ID, total_OOD


def test_MoE_wo_OoD_selection_with_threthod(args, model_set, testloader_MoE, loss_fun, evaluation_score, use_xent, user_class, T, distributionRate_labels_users, dataset_client, optimal_client, OoD_threthod, self_id):
    """Run test single model.
    Returns: loss, acc
    """

    # optimal_client = selection(distributionRate_labels_users, dataset_client)
    dataset_value = list(set(dataset_client.values()))
    model_self = model_set[int(self_id)]

    # for key in dataset_value:
    #     optimal_client[key] = [i for i in range(20)]

    to_np = lambda x: x.data.cpu().numpy()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dataset = len(testloader_MoE)

    loss_all, total, correct = 0, 0, 0
    correct_dataset = 0
    total_ID = 0
    total_OOD = 0
    # for id_dataset in range(num_dataset):
    for key in dataset_value:
        output_set = torch.Tensor([])
        OoD_score_set = torch.Tensor([])
        for images, labels in tqdm(testloader_MoE[key], file=sys.stdout, disable=True):
            data = images.cuda()
            batch_size = len(images[0])
            model_self = model_self.to(device)  # 确保模型只转移一次
            with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                model_self.eval()
                model_self.to(device)
                output = model_self(data)
                output_class = output[:, user_class[self_id]]
                output = F.softmax(output, dim=1)

                # output_set = torch.cat((output_set, output), 0)
                smax = to_np(F.softmax(output_class, dim=1))
                if use_xent:
                    score = (to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                else:
                    if evaluation_score == 'energy':
                        score = (-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                    else:
                        score = (-np.max(smax, axis=1))
                score_tensor = torch.tensor(score) * -1  # 取相反数表示数值越大，可信度越高
                model_self.to('cpu')
                # print('score:', score)
                # positions_InD = (score <= OoD_threthod).nonzero(as_tuple=True)[0]
                positions_InD = [index for index, value in enumerate(score_tensor) if value >= OoD_threthod]
                labels_InD = labels[positions_InD]
                labels_InD = labels_InD.cuda()
                total_ID += len(positions_InD)
                _, pred_InD = torch.max(output[positions_InD], dim=1)
                correct += pred_InD.eq(labels_InD.view(-1)).sum().item()

                # positions_ood = (score > OoD_threthod).nonzero(as_tuple=True)[0]
                positions_ood = [index for index, value in enumerate(score_tensor) if value < OoD_threthod]
                if len(positions_ood) != 0:
                    total_OOD += len(positions_ood)
                    _score_ood = []
                    output_set_ood = []
                    data_ood = images[positions_ood]
                    labels_ood = labels[positions_ood]
                    labels_ood = labels_ood.cuda()
                    data_ood = data_ood.cuda()
                    _score = {}
                    output_set = {}
                    score_tensor = {}
                    confidence_level = {}
                    collaborative_results = {}
                    for dset in dataset_value:
                        _score[dset] = []
                        output_set[dset] = []
                        for id_client in optimal_client[dset]:
                            model = model_set[int(id_client)].to(device)  # 确保模型只转移一次
                            with torch.no_grad():  # 禁用梯度计算，减少内存消耗
                                model.eval()
                                model.to(device)
                                output = model(data_ood)
                                output_class = output[:, user_class[id_client]]
                                # output = F.softmax(output, dim=1)
                                # output_squeeze = output.squeeze(0)
                                output_set[dset].append(output)
                                smax = to_np(F.softmax(output_class, dim=1))
                                if use_xent:
                                    _score[dset].append(
                                        to_np((output_class.mean(1) - torch.logsumexp(output_class, dim=1))))
                                else:
                                    if evaluation_score == 'energy':
                                        _score[dset].append(-to_np((T * torch.logsumexp(output_class / T, dim=1))))
                                    else:  # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                                        _score[dset].append(-np.max(smax, axis=1))

                        score_tensor[dset] = torch.tensor(_score[dset]) * -1  # num_selected * batch size 取相反数表示数值越大，可信度越高
                        # print('size of score_tensor[dset]:', score_tensor[dset].size())
                        confidence_level[dset] = score_tensor[dset].sum(dim=0, keepdim=True)  # 1 * batch size
                        confidence_level[dset] = confidence_level[dset] / score_tensor[dset].size()[0]

                        score_tensor[dset] = score_tensor[dset].unsqueeze(2)  # num_selected * batch size * 1
                        score_tensor[dset] = F.softmax(score_tensor[dset], dim=0)
                        score_tensor[dset] = score_tensor[dset].cuda()
                        output_tensor = torch.stack(output_set[dset])  # num_selected * batch size * 10
                        # multiplied_tensor = output_tensor * score_tensor[dset]
                        multiplied_tensor = output_tensor
                        collaborative_results[dset] = multiplied_tensor.sum(dim=0, keepdim=True)  # torch.Size([1, 128, 10]
                        collaborative_results[dset] = collaborative_results[dset].squeeze(0)  # torch.Size([128, 10])
                    ########################################################################
                    stacked = torch.cat(list(confidence_level.values()), dim=0)
                    # 获取最大值和对应的索引
                    max_values, indices = torch.max(stacked, dim=0)
                    # 通过索引找到对应的键
                    keys_dataset = list(confidence_level.keys())
                    result_keys = [keys_dataset[index] for index in indices]
                    target_keys = [key for _ in range(len(result_keys))]
                    # target_keys = [key for _ in range(len(result_keys))]
                    matching_count = sum(1 for x, y in zip(result_keys, target_keys) if x == y)
                    correct_dataset = correct_dataset + matching_count
                    ########################################################################
                    for id_sample in range(len(result_keys)):
                        target_dset = result_keys[id_sample]
                        _, pred = torch.max(collaborative_results[target_dset][id_sample].unsqueeze(0), dim=1)
                        correct += pred.eq(labels_ood[id_sample].unsqueeze(0).view(-1)).sum().item()

            total += labels.size(0)
            # OoD_score_set = torch.cat((OoD_score_set, score), 0)

    return loss_all / total, correct / total, total, total_ID, total_OOD

