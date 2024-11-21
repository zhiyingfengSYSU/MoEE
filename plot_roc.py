import pickle
import os
import matplotlib.pyplot as plt
# 假设你有三个字符串



in_name = 'stl'

if in_name == 'cifar10':
    # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
    # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
    out_name = ["Textures", "LSUN-C", "LSUN-R", "iSUN", "MNIST", "FASHION-MNIST"]
elif in_name== 'stl':
    # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
    # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
    out_name = ["Textures", "LSUN-C", "LSUN-R", "iSUN", "MNIST", "FASHION-MNIST"]
elif in_name == 'mnist':
    # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
    # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
    out_name = ["Textures", "LSUN-C", "LSUN-R", "iSUN", "FASHION-MNIST", "CIFAR10"]
elif in_name == 'FashionMNIST':
    # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100", "MNIST"]
    # dataset_name = ["Texture", "LSUN_C", "LSUN_Resize", "iSUN", "CIFAR100"]
    out_name = ["Textures", "LSUN-C", "LSUN-R", "iSUN", "MNIST", "CIFAR10"]
elif in_name == 'SVHN':
    out_name = ["Textures", "LSUN-C", "LSUN-R", "iSUN", "FASHION-MNIST", "CIFAR10"]
else:
    # dataset_name = ["Texture", "Places365", "LSUN_C", "LSUN_Resize", "iSUN"]
    out_name = ["Textures", "LSUN-C", "LSUN-R", "iSUN", "FASHION-MNIST"]

# out_name = ['Textures', 'LSUN-C', 'LSUN-R', 'iSUN', 'MNIST', 'FASHION-MNIST']

for id in range(20):

    id_client = str(id)
    use_external = {'MSP':'None', 'ODIN':'None', 'VOS':'None', 'FOSTER':'gen_inverse', 'Ours':'ours', 'energy':'dataset'}
    evaluation_score = {'MSP':'msp', 'ODIN':'odin', 'VOS':'msp', 'FOSTER':'energy', 'Ours':'energy', 'energy':'energy'}
    loss_weight = {'MSP':'0.0', 'ODIN':'0.0', 'VOS':'0.05', 'FOSTER':'0.1', 'Ours':'0.1', 'energy':'0.1'}
    federated = 0


    for i in range(len(out_name)):

        file_name_MSP = f"{in_name}_{out_name[i]}_{id_client}_{use_external['MSP']}_{evaluation_score['MSP']}_{loss_weight['MSP']}_{federated}.pkl"     # MSP
        file_name_ODIN = f"{in_name}_{out_name[i]}_{id_client}_{use_external['ODIN']}_{evaluation_score['ODIN']}_{loss_weight['ODIN']}_{federated}.pkl"     # ODIN
        file_name_VOS = f"{in_name}_{out_name[i]}_{id_client}_{use_external['VOS']}_{evaluation_score['VOS']}_{loss_weight['VOS']}_{federated}.pkl"     # VOS
        file_name_energy = f"{in_name}_{out_name[i]}_{id_client}_{use_external['energy']}_{evaluation_score['energy']}_{loss_weight['energy']}_{federated}.pkl"  # VOS
        file_name_Ours = f"{in_name}_{out_name[i]}_{id_client}_{use_external['Ours']}_{evaluation_score['Ours']}_{loss_weight['Ours']}_{federated}.pkl"     # Ours

        # 给定的路径
        file_path_MSP = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_MSP}'
        file_path_ODIN = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_ODIN}'
        file_path_VOS = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_VOS}'
        # file_path_FOSTER = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_FOSTER}'
        file_path_energy = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_energy}'
        file_path_Ours = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/{file_name_Ours}'

        # 从文件中加载数据
        with open(file_path_MSP, 'rb') as file:
            fpr_MSP, tpr_MSP = pickle.load(file)
        with open(file_path_ODIN, 'rb') as file:
            fpr_ODIN, tpr_ODIN = pickle.load(file)
        with open(file_path_VOS, 'rb') as file:
            fpr_VOS, tpr_VOS = pickle.load(file)
        # with open(file_path_FOSTER, 'rb') as file:
        #     fpr_FOSTER, tpr_FOSTER = pickle.load(file)
        with open(file_path_energy, 'rb') as file:
            fpr_energy, tpr_energy = pickle.load(file)
        with open(file_path_Ours, 'rb') as file:
            fpr_Ours, tpr_Ours = pickle.load(file)

        figsize = (8, 6)
        plt.figure(figsize=figsize)
        plt.plot(fpr_MSP, tpr_MSP, lw=2, label='MSP')
        plt.plot(fpr_ODIN, tpr_ODIN, color='blue', lw=2, label='ODIN')
        plt.plot(fpr_VOS, tpr_VOS, color='green', lw=2, label='VOS')
        # plt.plot(fpr_FOSTER, tpr_FOSTER, color='red', lw=2, label='FOSTER')
        plt.plot(fpr_energy, tpr_energy, color='red', lw=2, label='Energy')
        plt.plot(fpr_Ours, tpr_Ours, color='cyan', lw=2, label='Ours')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right", fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        file_name_eps = f"{in_name}_{out_name[i]}_{id_client}_{federated}.eps"
        # 指定文件夹路径
        file_path_eps = f'D:/中山大学/github代码/FOSTER-Hergenerous/plot_data/eps_file/{file_name_eps}'
        # 确保文件夹存在，如果不存在则创建
        os.makedirs(os.path.dirname(file_path_eps), exist_ok=True)
        # plt.show()
        # 保存图形为.eps文件
        plt.savefig(file_path_eps, format='eps')




