import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, FashionMNIST, SVHN
from models.ViTransformer import VisionTransformer_cloud
from models.wrn_virtual import WideResNet_cloud
from nets.HeteFL.preresne import resnet18_cloud


########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 == 0), 'Depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-ResNet %dx%d' % (depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def WideResNet30():
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=30)



#######################################################################################


# 自定义数据集类以统一标签
class UnifiedDataset(Dataset):
    def __init__(self, cifar10, fashionmnist, svhn):
        self.cifar10 = cifar10
        self.fashionmnist = fashionmnist
        self.svhn = svhn

    def __len__(self):
        return len(self.cifar10) + len(self.fashionmnist) + len(self.svhn)

    def __getitem__(self, idx):
        if idx < len(self.cifar10):
            img, label = self.cifar10[idx]
            # CIFAR10 labels range from 0-9
            label = label
        elif idx < len(self.cifar10) + len(self.fashionmnist):
            img, label = self.fashionmnist[idx - len(self.cifar10)]
            # FashionMNIST labels range from 10-19
            label = label + 10
        else:
            img, label = self.svhn[idx - len(self.cifar10) - len(self.fashionmnist)]
            # SVHN labels range from 20-29
            label = label + 20
        return img, label

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
cifar10_train = CIFAR10(root='./dataset', train=True, download=True, transform=transform)
fashionmnist_train = FashionMNIST(root='./dataset', train=True, download=True, transform=transform)
svhn_train = SVHN(root='./dataset', split='train', download=True, transform=transform)

cifar10_test = CIFAR10(root='./dataset', train=False, download=True, transform=transform)
fashionmnist_test = FashionMNIST(root='./dataset', train=False, download=True, transform=transform)
svhn_test = SVHN(root='./dataset', split='test', download=True, transform=transform)

# 创建统一的数据集
train_dataset = UnifiedDataset(cifar10_train, fashionmnist_train, svhn_train)
test_dataset = UnifiedDataset(cifar10_test, fashionmnist_test, svhn_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 实例化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = SimpleCNN(num_classes=30)
# model = WideResNet_cloud(40, 30, 2, dropRate=0.3, track_running_stats=False).to(device)
model = WideResNet30()
# model = resnet18_cloud( track_running_stats=True, num_classes=30, width_scale=1).to(device)
# model = VisionTransformer_cloud().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10

model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
