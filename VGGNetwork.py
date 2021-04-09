
import torch
from torch import nn, optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from typing import Union, List, Dict, cast
import sys

# 定义一些超参数
batch_size = 256  # 64
learning_rate = 0.001  # *****************  0.001 0.005 0.01 0.1 0.5, 0.002
num_epoches = 25  # 100

# #########################################################################
# 加载手写数据集MNIST
# #########################################################################
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])  # 把图片转成tensor并归一化
# 获得训练集
train_dataset = datasets.MNIST(root='../data',
                               train=True, transform=transform, download=False)
# 获得测试集
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=False)

# 读取数据
if sys.platform.startswith('win'):
    num_workers = 0  # 不用额外的进程来加速读取数据
else:
    num_workers = 4

train_iter = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)
test_iter = DataLoader(test_dataset, batch_size=batch_size,

                       shuffle=True, num_workers=num_workers)

# #########################################################################
# 搭建VGG11网络
# #########################################################################
cfgs = {
    # M表示MaxPool2d,其他数值表示各卷积层输出的深度
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
}


class VGG11(nn.Module):

    def __init__(self, vgg_name, init_weights=True):
        super(VGG11, self).__init__()
        self.features = self.make_layers(cfgs[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # FC-4096
            nn.ReLU(True),
            nn.Dropout(p=0.5),   # *****
            nn.Linear(4096, 4096),  # FC-4096
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),  # FC-10
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 推平成一个 (batch_size,n) 的 tensor
        x = self.classifier(x)
        return x

    def _initialize_weights(self):  #   初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg, batch_norm: bool = False):  #  ********
        layers: List[nn.Module] = []
        in_channels = 1    # MNIST图片为单通道
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # (n-2)/2+1 = n/2, 缩小一半
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # (n-3+1x2)/1+1 = n, 保持不变
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]   # *****
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


# #########################################################################
# 定义神经网络模型
# #########################################################################
net = VGG11('vgg11')

net = net.cuda()    # GPU加速

# #########################################################################
# 定义损失函数和优化器
# #########################################################################
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # 动量随机梯度下降法

# #########################################################################
# 训练模型
# #########################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on", device)  # 打印出在GPU还是在CPU上训练

net.train()

train_writer = SummaryWriter('Dropout_lr0.001')    # 使用tensorboard 可视化  ****************

for epoch in range(num_epoches):
    for data in train_iter:
        img, label = data

        img = img.cuda()
        label = label.cuda()

        out = net(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))

    train_writer.add_scalar('training loss',
                            loss.data.item(),
                            epoch)               # tensorboard可视化训练误差， tensorboard --logdir=dir
train_writer.close()
torch.save(net.state_dict(), '../result/vgg11_Dropout_lr0.001.pkl')   # ********************
print('****Finished Training****')

# ######################################################################
# 模型评估
# ######################################################################
net.eval()
correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for data in test_iter:
        img, label = data

        img = img.cuda()
        label = label.cuda()

        out = net(img)

        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        correct += num_correct.item()
        total += label.size(0)
print('****Finished Testing****')
print('Accuracy: {:.2f}%'.format(correct / total * 100))
