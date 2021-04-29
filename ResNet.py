
import torch
from torch import nn, optim
from torch import Tensor
# from .utils import load_state_dict_from_url
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter



from typing import Union, List, Dict, cast
from typing import Type, Any, Callable, Union, List, Optional

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

images, _ = iter(train_iter).next()
# #########################################################################
# 搭建ResNet18网络
# #########################################################################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(
        self,
        block=BasicBlock,    # resnet-18: 只有BasicBlock, 没有Bottleneck
        layers=[2, 2, 2, 2],
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None
    ) :
        super(ResNet18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 训练MNIST ，单通道，由3改成1
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) :
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # resnet 18,34,50,101,152等都有的层
        x = self.conv1(x)  # conv1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # conv2 X 2
        x = self.layer2(x) # conv3 X 2
        x = self.layer3(x) # conv4 X 2
        x = self.layer4(x) # conv5 X 2

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# #########################################################################
# 定义神经网络模型
# #########################################################################

net = ResNet18()
train_writer = SummaryWriter('Resnet_18_plain')    # 使用tensorboard 可视化  ****************
train_writer.add_graph(net, images)
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
# torch.save(net.state_dict(), '../result/Resnet_18_plain.pkl')   # ********************
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
