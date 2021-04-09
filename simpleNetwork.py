
import torch
from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import sys

# import matplotlib.pyplot as plt

# 定义一些超参数
batch_size = 256  # 64
learning_rate = 0.001  # 0.001
num_epoches = 150  # 100

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

'''
# 数据集可视化
examples = enumerate(train_iter)
batch_idx, (example_data, example_targets) = next(examples)
fig = plt.figure()
for i in range(6):
  plt.subplot(2, 3, i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
'''


# #########################################################################
# 搭建只有一个隐藏层的简单神经网络
# #########################################################################
class simpleNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(simpleNet, self).__init__()

        self.hidden = nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function: relu
        x = self.out(x)
        return x


# #########################################################################
# 定义神经网络模型
# #########################################################################
net = simpleNet(28*28, 900, 10)   # 读取数据 MNIST每张图片大小为28*28，隐藏层设为100个神经元，输出层为10个神经元（0-9十个数字）***
net = net.cuda()    # GPU加速

# #########################################################################
# 定义损失函数和优化器
# #########################################################################
criterion = nn.CrossEntropyLoss()  # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # 动量随机梯度下降法


# #########################################################################
# 训练模型
# #########################################################################
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on", device)  # 打印出在GPU还是在CPU上训练

    net.train()

    train_writer = SummaryWriter('neuronNum900')    # 使用tensorboard 可视化  ***

    for epoch in range(num_epoches):
        for data in train_iter:
            img, label = data
            img = img.view(-1, 28*28)

            img = img.cuda()
            label = label.cuda()

            out = net(img)
            loss = criterion(out, label)   # 训练损失

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()        # backpropagation, compute gradients
            optimizer.step()       # apply gradients

        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
        if epoch % 2 == 0:
            train_writer.add_scalar('training loss',
                                    loss.data.item(),
                                    epoch)               # tensorboard可视化训练误差， tensorboard --logdir=run
    train_writer.close()
    torch.save(net.state_dict(), '../result/net_params_neuronNum900.pkl')   # ***
    print('****Finished Training****')


# #######################################################################
# 模型评估
# ######################################################################
def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():   # 不计算梯度
        for data in test_iter:
            img, label = data
            img = img.view(img.size(0), -1)

            img = img.cuda()
            label = label.cuda()

            out = net(img)

            _, pred = torch.max(out, 1)  # 预测值
            num_correct = (pred == label).sum()  # 计算预测对的个数
            correct += num_correct.item()
            total += label.size(0)
    print('****Finished Testing****')
    print('Accuracy: {:.2f}%'.format(correct / total * 100))


if __name__ == '__main__':
    torch.cuda.manual_seed(1)  # 保证每次初始化的参数都相同

    test()  # 先初始化模型参数
    train()
    test()
