import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)  # 线性变换
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

PATH = './classifier_para.pth'
net.load_state_dict(torch.load(PATH))

trainloader = torch.load("data.pth")
# print(trainloader)
labels = trainloader["label"]
# print(labels)
feature = trainloader["feature"]

totalnum = 0
correctnum = 0
# 没有训练，因此不需要计算输出的梯度
with torch.no_grad():
    # for data in testloader:
    for i in range(57000, 60000):
        # print(data)
        label = labels[i]
        input = feature[i]
        # ouputs = net(inputs)
        # images,label = data
        # 前向传播
        outputs = net(input)
        # print(outputs)
        _, predicted = torch.max(outputs, 0)
        # totalnum所有测试图像数量，correctnum预测准确图像数量
        totalnum += 1
        correctnum += (predicted == label).sum().item()
        # if predicted == label:
        #     correctnum += 1

print("Accuracy of the network on the 3000 test images:%.5f %%" % (100.0 * correctnum / totalnum))