import torch.nn as nn
from relu import DynamicReLUA, DynamicReLUB
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.relu = DynamicReLUB(10, reduction=2,conv_type= '2d')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


if __name__=='__main__':
    model = Model()
    x = torch.randn(1, 3, 32, 32) 
    y = model(x)
    print(y.shape)
    # print(model)
    # print(model.relu.lambdas)
    # print(model.relu.init_v)
    # print(model.relu.fc_1.weight)
    # print
    a = torch.randn(1, 10, 20 ,30,1)
    b = torch.randn(1,30,2 )
    c = a*b
    print(c.shape, "c")