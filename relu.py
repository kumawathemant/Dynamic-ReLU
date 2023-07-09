import torch 
import torch.nn as nn 


class DynamicReLU(nn.Module): 
    def __init__(self, channels, k =2 , reduction = 4 , conv_type = '2d'):
        super(DynamicReLU, self).__init__() 
        self.channels = channels 
        self.k = k
        self.conv_type = conv_type 
        assert self.conv_type in ['1d', '2d'] 

        self.fc_1 = nn.Linear(channels, channels // reduction) 
        self.relu = nn.ReLU(inplace = True) 
        self.fc_2 = nn.Linear(channels // reduction, 2 * k) 
        self.sigmoid = nn.Sigmoid() 
        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor(
            [1.] + [0.]*(2*k - 1)).float())
    def relu_coeff(self, x):
        theta = torch.mean(x, dim = -1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, dim = -1)   
        theta = self.fc_1(theta) 
        theta = self.relu(theta)
        theta = self.fc_2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta 
    
    def forward(self, x):
        raise NotImplementedError
    

class DynamicReLUA(DynamicReLU): 
    def __init__(self, channels, k = 2, reduction = 4, conv_type = '2d'):
        super(DynamicReLUA, self).__init__(channels, k, reduction, conv_type) 
        # self.fc_2 = nn.Linear(channels // reduction, k) 

    def forward(self, x): 
        print(x.shape)
        assert x.size(1) == self.channels 
        theta = self.relu_coeff(x) 
        ## x = bacth , channel, h, w 
        ## theta = batch, 2k 
        relu_coefs = theta.view(-1, 2 * self.k) * self.lambdas + self.init_v  
        ## relu_coefs = batch, 2k 
        x_perm = x.transpose(0, -1).unsqueeze(-1) 
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result

class DynamicReLUB(DynamicReLU): 
    def __init__(self, channels, k = 2, reduction = 4, conv_type = '2d'):
        super(DynamicReLUB, self).__init__(channels, k, reduction, conv_type) 
        self.fc_2 = nn.Linear(channels // reduction, 2*k*channels)
        # self.fc_2 = nn.Linear(channels // reduction, k) 

    def forward(self, x): 
        print(x.shape)
        assert x.size(1) == self.channels 
        theta = self.relu_coeff(x) 
        ## x = bacth , channel, h, w 
        ## theta = batch, 2k 
        relu_coefs = theta.view(-1, self.channels, 2 *
                                self.k) * self.lambdas + self.init_v 
        print(relu_coefs.shape)
        
        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :,
                                         :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :,
                                         :self.k] + relu_coefs[:, :, self.k:]
            
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result

        

    


    