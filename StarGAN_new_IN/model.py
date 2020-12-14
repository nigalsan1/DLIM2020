import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, c_dim):
        super(ResidualBlock, self).__init__()
        self.L1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.L2 = ccbn(input_size=c_dim,output_size=dim_out)
        self.L3 = nn.ReLU(inplace=True)
        self.L4 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.L5 = ccbn(input_size=c_dim,output_size=dim_out)

    def forward(self, x, c):
        x0 = x
        x = self.L1(x)
        x = self.L2(x,c)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x,c)
        return x0 + x

class ccbn(nn.Module): #Not sure if the nn.Module is needed
    def __init__(self, output_size, input_size, eps=1e-5, momentum=0.1):
        super(ccbn, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = nn.Embedding(input_size, output_size)
        self.bias = nn.Embedding(input_size, output_size)

        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var',  torch.ones(output_size)) 
        
        
    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)

        out = nn.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                                self.training, 0.1, self.eps)

        return out * gain + bias




class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        self.L1 = nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.L2 = ccbn(output_size = conv_dim, input_size = c_dim) #Input & Output Size?
        self.L3 = nn.ReLU(inplace=True)

        # Down-sampling layers.
        curr_dim = conv_dim
        self.L4 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.L5 = ccbn(output_size = curr_dim*2, input_size = c_dim) 
        self.L6 = nn.ReLU(inplace=True)

        curr_dim = curr_dim * 2

        self.L7 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.L8 = ccbn(output_size = curr_dim*2, input_size = c_dim)
        self.L9 = nn.ReLU(inplace=True)

        #Bottleneck Layers
        self.L10 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, c_dim=c_dim)
        self.L12 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, c_dim=c_dim)
        self.L13 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, c_dim=c_dim)
        self.L14 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, c_dim=c_dim)
        self.L15 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, c_dim=c_dim)

        # Up-sampling layers.
        self.L16 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.L17 = ccbn(output_size = curr_dim//2, input_size = c_dim)
        self.L18 = nn.ReLU(inplace=True)

        curr_dim = curr_dim // 2

        self.L19 = nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False)
        self.L20 = ccbn(output_size = curr_dim//2, input_size = c_dim) 
        self.L21 = nn.ReLU(inplace=True)
        
        self.L22 = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.L23 = nn.Tanh()

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        
        x = self.L1(x)
        x = self.L2(x,c)
        x = self.L3(x)

        # Down-sampling layers.
        x = self.L4(x)
        x = self.L5(x,c)
        x = self.L6(x)
        x = self.L7(x)
        x = self.L8(x,c)
        x = self.L9(x)

        #Bottleneck Layers
        x = self.L10(x)
        x = self.L11(x)
        x = self.L12(x)
        x = self.L13(x)
        x = self.L14(x)
        x = self.L15(x)

        # Up-sampling layers.
        x = self.L16(x)
        x = self.L17(x,c)
        x = self.L18(x)
        x = self.L19(x)
        x = self.L20(x,c)
        x = self.L21(x)

        x = self.L22(x)
        x = self.L23(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
