# Revival: Convert Minecraft to Real World
# CS 175: Project in Artificial Intelligence (Spring 2019)
#
# networks.py
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(conv_dim*2),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=4, stride=2, padding=1),
                  nn.InstanceNorm2d(conv_dim*4),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=4, stride=1, padding=1),
                  nn.InstanceNorm2d(conv_dim*8),
                  nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(conv_dim*8, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


class ResidualBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(conv_dim, conv_dim, kernel_size=3),
                                   nn.InstanceNorm2d(conv_dim),
                                   nn.ReLU(True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(conv_dim, conv_dim, kernel_size=3),
                                   nn.InstanceNorm2d(conv_dim))

    def forward(self, x):
        out = x + self.block(x)
        return out


class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64):
        super(CycleGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, conv_dim, kernel_size=7),
                 nn.InstanceNorm2d(conv_dim),
                 nn.ReLU(True)]

        model += [nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(conv_dim*2),
                  nn.ReLU(True)]

        model += [nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(conv_dim*4),
                  nn.ReLU(True)]

        # for i in range(9):
        #     model += [ResidualBlock(conv_dim*4)]
        model += [ResidualBlock(conv_dim * 4) for i in range(6)]

        model += [nn.ConvTranspose2d(conv_dim*4, conv_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(conv_dim*2),
                  nn.ReLU(True)]

        model += [nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(conv_dim),
                  nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(conv_dim, 3, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
