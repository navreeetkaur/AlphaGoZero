import numpy as np

import torch
import torch.nn as nn


def conv3x3(num_in, num_out, stride=1):
    return nn.Conv2d(in_channels=num_in, out_channels=num_out, stride=stride, kernel=3, padding=1, bias=False)

def conv1x1(num_in, num_out, stride):
    return nn.Conv2d(in_channels=num_in, out_channels=num_out, stride=stride, kernel=3, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, num_in, num_out, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(num_in, num_out, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_out, num_out, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_out)
        
        self.downsample = False
        if num_in != num_out or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(num_in, num_out, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(num_out)
            
    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out+=residual
        out = self.relu(out)
        return out


class AlphaGoNNet(nn.Module):
    def __init__(self, res_layers, board_size, action_size, num_channels=256):
        super(AlphaGoNNet, self).__init__()
        # convolutional block
        self.conv = conv3x3(3, num_channels, stride=1)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        # residual tower
        res_list = [ResidualBlock(num_channels, num_channels) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_list)
        # policy head
        self.p_conv = conv1x1(num_in=num_channels, num_out=2, stride=2)
        self.p_bn = nn.BatchNorm2d(num_features=2)
        self.p_fc = nn.Linear(2 * board_size ** 2, action_size)
        self.p_log_softmax = nn.LogSoftmax(dim=1)
        # value head
        self.v_conv = conv1x1(num_in=num_channels, num_out=1, stride=1)
        self.v_bn = nn.BatchNorm2d(num_features=1)
        self.v_fc1 = nn.Linear(board_size**2, num_channels)
        self.v_fc2 = nn.Linear(num_channels,1)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        # first conv block
        out = self.bn( self.conv(state) )
        out = self.relu(out)
        # residual tower
        out = self.res_layers(out)
        # policy network
        p_out = self.p_bn( self.p_conv(out) )
        p_out = self.relu(p_out)
        p_out = p_out.view(p_out.size(0), -1)
        p_out = self.p_fc(p_out)
        p_out = self.p_log_softmax(p_out)
        # value network
        v_out = self.v_bn( self.v_conv(out) )
        v_out = self.relu(v_out)
        v_out = v_out.view(v_out.size(0), -1)
        v_out = self.v_fc1(v_out)
        v_out = self.relu(v_out)
        v_out = self.v_fc2(v_out)
        v_out = self.tanh(v_out)
        
        return p,v



