#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from util import BinaryConv2d, TernaryConv2d, BinaryConv2d_FOBNN, TernaryConv2d_FOBNN, BNNActivationLayer

# CifarNet2
class BWNs_BC1_FOBNN_test_3(nn.Module):
    def __init__(self, class_num=10):
        super(BWNs_BC1_FOBNN_test_3, self).__init__()
        self.class_num = class_num
        self.conv1_1 = BinaryConv2d_FOBNN(3, 64, kernel_size=3, padding=0)
        self.bn_conv1_1 = nn.BatchNorm2d(64)
        self.activation1_1 = BNNActivationLayer()
        self.conv1_2 = BinaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = BNNActivationLayer()

        self.conv2_1 = BinaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv2_1 = nn.BatchNorm2d(64)
        self.activation2_1 = BNNActivationLayer()
        self.conv2_2 = BinaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv2_2 = nn.BatchNorm2d(64)
        self.activation2_2 = BNNActivationLayer()

        self.conv3_1 = BinaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv3_1 = nn.BatchNorm2d(64)
        self.activation3_1 = BNNActivationLayer()
        self.conv3_2 = BinaryConv2d_FOBNN(64, 64, kernel_size=1, padding=0)
        self.bn_conv3_2 = nn.BatchNorm2d(64)
        self.activation3_2 = BNNActivationLayer()
        self.conv3_3 = BinaryConv2d_FOBNN(64, 16, kernel_size=1, padding=0)
        self.bn_conv3_3 = nn.BatchNorm2d(16)
        self.activation3_3 = BNNActivationLayer()

        self.fc = BinaryConv2d_FOBNN(144, 10, kernel_size=1)
        self.bn_fc = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.activation1_1(self.bn_conv1_1(self.conv1_1(x)))
        x = self.activation1_2(self.bn_conv1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation2_1(self.bn_conv2_1(self.conv2_1(x)))
        x = self.activation2_2(self.bn_conv2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation3_1(self.bn_conv3_1(self.conv3_1(x)))
        x = self.activation3_2(self.bn_conv3_2(self.conv3_2(x)))
        x = self.activation3_3(self.bn_conv3_3(self.conv3_3(x)))

        x = x.view(-1, 144, 1, 1)
        x = self.fc(x)
        x = x.view(-1, 10)
        x = self.bn_fc(x)
        x = self.log_Softmax(x)

        return x

class TWNs_BC1_FOBNN_test_3(nn.Module):
    def __init__(self, class_num=10):
        super(TWNs_BC1_FOBNN_test_3, self).__init__()
        self.class_num = class_num
        self.conv1_1 = TernaryConv2d_FOBNN(3, 64, kernel_size=3, padding=0)
        self.bn_conv1_1 = nn.BatchNorm2d(64)
        self.activation1_1 = BNNActivationLayer()
        self.conv1_2 = TernaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = BNNActivationLayer()

        self.conv2_1 = TernaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv2_1 = nn.BatchNorm2d(64)
        self.activation2_1 = BNNActivationLayer()
        self.conv2_2 = TernaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv2_2 = nn.BatchNorm2d(64)
        self.activation2_2 = BNNActivationLayer()

        self.conv3_1 = TernaryConv2d_FOBNN(64, 64, kernel_size=3, padding=0)
        self.bn_conv3_1 = nn.BatchNorm2d(64)
        self.activation3_1 = BNNActivationLayer()
        self.conv3_2 = TernaryConv2d_FOBNN(64, 64, kernel_size=1, padding=0)
        self.bn_conv3_2 = nn.BatchNorm2d(64)
        self.activation3_2 = BNNActivationLayer()
        self.conv3_3 = TernaryConv2d_FOBNN(64, 16, kernel_size=1, padding=0)
        self.bn_conv3_3 = nn.BatchNorm2d(16)
        self.activation3_3 = BNNActivationLayer()

        self.fc = TernaryConv2d_FOBNN(144, 10, kernel_size=1)
        self.bn_fc = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.activation1_1(self.bn_conv1_1(self.conv1_1(x)))
        x = self.activation1_2(self.bn_conv1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation2_1(self.bn_conv2_1(self.conv2_1(x)))
        x = self.activation2_2(self.bn_conv2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation3_1(self.bn_conv3_1(self.conv3_1(x)))
        x = self.activation3_2(self.bn_conv3_2(self.conv3_2(x)))
        x = self.activation3_3(self.bn_conv3_3(self.conv3_3(x)))

        x = x.view(-1, 144, 1, 1)
        x = self.fc(x)
        x = x.view(-1, 10)
        x = self.bn_fc(x)
        x = self.log_Softmax(x)

        return x



# CifarNet1
class BWNs_BC2_FOBNN_scale3_test_3(nn.Module):
    def __init__(self, class_num=10):
        super(BWNs_BC2_FOBNN_scale3_test_3, self).__init__()
        self.class_num = class_num
        self.conv1_1 = BinaryConv2d_FOBNN(3, 48, kernel_size=3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(48)
        self.activation1_1 = BNNActivationLayer()
        self.conv1_2 = BinaryConv2d_FOBNN(48, 48, kernel_size=3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(48)
        self.activation1_2 = BNNActivationLayer()
        self.conv1_3 = BinaryConv2d_FOBNN(48, 48, kernel_size=3, padding=1)
        self.bn_conv1_3 = nn.BatchNorm2d(48)
        self.activation1_3 = BNNActivationLayer()

        self.conv2_1 = BinaryConv2d_FOBNN(48, 96, kernel_size=3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(96)
        self.activation2_1 = BNNActivationLayer()
        self.conv2_2 = BinaryConv2d_FOBNN(96, 96, kernel_size=3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(96)
        self.activation2_2 = BNNActivationLayer()
        self.conv2_3 = BinaryConv2d_FOBNN(96, 96, kernel_size=3, padding=1)
        self.bn_conv2_3 = nn.BatchNorm2d(96)
        self.activation2_3 = BNNActivationLayer()

        self.conv3_1 = BinaryConv2d_FOBNN(96, 144, kernel_size=3, padding=0)
        self.bn_conv3_1 = nn.BatchNorm2d(144)
        self.activation3_1 = BNNActivationLayer()
        self.conv3_2 = BinaryConv2d_FOBNN(144, 144, kernel_size=3, padding=0)
        self.bn_conv3_2 = nn.BatchNorm2d(144)
        self.activation3_2 = BNNActivationLayer()
        self.conv3_3 = BinaryConv2d_FOBNN(144, 192, kernel_size=3, padding=0)
        self.bn_conv3_3 = nn.BatchNorm2d(192)
        self.activation3_3 = BNNActivationLayer()

        self.fc1 = BinaryConv2d_FOBNN(192, 10, kernel_size=1)
        self.bn_fc = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.activation1_1(self.bn_conv1_1(self.conv1_1(x)))
        x = self.activation1_2(self.bn_conv1_2(self.conv1_2(x)))
        x = self.activation1_3(self.bn_conv1_3(self.conv1_3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation2_1(self.bn_conv2_1(self.conv2_1(x)))
        x = self.activation2_2(self.bn_conv2_2(self.conv2_2(x)))
        x = self.activation2_3(self.bn_conv2_3(self.conv2_3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation3_1(self.bn_conv3_1(self.conv3_1(x)))
        x = self.activation3_2(self.bn_conv3_2(self.conv3_2(x)))
        x = self.activation3_3(self.bn_conv3_3(self.conv3_3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 192, 1, 1)
        x = self.fc1(x)
        x = x.view(-1, 10)
        x = self.bn_fc(x)
        x = self.log_Softmax(x)

        x = x.view(-1, 10)

        return x



class TWNs_BC2_FOBNN_scale3_test_3(nn.Module):
    def __init__(self, class_num=10):
        super(TWNs_BC2_FOBNN_scale3_test_3, self).__init__()
        self.class_num = class_num
        self.conv1_1 = TernaryConv2d_FOBNN(3, 48, kernel_size=3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(48)
        self.activation1_1 = BNNActivationLayer()
        self.conv1_2 = TernaryConv2d_FOBNN(48, 48, kernel_size=3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(48)
        self.activation1_2 = BNNActivationLayer()
        self.conv1_3 = TernaryConv2d_FOBNN(48, 48, kernel_size=3, padding=1)
        self.bn_conv1_3 = nn.BatchNorm2d(48)
        self.activation1_3 = BNNActivationLayer()

        self.conv2_1 = TernaryConv2d_FOBNN(48, 96, kernel_size=3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(96)
        self.activation2_1 = BNNActivationLayer()
        self.conv2_2 = TernaryConv2d_FOBNN(96, 96, kernel_size=3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(96)
        self.activation2_2 = BNNActivationLayer()
        self.conv2_3 = TernaryConv2d_FOBNN(96, 96, kernel_size=3, padding=1)
        self.bn_conv2_3 = nn.BatchNorm2d(96)
        self.activation2_3 = BNNActivationLayer()

        self.conv3_1 = TernaryConv2d_FOBNN(96, 144, kernel_size=3, padding=0)
        self.bn_conv3_1 = nn.BatchNorm2d(144)
        self.activation3_1 = BNNActivationLayer()
        self.conv3_2 = TernaryConv2d_FOBNN(144, 144, kernel_size=3, padding=0)
        self.bn_conv3_2 = nn.BatchNorm2d(144)
        self.activation3_2 = BNNActivationLayer()
        self.conv3_3 = TernaryConv2d_FOBNN(144, 192, kernel_size=3, padding=0)
        self.bn_conv3_3 = nn.BatchNorm2d(192)
        self.activation3_3 = BNNActivationLayer()

        self.fc1 = TernaryConv2d_FOBNN(192, 10, kernel_size=1)
        self.bn_fc = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.activation1_1(self.bn_conv1_1(self.conv1_1(x)))
        x = self.activation1_2(self.bn_conv1_2(self.conv1_2(x)))
        x = self.activation1_3(self.bn_conv1_3(self.conv1_3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation2_1(self.bn_conv2_1(self.conv2_1(x)))
        x = self.activation2_2(self.bn_conv2_2(self.conv2_2(x)))
        x = self.activation2_3(self.bn_conv2_3(self.conv2_3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation3_1(self.bn_conv3_1(self.conv3_1(x)))
        x = self.activation3_2(self.bn_conv3_2(self.conv3_2(x)))
        x = self.activation3_3(self.bn_conv3_3(self.conv3_3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 192, 1, 1)
        x = self.fc1(x)
        x = x.view(-1, 10)
        x = self.bn_fc(x)
        x = self.log_Softmax(x)

        x = x.view(-1, 10)

        return x


# CifarNet3
class BWNs_BC3_FOBNN_scale2_test_3(nn.Module):
    def __init__(self, class_num=10):
        super(BWNs_BC3_FOBNN_scale2_test_3, self).__init__()
        self.class_num = class_num
        self.conv1_1 = BinaryConv2d_FOBNN(3, 32, kernel_size=3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(32)
        self.activation1_1 = BNNActivationLayer()
        self.conv1_2 = BinaryConv2d_FOBNN(32, 64, kernel_size=3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = BNNActivationLayer()
        self.conv1_3 = BinaryConv2d_FOBNN(64, 64, kernel_size=3, padding=1)
        self.bn_conv1_3 = nn.BatchNorm2d(64)
        self.activation1_3 = BNNActivationLayer()

        self.conv2_1 = BinaryConv2d_FOBNN(64, 96, kernel_size=3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(96)
        self.activation2_1 = BNNActivationLayer()
        self.conv2_2 = BinaryConv2d_FOBNN(96, 128, kernel_size=3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(128)
        self.activation2_2 = BNNActivationLayer()
        self.conv2_3 = BinaryConv2d_FOBNN(128, 160, kernel_size=3, padding=1)
        self.bn_conv2_3 = nn.BatchNorm2d(160)
        self.activation2_3 = BNNActivationLayer()

        self.conv3_1 = BinaryConv2d_FOBNN(160, 192, kernel_size=3, padding=0)
        self.bn_conv3_1 = nn.BatchNorm2d(192)
        self.activation3_1 = BNNActivationLayer()
        self.conv3_2 = BinaryConv2d_FOBNN(192, 192, kernel_size=3, padding=0)
        self.bn_conv3_2 = nn.BatchNorm2d(192)
        self.activation3_2 = BNNActivationLayer()
        self.conv3_3 = BinaryConv2d_FOBNN(192, 256, kernel_size=3, padding=0)
        self.bn_conv3_3 = nn.BatchNorm2d(256)
        self.activation3_3 = BNNActivationLayer()

        self.fc = BinaryConv2d_FOBNN(256, 10, kernel_size=1)
        self.bn_fc = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn_conv1_1(x)
        x = self.activation1_1(x)
        x = self.conv1_2(x)
        x = self.bn_conv1_2(x)
        x = self.activation1_2(x)
        x = self.conv1_3(x)
        x = self.bn_conv1_3(x)
        x = self.activation1_3(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2_1(x)
        x = self.bn_conv2_1(x)
        x = self.activation2_1(x)
        x = self.conv2_2(x)
        x = self.bn_conv2_2(x)
        x = self.activation2_2(x)
        x = self.conv2_3(x)
        x = self.bn_conv2_3(x)
        x = self.activation2_3(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3_1(x)
        x = self.bn_conv3_1(x)
        x = self.activation3_1(x)
        x = self.conv3_2(x)
        x = self.bn_conv3_2(x)
        x = self.activation3_2(x)
        x = self.conv3_3(x)
        x = self.bn_conv3_3(x)
        x = self.activation3_3(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 256, 1, 1)
        x = self.fc(x)
        x = x.view(-1, 10)
        x = self.bn_fc(x)
        x = self.log_Softmax(x)

        return x


class TWNs_BC3_FOBNN_scale2_test_3(nn.Module):
    def __init__(self, class_num=10):
        super(TWNs_BC3_FOBNN_scale2_test_3, self).__init__()
        self.class_num = class_num
        self.conv1_1 = TernaryConv2d_FOBNN(3, 32, kernel_size=3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(32)
        self.activation1_1 = BNNActivationLayer()
        self.conv1_2 = TernaryConv2d_FOBNN(32, 64, kernel_size=3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = BNNActivationLayer()
        self.conv1_3 = TernaryConv2d_FOBNN(64, 64, kernel_size=3, padding=1)
        self.bn_conv1_3 = nn.BatchNorm2d(64)
        self.activation1_3 = BNNActivationLayer()

        self.conv2_1 = TernaryConv2d_FOBNN(64, 96, kernel_size=3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(96)
        self.activation2_1 = BNNActivationLayer()
        self.conv2_2 = TernaryConv2d_FOBNN(96, 128, kernel_size=3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(128)
        self.activation2_2 = BNNActivationLayer()
        self.conv2_3 = TernaryConv2d_FOBNN(128, 160, kernel_size=3, padding=1)
        self.bn_conv2_3 = nn.BatchNorm2d(160)
        self.activation2_3 = BNNActivationLayer()

        self.conv3_1 = TernaryConv2d_FOBNN(160, 192, kernel_size=3, padding=0)
        self.bn_conv3_1 = nn.BatchNorm2d(192)
        self.activation3_1 = BNNActivationLayer()
        self.conv3_2 = TernaryConv2d_FOBNN(192, 192, kernel_size=3, padding=0)
        self.bn_conv3_2 = nn.BatchNorm2d(192)
        self.activation3_2 = BNNActivationLayer()
        self.conv3_3 = TernaryConv2d_FOBNN(192, 256, kernel_size=3, padding=0)
        self.bn_conv3_3 = nn.BatchNorm2d(256)
        self.activation3_3 = BNNActivationLayer()

        self.fc = TernaryConv2d_FOBNN(256, 10, kernel_size=1)
        self.bn_fc = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn_conv1_1(x)
        x = self.activation1_1(x)
        x = self.conv1_2(x)
        x = self.bn_conv1_2(x)
        x = self.activation1_2(x)
        x = self.conv1_3(x)
        x = self.bn_conv1_3(x)
        x = self.activation1_3(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2_1(x)
        x = self.bn_conv2_1(x)
        x = self.activation2_1(x)
        x = self.conv2_2(x)
        x = self.bn_conv2_2(x)
        x = self.activation2_2(x)
        x = self.conv2_3(x)
        x = self.bn_conv2_3(x)
        x = self.activation2_3(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3_1(x)
        x = self.bn_conv3_1(x)
        x = self.activation3_1(x)
        x = self.conv3_2(x)
        x = self.bn_conv3_2(x)
        x = self.activation3_2(x)
        x = self.conv3_3(x)
        x = self.bn_conv3_3(x)
        x = self.activation3_3(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 256, 1, 1)
        x = self.fc(x)
        x = x.view(-1, 10)
        x = self.bn_fc(x)
        x = self.log_Softmax(x)

        return x





# MnistNet1
class BWN_BM2_3_scale4_test_3(nn.Module):
    def __init__(self):
        super(BWN_BM2_3_scale4_test_3, self).__init__()
        print("BINARY BM2 VERSION...")
        self.conv1 = BinaryConv2d_FOBNN(1, 20, kernel_size=5, stride=2)
        self.bn_conv1 = nn.BatchNorm2d(20)
        self.activation1 = BNNActivationLayer()

        self.fc1 = BinaryConv2d_FOBNN(2880, 400, kernel_size=1)
        self.bn_fc1 = nn.BatchNorm2d(400)
        self.activation3 = BNNActivationLayer()
        self.fc2 = BinaryConv2d_FOBNN(400, 10, kernel_size=1)
        self.bn_fc2 = nn.BatchNorm1d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation1(x)

        x = x.view(-1, 2880, 1, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.activation3(x)

        x = self.fc2(x)
        x = x.view(-1, 10)
        x = self.bn_fc2(x)
        x = self.log_Softmax(x)
        x = x.view(-1, 10)

        return x

class TWN_BM2_3_scale4_test_3(nn.Module):
    def __init__(self):
        super(TWN_BM2_3_scale4_test_3, self).__init__()
        print("TERNARY BM2 VERSION...")
        self.conv1 = TernaryConv2d_FOBNN(1, 20, kernel_size=5, stride=2)
        self.bn_conv1 = nn.BatchNorm2d(20)
        self.activation1 = BNNActivationLayer()

        self.fc1 = TernaryConv2d_FOBNN(2880, 400, kernel_size=1)
        self.bn_fc1 = nn.BatchNorm2d(400)
        self.activation3 = BNNActivationLayer()
        self.fc2 = TernaryConv2d_FOBNN(400, 10, kernel_size=1)
        self.bn_fc2 = nn.BatchNorm2d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation1(x)

        x = x.view(-1, 2880, 1, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.activation3(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = x.view(-1, 10)
        x = self.log_Softmax(x)
        x = x.view(-1, 10)

        return x


# MnistNet2
class BWN_BM3_FOBNN_scale2_v3(nn.Module):
    def __init__(self):
        super(BWN_BM3_FOBNN_scale2_v3, self).__init__()
        print("BINARY BM3 VERSION...")
        self.conv1 = BinaryConv2d_FOBNN(1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.activation1 = BNNActivationLayer()
        self.conv2 = BinaryConv2d_FOBNN(32, 32, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.activation2 = BNNActivationLayer()

        self.fc1 = BinaryConv2d_FOBNN(512, 200, kernel_size=1)
        self.bn_fc1 = nn.BatchNorm2d(200)
        self.activation3 = BNNActivationLayer()
        self.fc2 = BinaryConv2d_FOBNN(200, 10, kernel_size=1)
        self.bn_fc2 = nn.BatchNorm2d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.activation2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 512, 1, 1)
        x = self.activation3(self.bn_fc1(self.fc1(x)))

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = x.view(-1, 10)
        x = self.log_Softmax(x)

        return x

class TWN_BM3_FOBNN_scale2_v3(nn.Module):
    def __init__(self):
        super(TWN_BM3_FOBNN_scale2_v3, self).__init__()
        self.conv1 = TernaryConv2d_FOBNN(1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.activation1 = BNNActivationLayer()
        self.conv2 = TernaryConv2d_FOBNN(32, 32, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.activation2 = BNNActivationLayer()

        self.fc1 = TernaryConv2d_FOBNN(512, 200, kernel_size=1)
        self.bn_fc1 = nn.BatchNorm2d(200)
        self.activation3 = BNNActivationLayer()
        self.fc2 = TernaryConv2d_FOBNN(200, 10, kernel_size=1)
        self.bn_fc2 = nn.BatchNorm2d(10)
        self.log_Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.activation2(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 512, 1, 1)
        x = self.activation3(self.bn_fc1(self.fc1(x)))

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = x.view(-1, 10)
        x = self.log_Softmax(x)

        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn_conv1(x)), 2)
        x = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn_conv2(x)), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class VGG16(nn.Module):
    def __init__(self, class_num=10):
        super(VGG16, self).__init__()
        self.class_num = class_num
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_conv1_1 = nn.BatchNorm2d(64)
        self.activation1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_conv1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_conv2_1 = nn.BatchNorm2d(128)
        self.activation2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_conv2_2 = nn.BatchNorm2d(128)
        self.activation2_2 = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_conv3_1 = nn.BatchNorm2d(256)
        self.activation3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_conv3_2 = nn.BatchNorm2d(256)
        self.activation3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_conv3_3 = nn.BatchNorm2d(256)
        self.activation3_3 = nn.ReLU(inplace=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_conv4_1 = nn.BatchNorm2d(512)
        self.activation4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_conv4_2 = nn.BatchNorm2d(512)
        self.activation4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_conv4_3 = nn.BatchNorm2d(512)
        self.activation4_3 = nn.ReLU(inplace=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_conv5_1 = nn.BatchNorm2d(512)
        self.activation5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_conv5_2 = nn.BatchNorm2d(512)
        self.activation5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_conv5_3 = nn.BatchNorm2d(512)
        self.activation5_3 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(512, 256, bias=True)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.activation_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.activation_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.activation1_1(self.bn_conv1_1(self.conv1_1(x)))
        x = self.activation1_2(self.bn_conv1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation2_1(self.bn_conv2_1(self.conv2_1(x)))
        x = self.activation2_2(self.bn_conv2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation3_1(self.bn_conv3_1(self.conv3_1(x)))
        x = self.activation3_2(self.bn_conv3_2(self.conv3_2(x)))
        x = self.activation3_3(self.bn_conv3_3(self.conv3_3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation4_1(self.bn_conv4_1(self.conv4_1(x)))
        x = self.activation4_2(self.bn_conv4_2(self.conv4_2(x)))
        x = self.activation4_3(self.bn_conv4_3(self.conv4_3(x)))
        x = F.max_pool2d(x, 2)

        x = self.activation5_1(self.bn_conv5_1(self.conv5_1(x)))
        x = self.activation5_2(self.bn_conv5_2(self.conv5_2(x)))
        x = self.activation5_3(self.bn_conv5_3(self.conv5_3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 512)
        x = self.activation_fc1(self.bn_fc1(self.fc1(x)))
        x = self.activation_fc2(self.bn_fc2(self.fc2(x)))
        x = self.log_softmax(self.bn_fc3(self.fc3(x)))

        return x



# ConvMixer
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.depthwise = Residual(nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size,
                      groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        ))
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, class_num=10, dim=256, depth=8,
                 kernel_size=5, patch_size=2):
        super().__init__()
        self.class_num = class_num

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        self.blocks = nn.Sequential(
            *[ConvMixerLayer(dim, kernel_size) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, class_num),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

class ConvMixer_mnist(nn.Module):
    def __init__(self, class_num=10, dim=256, depth=8,
                 kernel_size=5, patch_size=2):
        super().__init__()
        self.class_num = class_num

        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        self.blocks = nn.Sequential(
            *[ConvMixerLayer(dim, kernel_size) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, class_num),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.head(x)
        return x




# Airbench94
class Airbench94ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bn1 = nn.BatchNorm2d(
            out_channels, eps=1e-12, momentum=0.4
        )
        self.bn1.weight.requires_grad = False

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels, eps=1e-12, momentum=0.4
        )
        self.bn2.weight.requires_grad = False

        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x


class Airbench94_CIFAR10(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()

        self.conv0 = nn.Conv2d(
            3, 24, kernel_size=2, stride=1,
            padding=0, bias=True
        )
        self.act0 = nn.GELU()

        self.block1 = Airbench94ConvGroup(24, 64)
        self.block2 = Airbench94ConvGroup(64, 256)
        self.block3 = Airbench94ConvGroup(256, 256)

        self.pool = nn.MaxPool2d(kernel_size=3)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256, class_num, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        x = x / 9.0

        return x

class Airbench94_mnist(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()

        self.conv0 = nn.Conv2d(
            1, 24, kernel_size=2, stride=1,
            padding=0, bias=True
        )
        self.act0 = nn.GELU()

        self.block1 = Airbench94ConvGroup(24, 64)
        self.block2 = Airbench94ConvGroup(64, 256)
        self.block3 = Airbench94ConvGroup(256, 256)

        self.pool = nn.MaxPool2d(kernel_size=3)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256, class_num, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.act0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        x = x / 9.0

        return x



