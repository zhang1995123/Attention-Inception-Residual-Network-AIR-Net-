
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from se_module import SELayer

def inception_3(pretrained=False, progress=True, **kwargs):
    
    model = Inception(**kwargs)
    
    return model


class Inception(nn.Module):

    def __init__(self, num_classes=10, inception_blocks=None):
        super(Inception, self).__init__()
        
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionOut

        self.Conv3d_1a_3x3 = BasicConv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1)

        self.Conv3d_2a_3x3 = BasicConv3d(16, 32, kernel_size=3, stride=1, padding=1)

        self.Conv3d_3a_1x1 = BasicConv3d(32, 64, kernel_size=1, padding=0)
        self.Conv3d_3b_3x3 = BasicConv3d(64, 128, kernel_size=3, padding=1)
        
        self.aux = InceptionAux(128)

        self.Mixed_a1 = inception_a(128)
        self.Mixed_a2 = inception_a(128)
        self.Mixed_a3 = inception_a(128)
        self.Mixed_b1 = inception_b(128)  # 128 to 256
        self.Mixed_c1 = inception_c(256)
        self.Mixed_c2 = inception_c(128)
        self.Mixed_c3 = inception_c(128)

        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 16 x 128 
        x = self.Conv3d_1a_3x3(x)
        # 16 x 64
        x = self.Conv3d_2a_3x3(x)
        # 16 x 32
        x = F.max_pool3d(x, kernel_size=3, stride=(1, 2, 2), padding=1)

        # 16 x 32
        x = self.Conv3d_3a_1x1(x)
        x = self.Conv3d_3b_3x3(x)
        x = F.max_pool3d(x, kernel_size=3, stride=(1, 2, 2), padding=1)
        # print(x.shape)

        aux = self.aux(x)
        x = self.Mixed_a1(x)
        x = self.Mixed_a2(x)
        x = self.Mixed_a3(x)

        x = self.Mixed_b1(x)
        # print(x.shape)
        x = self.Mixed_c1(x)
        x = self.Mixed_c2(x)
        x = self.Mixed_c3(x)

        # 4 x 4 x 4
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)

        # N x 256
        x = self.dropout(x)
        x = self.fc(x)
        # x = self.dropout(x)

        return (x+aux)/2


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels=32, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv3d
        self.branch1x1 = conv_block(in_channels, out_channels, kernel_size=1, padding=0)

        self.branch3x3_1 = conv_block(in_channels, out_channels//2, kernel_size=1, padding=0)
        self.branch3x3_2 = conv_block(out_channels//2, out_channels, kernel_size=3, padding=1)

        self.branch5x5_1 = conv_block(in_channels, out_channels//2, kernel_size=1, padding=0)
        self.branch5x5_2 = conv_block(out_channels//2, out_channels, kernel_size=3, padding=1)
        self.branch5x5_3 = conv_block(out_channels, out_channels, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, out_channels, kernel_size=1, padding=0)

        self.sample = BasicConv3d(in_channels, out_channels*4, kernel_size=1)
        self.se = SELayer(out_channels*4)

    def _forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.se(outputs)
        
        if x.shape[1] != outputs.shape[1]:
            x = self.sample(x)
            
        outputs = x + outputs
        outputs = F.relu(outputs, inplace=True)
        return outputs


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()

        self.branch_1 = BasicConv3d(in_channels, 256, kernel_size=1, stride=2)

        self.branch_3_1 = BasicConv3d(in_channels, 64, kernel_size=1)
        self.branch_3_2 = BasicConv3d(64, 128, kernel_size=3, padding=1, stride=2)

        self.branch_pool = BasicConv3d(in_channels, 128, kernel_size=1)


    def _forward(self, x):
        # branch_1 = self.branch_1(x)

        branch_3 = self.branch_3_1(x)
        branch_3 = self.branch_3_2(branch_3)

        branch_pool = self.branch_pool(x)
        branch_pool = F.max_pool3d(branch_pool, kernel_size=3, padding=1, stride=2)

        outputs = [branch_3, branch_pool]
        return outputs

    def forward(self, x):
        branch_res = self.branch_1(x)
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = branch_res + outputs
        # outputs = F.relu(outputs, inplace=True)
        return outputs


class InceptionAux(nn.Module):
    def __init__(self, in_channels):
        super(InceptionAux, self).__init__()
        self.conv0 = SeparableConv3d(in_channels, 128, kernel_size=3, padding=1)
        self.conv1 = SeparableConv3d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool3d(x, (1,1,1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class InceptionOut(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionOut, self).__init__()
        self.group_conv = SeparableConv3d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.sample = BasicConv3d(in_channels, 128, kernel_size=1)
        self.se = SELayer(128)

    def forward(self, x):
        outputs = self.group_conv(x)
        outputs = self.se(outputs)

        if x.shape[1] != outputs.shape[1]:
            x = self.sample(x)

        outputs = x + outputs
        outputs = self.relu(outputs)
        return outputs


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(SeparableConv3d, self).__init__()

        self.conv = nn.Conv3d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=32, bias=False)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.pointwise = nn.Conv3d(inplanes, outplanes, 1, 1, 0, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(outplanes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return F.relu(x, inplace=True)


if __name__=='__main__':
    model = inception_3()
    model = model.cuda()
    # print(model)

    in_ = torch.randn(16, 3, 16, 128, 128)
    in_ = in_.cuda()
    out_ = model(in_)
    print(out_.shape)
