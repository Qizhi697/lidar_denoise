import numpy as np
import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# pretrained_models = {
#     'kitti': {
#         'url': 'https://github.com/TheCodez/pytorch-WeatherNet/releases/download/0.1/WeatherNet_45.5-75c06618.pth',
#         'num_classes': 4
#     }
# }


def WeatherNet(pretrained=None, num_classes=13):
    """Constructs a WeatherNet model.

    Args:
        pretrained (string): If not ``None``, returns a pre-trained model. Possible values: ``kitti``.
        num_classes (int): number of output classes. Automatically set to the correct number of classes
            if ``pretrained`` is specified.
    """
    # if pretrained is not None:
    #     model = WeatherNet(pretrained_models[pretrained]['num_classes'])
    #     model.load_state_dict(hub.load_state_dict_from_url(pretrained_models[pretrained]['url']))
    #     return model

    model = WeatherNet(num_classes)
    return model


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class Basic_conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(Basic_conv_layer, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(self.bn(x))


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class WeatherNet(nn.Module):
    """
    Implements WeatherNet model from
    `"CNN-based Lidar Point Cloud De-Noising in Adverse Weather"
    <https://arxiv.org/pdf/1912.03874.pdf>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, num_classes=4):
        super(WeatherNet, self).__init__()

        self.lila1 = LiLaBlock(2, 32)
        self.lila2 = LiLaBlock(32, 64)
        self.lila3 = LiLaBlock(64, 96)
        self.lila4 = LiLaBlock(96, 96)
        self.drop_layer = nn.Dropout()
        self.lila5 = LiLaBlock(96, 64)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, distance, reflectivity):
        # print("distance: '{}'".format(distance.shape))
        # print("reflectivity: '{}'".format(reflectivity.shape))
        x = torch.cat([distance, reflectivity], 1)
        x = self.lila1(x)
        x = self.lila2(x)
        x = self.lila3(x)
        x = self.lila4(x)
        x = self.drop_layer(x)
        x = self.lila5(x)

        x = self.classifier(x)

        return x


class LiLaBlock(nn.Module):

    def __init__(self, in_channels, n):
        super(LiLaBlock, self).__init__()

        self.branch1 = BasicConv2d(in_channels, n, kernel_size=(7, 3), padding=(2, 0), stride=(1, 1))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3, stride=(1, 1))
        self.branch3 = BasicConv2d(in_channels, n, kernel_size=3, dilation=(2, 2), padding=1, stride=(1, 1))
        self.branch4 = BasicConv2d(in_channels, n, kernel_size=(3, 7), padding=(0, 2), stride=(1, 1))
        self.conv = BasicConv2d(n * 4, n, kernel_size=1, padding=1, stride=(1, 1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        output = self.conv(output)

        return output


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)
        # return F.relu(x, inplace=True)


class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=8):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

        # groups all attention
        # self.cweight = Parameter(torch.zeros(1, channel // groups, 1, 1))
        # self.cbias = Parameter(torch.ones(1, channel // groups, 1, 1))
        # self.sweight = Parameter(torch.zeros(1, channel // groups, 1, 1))
        # self.sbias = Parameter(torch.ones(1, channel // groups, 1, 1))
        #
        # self.sigmoid = nn.Sigmoid()
        # self.gn = nn.GroupNorm(channel // groups, channel // groups)

        # no group all attention
        # self.cweight = Parameter(torch.zeros(1, channel, 1, 1))
        # self.cbias = Parameter(torch.ones(1, channel, 1, 1))
        # self.sweight = Parameter(torch.zeros(1, channel, 1, 1))
        # self.sbias = Parameter(torch.ones(1, channel, 1, 1))
        #
        # self.sigmoid = nn.Sigmoid()
        # self.gn = nn.GroupNorm(channel, channel)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)

        # all attention
        # xc = self.avg_pool(x)
        # xc = self.cweight * xc + self.cbias
        #
        # xs = self.gn(x)
        # xs = self.sweight * xs + self.sbias
        #
        # out = x * self.sigmoid(xc) * self.sigmoid(xs)

        # half channel half spatial
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


# class WIRW(nn.Module):
#     def __init__(self, nf, ks, groups):
#         super(WIRW, self).__init__()
#         self.conv1_1 = BasicConv2d(nf, nf, kernel_size=(1, 1))
#         # self.act = nn.ReLU(inplace=True)
#         self.act = nn.ReLU()
#         self.conv1_2 = BasicConv2d(nf, nf, kernel_size=(1, 1))
#         self.conv3_1 = BasicConv2d(nf, nf, kernel_size=(ks, ks), stride=(1, 1), padding=(ks - 1) // 2)
#         self.sa_attn = sa_layer(nf, groups)
#
#     def forward(self, x):
#         x1 = self.act(self.conv1_1(x))
#         x1 = self.conv1_2(x1)
#         x1 = self.conv3_1(x1)
#         x1 = self.sa_attn(x1)
#
#         out = x1 + x
#         return out
#
#
# class WCRW(nn.Module):
#     def __init__(self, nf, ks, groups):
#         super(WCRW, self).__init__()
#         self.conv1_1 = BasicConv2d(nf, nf, kernel_size=1)
#         # self.act = nn.ReLU(inplace=True)
#         self.act = nn.ReLU()
#         self.conv1_2 = BasicConv2d(nf, nf, kernel_size=1)
#         self.conv3_1 = BasicConv2d(nf, nf, kernel_size=(ks, ks), stride=(1, 1), padding=(ks - 1) // 2)
#         self.conv3_2 = BasicConv2d(nf, nf, kernel_size=(ks, ks), stride=(1, 1), padding=(ks - 1) // 2)
#         self.sa_attn = sa_layer(nf, groups)
#
#     def forward(self, x):
#         x1 = self.act(self.conv1_1(x))
#         x1 = self.conv1_2(x1)
#         x1 = self.conv3_1(x1)
#         x1 = self.sa_attn(x1)
#
#         res1 = self.conv3_2(x)
#         out = x1 + res1
#         return out
#
#
# class SCF(nn.Module):
#     def __init__(self, nf, ks, groups):
#         super(SCF, self).__init__()
#         self.wcrw = WCRW(2 * nf, ks, groups)
#         self.conv1 = BasicConv2d(2 * nf, nf, kernel_size=(1, 1))
#         self.sigmoid = nn.Sigmoid()
#         self.conv2 = BasicConv2d(2 * nf, nf, kernel_size=(1, 1))
#         self.act = nn.ReLU()
#         # self.act = nn.ReLU(inplace=True)
#
#     def forward(self, inp1, inp2):
#         inp = torch.cat([inp1, inp2], dim=1)
#         inp_res = self.conv2(self.act(self.wcrw(inp)))
#         inp_attn = self.sigmoid(self.conv1(inp))
#         out = inp_res + torch.mul(inp_attn, inp2)
#         return out
#
#
# class Single_Block(nn.Module):
#     def __init__(self, nf, ks, groups):
#         super(Single_Block, self).__init__()
#         self.act = nn.ReLU()
#         self.wirw = WIRW(nf, ks, groups)
#         self.scf = SCF(nf, ks, groups)
#
#     def forward(self, fea):
#         fea_enh = self.wirw(fea)
#         fea_dis = self.act(self.scf(fea, fea_enh)) + fea
#         return fea_dis

class WIRW(nn.Module):
    def __init__(
            self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True)):
        super(WIRW, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, n_feats * 6, kernel_size=1, padding=0)))
        # body.append(nn.BatchNorm2d(n_feats))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats * 6, n_feats // 2, kernel_size=1, padding=0)))
        # body.append(nn.BatchNorm2d(n_feats))
        body.append(
            wn(nn.Conv2d(n_feats // 2, n_feats, kernel_size=3, padding=1)))
        # body.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*body)
        self.SAlayer = sa_layer(n_feats)

    def forward(self, x):
        y = self.res_scale(self.SAlayer(self.body(x))) + self.x_scale(x)
        return y


class WCRW(nn.Module):
    def __init__(
            self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True)):
        super(WCRW, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        body = []
        body.append(
            wn(nn.Conv2d(n_feats, n_feats * 6, kernel_size=1, padding=0)))
        # body.append(nn.BatchNorm2d(n_feats))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats * 6, n_feats // 2, kernel_size=1, padding=0)))
        # body.append(nn.BatchNorm2d(n_feats))
        body.append(
            wn(nn.Conv2d(n_feats // 2, n_feats // 2, kernel_size=3, padding=1)))
        # body.append(nn.BatchNorm2d(n_feats // 2))

        self.body = nn.Sequential(*body)
        self.SAlayer = sa_layer(n_feats // 2)
        self.conv = nn.Conv2d(n_feats, n_feats // 2, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.res_scale(self.SAlayer(self.body(x))) + self.x_scale(self.conv(x))
        return y


class SCF(nn.Module):
    def __init__(self, nf):
        super(SCF, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.scale_x1 = Scale(1)
        self.scale_x2 = Scale(1)
        self.fuse1 = WCRW(nf * 2)
        self.fuse2 = nn.Conv2d(2 * nf, nf, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, inp1, inp2):
        inp = torch.cat([self.scale_x1(inp1), self.scale_x2(inp2)], dim=1)
        out1 = self.fuse1(inp)
        out2 = self.sigmoid(self.fuse2(inp)) * inp2
        out = out1 + out2
        return out


class Single_Block(nn.Module):
    def __init__(self, nf):
        super(Single_Block, self).__init__()
        self.act = nn.ReLU()
        self.wirw = WIRW(nf)
        self.scf = SCF(nf)
        # self.wirw = WIRW(nf, ks, groups)
        # self.scf = SCF(nf, ks, groups)

    def forward(self, fea):
        fea_enh = self.wirw(fea)
        fea_dis = self.scf(fea, fea_enh) + fea
        # fea_dis = self.act(self.scf(fea, fea_enh)) + fea
        return fea_dis


class Var_Nine(nn.Module):
    def __init__(self, num_classes=4, k_dis=True):
        super(Var_Nine, self).__init__()
        self.k_dis = k_dis
        # self.lila1 = LiLaBlock(3, 32)
        if self.k_dis:
            self.lila1 = BasicConv2d(3, 32, kernel_size=1, padding=0)
            # self.lila1 = LiLaBlock(3, 32)
        else:
            self.lila1 = BasicConv2d(2, 32, kernel_size=1, padding=0)
            # self.lila1 = LiLaBlock(2, 32)
        self.block1 = Single_Block(32)
        # self.lila2 = LiLaBlock(32, 64)
        self.lila2 = BasicConv2d(32, 64, kernel_size=1, padding=0)
        self.block2 = Single_Block(64)
        # self.drop_layer = nn.Dropout()
        # self.lila3 = LiLaBlock(64, 32)
        self.lila3 = BasicConv2d(64, 32, kernel_size=1, padding=0)
        self.block3 = Single_Block(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=(1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, param_lists):
        if self.k_dis:
            assert len(param_lists) == 3
            x = torch.cat([param_lists[0], param_lists[1], param_lists[2]], 1)
        else:
            assert len(param_lists) == 2
            x = torch.cat([param_lists[0], param_lists[1]], 1)
        x = self.lila1(x)
        x = self.block1(x)
        x = self.lila2(x)
        x = self.block2(x)
        # x = self.drop_layer(x)
        x = self.lila3(x)
        x = self.block3(x)
        x = self.classifier(x)

        return x
