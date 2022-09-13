import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from torchvision import models
from Network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def conv_block(in_dim, out_dim, kernel_size, stride, padding, activation):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_dim),
        activation
    )
    return model


def LinearBlock(in_dim, out_dim, activation):
    model = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        activation,
        nn.Linear(out_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        activation
    )
    return model


class Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim, activation):
        super(Residual_Block, self).__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(mid_dim, in_dim, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.residual_block(x)
        out = out + x
        return out


class AdIN_Residual_Block(nn.Module):
    def __init__(self, in_dim, mid_dim, activation):
        super(AdIN_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        self.AdIN1 = AdaptiveInstanceNorm2d(mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.AdIN2 = AdaptiveInstanceNorm2d(in_dim)

        self.activation = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.AdIN1(out)
        out = self.conv2(out)
        out = self.AdIN2(out)
        out = self.activation(out)
        out = out + x
        return out


class ASPPModule(nn.Module):
    # https://github.com/jfzhang95/pytorch-deeplab-xception/tree/master/modeling
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Encoder_s(nn.Module):
    def __init__(self, in_dim, num_filter):
        super(Encoder_s, self).__init__()
        self.in_dim = in_dim
        self.num_filter = num_filter

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = conv_block(in_dim=self.in_dim, out_dim=self.num_filter * 1, kernel_size=3, stride=2, padding=1,
                                activation=self.activation)
        self.conv2 = conv_block(in_dim=self.num_filter * 1, out_dim=self.num_filter * 2, kernel_size=3, stride=1,
                                padding=1, activation=self.activation)
        self.conv3 = conv_block(in_dim=self.num_filter * 2, out_dim=self.num_filter * 2, kernel_size=3, stride=1,
                                padding=1, activation=self.activation)
        self.conv4 = conv_block(in_dim=self.num_filter * 2, out_dim=self.num_filter * 4, kernel_size=3, stride=2,
                                padding=1, activation=self.activation)

        self.ResBlock1 = Residual_Block(self.num_filter * 4, self.num_filter * 4, self.activation)
        self.ResBlock2 = Residual_Block(self.num_filter * 4, self.num_filter * 4, self.activation)
        self.ResBlock3 = Residual_Block(self.num_filter * 4, self.num_filter * 4, self.activation)
        self.ResBlock4 = Residual_Block(self.num_filter * 4, self.num_filter * 4, self.activation)

        self.ASSP1 = ASPPModule(inplanes=self.num_filter * 4, planes=self.num_filter * 2, kernel_size=1, padding=0,
                                dilation=0, BatchNorm=nn.BatchNorm2d)
        self.ASSP2 = ASPPModule(inplanes=self.num_filter * 2, planes=self.num_filter * 2, kernel_size=1, padding=0,
                                dilation=0, BatchNorm=nn.BatchNorm2d)
        self.ASSP3 = ASPPModule(inplanes=self.num_filter * 2, planes=self.num_filter * 2, kernel_size=3, padding=1,
                                dilation=1, BatchNorm=nn.BatchNorm2d)
        self.ASSP4 = ASPPModule(inplanes=self.num_filter * 2, planes=self.num_filter * 2, kernel_size=1, padding=0,
                                dilation=0, BatchNorm=nn.BatchNorm2d)
        self.ASSP5 = ASPPModule(inplanes=self.num_filter * 2, planes=self.num_filter * 2, kernel_size=3, padding=1,
                                dilation=1, BatchNorm=nn.BatchNorm2d)
        self.ASSP6 = ASPPModule(inplanes=self.num_filter * 2, planes=self.num_filter * 2, kernel_size=1, padding=0,
                                dilation=0, BatchNorm=nn.BatchNorm2d)
        self.ASSP7 = ASPPModule(inplanes=self.num_filter * 2, planes=self.num_filter * 8, kernel_size=3, padding=1,
                                dilation=1, BatchNorm=nn.BatchNorm2d)

        self.conv5 = conv_block(in_dim=self.num_filter * 8, out_dim=self.num_filter * 8, kernel_size=1, stride=1,
                                padding=0, activation=self.activation)

        self.init_weights()

    def forward(self, image1):
        x = torchvision.transforms.Grayscale()(image1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)

        x = self.ASSP1(x)
        x = self.ASSP2(x)
        x = self.ASSP3(x)
        x = self.ASSP4(x)
        x = self.ASSP5(x)
        x = self.ASSP6(x)
        x = self.ASSP7(x)

        x = self.conv5(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0)


class Encoder_a(nn.Module):
    def __init__(self, class_num):
        super(ReID, self).__init__()
        model = models.resnet50(pretrained=True)

        model.partpool = nn.AdaptiveMaxPool2d((4, 1))
        model.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        model.layer4[0].downsample[0].stride = (1, 1)
        model.layer4[0].conv2.stride = (1, 1)

        self.model = model
        self.classifier = LinearBlock(2048, class_num, nn.ReLU())

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        x = self.model.avgpool(x)

        x = x.view(x.size(0), x.size(1))
        f = f.view(f.size(0), f.size(1) * 4)

        x = self.classifier(x)

        return f, x


class Discriminator(nn.Module):
    def __init__(self, in_dim, num_filter):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.num_filter = num_filter

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = conv_block(in_dim=self.in_dim, out_dim=self.num_filter * 1, kernel_size=1, stride=1, padding=0,
                                activation=self.activation)
        self.conv2 = conv_block(in_dim=self.num_filter * 1, out_dim=self.num_filter * 1, kernel_size=3, stride=1,
                                padding=1, activation=self.activation)
        self.conv3 = conv_block(in_dim=self.num_filter * 1, out_dim=self.num_filter * 1, kernel_size=3, stride=2,
                                padding=1, activation=self.activation)
        self.conv4 = conv_block(in_dim=self.num_filter * 1, out_dim=self.num_filter * 1, kernel_size=3, stride=1,
                                padding=1, activation=self.activation)
        self.conv5 = conv_block(in_dim=self.num_filter * 1, out_dim=self.num_filter * 2, kernel_size=3, stride=2,
                                padding=1, activation=self.activation)

        self.ResBlock1 = Residual_Block(self.num_filter * 2, self.num_filter * 2, self.activation)
        self.ResBlock2 = Residual_Block(self.num_filter * 2, self.num_filter * 2, self.activation)
        self.ResBlock3 = Residual_Block(self.num_filter * 2, self.num_filter * 2, self.activation)
        self.ResBlock4 = Residual_Block(self.num_filter * 2, self.num_filter * 2, self.activation)

        self.conv6 = conv_block(in_dim=self.num_filter * 2, out_dim=1, kernel_size=1, stride=1,
                                padding=0, activation=self.activation)

    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.conv6(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_dim):
        super(Generator, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.ResBlock1 = AdIN_Residual_Block(in_dim, in_dim, self.activation)
        self.ResBlock2 = AdIN_Residual_Block(in_dim, in_dim, self.activation)
        self.ResBlock3 = AdIN_Residual_Block(in_dim, in_dim, self.activation)
        self.ResBlock4 = AdIN_Residual_Block(in_dim, in_dim, self.activation)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = conv_block(in_dim=in_dim, out_dim=int(in_dim / 2), kernel_size=5, stride=1, padding=2,
                                activation=self.activation)
        self.conv2 = conv_block(in_dim=int(in_dim / 2), out_dim=int(in_dim / 4), kernel_size=5, stride=1, padding=2,
                                activation=self.activation)
        self.conv3 = conv_block(in_dim=int(in_dim / 4), out_dim=int(in_dim / 4), kernel_size=3, stride=1,
                                padding=1, activation=self.activation)
        self.conv4 = conv_block(in_dim=int(in_dim / 4), out_dim=int(in_dim / 4), kernel_size=3, stride=1,
                                padding=1, activation=self.activation)
        self.conv5 = conv_block(in_dim=int(in_dim / 4), out_dim=3, kernel_size=1, stride=1,
                                padding=0, activation=self.activation)

    def forward(self, x):
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = None
        self.bias = None

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])


class DGNet(nn.Module):
    def __init__(self):
        super(DGNet, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.Es = Encoder_s(1, 16)
        self.Ea = Encoder_a()
        self.Discriminator = Discriminator(3, 32)
        self.Generator = Generator(128)

        self.w1 = LinearBlock(2048, 512, self.activation)
        self.w2 = LinearBlock(2048, 512, self.activation)
        self.w3 = LinearBlock(2048, 512, self.activation)
        self.w4 = LinearBlock(2048, 512, self.activation)

        self.b1 = LinearBlock(2048, 512, self.activation)
        self.b2 = LinearBlock(2048, 512, self.activation)
        self.b3 = LinearBlock(2048, 512, self.activation)
        self.b4 = LinearBlock(2048, 512, self.activation)

    def forward(self, image1, image2, image3):
        Xj_s = self.Es(image1)

        Xi_a, Xi_x = self.Ea(image2)
        Xi_s = self.Es(image2)

        Xt_a, Xt_x = self.Ea(image3)

        self.decode(Xi_a)
        Xj_gen = self.Generator(Xj_s)
        Xi_gen = self.Generator(Xi_s)

        self.decode(Xt_a)
        Xt_gen = self.Generator(Xi_s)
        return Xj_gen, Xi_gen, Xt_gen, Xj_s, Xi_a, Xi_s, Xt_a, Xi_x, Xt_x

    def decode(self, Ea):
        Ea = Ea.view(-1, 8192)
        ID1 = Ea[:, :2048]
        ID2 = Ea[:, 2048:4096]
        ID3 = Ea[:, 4096:6144]
        ID4 = Ea[:, 6144:]
        print(ID1.shape)
        adain_params_w = torch.cat((self.w1(ID1), self.w2(ID2), self.w3(ID3), self.w4(ID4)), 1)
        adain_params_b = torch.cat((self.b1(ID1), self.b2(ID2), self.b3(ID3), self.b4(ID4)), 1)
        self.assign_adain_params(adain_params_w, adain_params_b, self.Generator)

    def assign_adain_params(self, adain_params_w, adain_params_b, model):
        dim = 128
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:, :dim].contiguous()
                std = adain_params_w[:, :dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1) > dim:
                    adain_params_b = adain_params_b[:, dim:]
                    adain_params_w = adain_params_w[:, dim:]


class ReID(nn.Module):
    def __init__(self, class_num):
        super(ReID, self).__init__()
        model = models.resnet50(pretrained=True)

        model.partpool = nn.AdaptiveMaxPool2d((4, 1))
        model.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        model.layer4[0].downsample[0].stride = (1, 1)
        model.layer4[0].conv2.stride = (1, 1)

        self.model = model
        self.classifier = LinearBlock(2048, class_num, nn.ReLU())

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        f = self.model.partpool(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        f = f.view(f.size(0), f.size(1) * self.part)
        x = self.classifier(x)

        return f, x
