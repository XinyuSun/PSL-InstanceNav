import math
import torch.nn as nn
from einops import reduce
from habitat import logger

__all__ = ["ResNet", "resnet50", "midfusion_resnet50", "resnet101"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        ngroups,
        stride=1,
        downsample=None,
        dropout_prob=0.0,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(ngroups, planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(ngroups, planes)
        self.downsample = downsample
        self.relu = nn.ReLU(True)

        gn_init(self.bn1)
        gn_init(self.bn2, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, ngroups, stride=1, downsample=None, dropout_prob=0.0
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(ngroups, planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.GroupNorm(ngroups, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(ngroups, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_prob)

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.dropout(self.relu(out))

        return out


def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2.0 / n))


def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0.0 if zero_init else 1.0)
    m.bias.data.zero_()


class ResNet(nn.Module):
    def __init__(
        self, in_channels, base_planes, ngroups, block, layers, dropout_prob=0.0
    ):
        self.inplanes = base_planes
        super(ResNet, self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Conv2d(
            in_channels, base_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.GroupNorm(ngroups, base_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(
            block, ngroups, base_planes * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, ngroups, base_planes * 4, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, ngroups, base_planes * 8, layers[3], stride=2
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2**5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_init(m)
        gn_init(self.bn1)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )
            m = downsample[1]
            assert isinstance(m, nn.GroupNorm)
            gn_init(m)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                downsample,
                dropout_prob=self.dropout_prob,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, ngroups, dropout_prob=self.dropout_prob)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def __init__(self, cond_c=[32,64,128,256], obs_c=[32,64,128,256], res=[32,16,8,4], reduction="none"):
        super().__init__()
        self.reduction = reduction
        if self.reduction == "none":
            self.f = nn.ModuleList([
                nn.Conv2d(cond_c[i], obs_c[i], kernel_size=1, stride=1)
                for i in range(len(cond_c))
            ])
            self.h = nn.ModuleList([
                nn.Conv2d(cond_c[i], obs_c[i], kernel_size=1, stride=1)
                for i in range(len(cond_c))
            ])
        elif self.reduction == "global":
            logger.info("init film layer with global pool {} ==> {}".format(res, 1))
            self.f = nn.ModuleList([
                nn.Linear(cond_c[i], obs_c[i])
                for i in range(len(cond_c))
            ])
            self.h = nn.ModuleList([
                nn.Linear(cond_c[i], obs_c[i])
                for i in range(len(cond_c))
            ])
        else:
            raise TypeError("not implemented")
        
    def forward(self, x, conds, i):
        if self.reduction == "none":
            gammas = self.f[i](conds).view_as(x)
            betas = self.h[i](conds).view_as(x)
        elif self.reduction == "global":
            gammas = self.f[i](reduce(conds, "b c h w -> b c", reduction="mean"))[:,:,None,None].expand_as(x)
            betas = self.h[i](reduce(conds, "b c h w -> b c", reduction="mean"))[:,:,None,None].expand_as(x)
        else:
            raise TypeError("not implemented")
        return (gammas * x) + betas


class MidFusionResNet(ResNet):
    def __init__(self, reduction, film_layers, cond_c, obs_c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.film = FiLM(reduction=reduction, cond_c=cond_c, obs_c=obs_c)
        self.film_layers = film_layers
        logger.info(f"filmed resnet encoder with layers: {film_layers}, reduction: {reduction}")

    def forward(self, x, g):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        self.layers = nn.ModuleList((
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ))

        for i,l in enumerate(self.layers):
            x = l(x)
            if i in self.film_layers:
                x = self.film(x, g[i], i)

        return x


def resnet18(in_channels, base_planes, ngroups, dropout_prob=0.0, **kwargs):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        BasicBlock,
        [2, 2, 2, 2],
        dropout_prob=dropout_prob,
    )
    # model = ResNet(in_channels, base_planes, ngroups, Bottleneck, [2, 2, 2, 2])
    return model


def resnet50(in_channels, base_planes, ngroups, dropout_prob=0.0, **kwargs):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        Bottleneck,
        [3, 4, 6, 3],
        dropout_prob=dropout_prob,
    )
    return model


def midfusion_resnet50(in_channels, base_planes, ngroups, dropout_prob=0.0, film_reduction="none", film_layers=[0,], cond_c=[256,], obs_c=[128,], **kwargs):
    model = MidFusionResNet(
        film_reduction,
        film_layers,
        cond_c,
        obs_c,
        in_channels,
        base_planes,
        ngroups,
        Bottleneck,
        [3, 4, 6, 3],
        dropout_prob=dropout_prob,
    )
    return model


def resnet101(in_channels, base_planes, ngroups, dropout_prob=0.0, **kwargs):
    model = ResNet(
        in_channels,
        base_planes,
        ngroups,
        Bottleneck,
        [3, 4, 23, 3],
        dropout_prob=dropout_prob,
    )
    return model
