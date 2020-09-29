import torch
import torch.nn as nn
import torchsummary as summary



def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,3), stride=stride,
                     padding=(0,1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_text_50(nn.Module):

    def __init__(self, args, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_text_50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if args.embedding_type=='BERT':
            self.inplanes = 768
        else:
            self.inplanes=args.embedding_size


        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = conv1x1(self.inplanes, 300)
        self.bn1 = norm_layer(300)
        self.relu = nn.ReLU(inplace=True)

        downsample1 = nn.Sequential(
            conv1x1(300, 256),
            norm_layer(256),
        )
        downsample2 = nn.Sequential(
            conv1x1(256, 512, stride=2),
            norm_layer(512),
        )
        downsample3 = nn.Sequential(
            conv1x1(512, 1024, stride=2),
            norm_layer(1024),
        )
        downsample4 = nn.Sequential(
            conv1x1(1024, 2048),
            norm_layer(2048),
        )
        # 3, 4, 6, 3
        self.layer1 = nn.Sequential(Bottleneck(inplanes=300, planes=256, width=64, downsample=downsample1),
                                    Bottleneck(inplanes=256, planes=256, width=64),
                                    Bottleneck(inplanes=256, planes=256, width=64)

                                    )
        self.layer2 = nn.Sequential(
            Bottleneck(inplanes=256, planes=512, width=128, downsample=downsample2, stride=(1, 2)),
            Bottleneck(inplanes=512, planes=512, width=128),
            Bottleneck(inplanes=512, planes=512, width=128),
            Bottleneck(inplanes=512, planes=512, width=128)
            )
        self.layer3 = nn.Sequential(
            Bottleneck(inplanes=512, planes=1024, width=256, downsample=downsample3, stride=(1, 2)),
            Bottleneck(inplanes=1024, planes=1024, width=256),
            Bottleneck(inplanes=1024, planes=1024, width=256),
            Bottleneck(inplanes=1024, planes=1024, width=256),
            Bottleneck(inplanes=1024, planes=1024, width=256),
            Bottleneck(inplanes=1024, planes=1024, width=256)
            )
        self.layer4 = nn.Sequential(Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample4),
                                    Bottleneck(inplanes=2048, planes=2048, width=512),
                                    Bottleneck(inplanes=2048, planes=2048, width=512)

                                    )


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)


    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)

        return x

if __name__=='__main__':
    from torch.autograd import Variable
    from train_config import parse_args
    args=parse_args()
    # args.embedding_type = 'glove'
    args.embedding_type = 'BERT'
    model=ResNet_text_50(args)
    # print(model)
    # input_1 = Variable(torch.ones(2, 100).long())
    input_1 = Variable(torch.Tensor(2, 768,1,120))#BERT
    # input_1 = Variable(torch.Tensor(2, 300, 1, 70))  # glove
    output=model(input_1)
    # print(output)
    print(output.shape)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # summary(model, (100,1,1))
    # print(summary(model,(100)))
    # print()


