from torch.nn import init
import torch.nn as nn
import torch


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def fc_linear(in_planes, out_planes):
    block = nn.Sequential(nn.Linear(in_planes, out_planes),
                          nn.BatchNorm1d(out_planes),
                          nn.ReLU(inplace=True),
                          )
    block.apply(weights_init_kaiming)
    return block

class attention_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(attention_block, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #view 相当于resize函数，改变形状
        y = self.att(x)
        output=x*y
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output=x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        output=self.sigmoid(x)*output
        return output

class Max_avg_att(nn.Module):
    def __init__(self):
        super(Max_avg_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.att = nn.Sequential(
            nn.Conv2d(2048, 2048 // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048 // 16, 2048, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc=fc_linear(2048,2048)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out=self.avg_pool(x)
        # print(avg_out.shape)
        avg_out = self.att(avg_out)
        max_out = self.max_pool(x)
        output = avg_out * max_out
        output = torch.flatten(output, 1)
        output=self.fc(output)
        return output

class Avg_max_att(nn.Module):
    def __init__(self):
        super(Avg_max_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.att = nn.Sequential(
            nn.Conv2d(2048, 2048 // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048 // 16, 2048, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc = fc_linear(2048, 2048)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.att(self.max_pool(x))
        avg_out = self.avg_pool(x)
        output = avg_out * max_out
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output

class AvgMax(nn.Module):
    def __init__(self):
        super(AvgMax, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = fc_linear(4096, 2048)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        output = torch.cat((avg_out,max_out),1)
        output = torch.flatten(output, 1)
        output=self.fc(output)
        return output


class AvgMax_share_cat(nn.Module):
    def __init__(self):
        super(AvgMax_share_cat, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc=fc_linear(2048,1024)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = torch.flatten(avg_out, 1)
        max_out = torch.flatten(max_out, 1)
        avg_out=self.fc(avg_out)
        max_out=self.fc(max_out)
        output = torch.cat((avg_out,max_out),1)
        return output


class AvgMax_share_add(nn.Module):
    def __init__(self):
        super(AvgMax_share_add, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc=fc_linear(2048,2048)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = torch.flatten(avg_out, 1)
        max_out = torch.flatten(max_out, 1)
        avg_out=self.fc(avg_out)
        max_out=self.fc(max_out)
        output = avg_out+max_out
        return output

class Max(nn.Module):
    def __init__(self):
        super(Max,self).__init__()
        self.pool=nn.AdaptiveMaxPool2d((1, 1))
        self.fc = fc_linear(2048, 2048)
    def forward(self, x):
        output=self.pool(x)
        # print(output.shape)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.fc(output)
        return output

class Avg(nn.Module):
    def __init__(self):
        super(Avg,self).__init__()
        self.pool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_linear(2048, 2048)
    def forward(self, x):
        output=self.pool(x)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output

class Max_attention(nn.Module):
    def __init__(self, reduction=16):
        super(Max_attention, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.attention=attention_block(2048)
        self.fc = fc_linear(2048, 2048)

    def forward(self, x):
        #view 相当于resize函数，改变形状
        output = self.pool(x)
        # print(output.shape)
        output = torch.flatten(output, 1)
        output = self.attention(output)
        output=self.fc(output)
        return output

class Avg_attention(nn.Module):
    def __init__(self, reduction=16):
        super(Avg_attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention=attention_block(2048)
        self.fc = fc_linear(2048, 2048)

    def forward(self, x):
        #view 相当于resize函数，改变形状
        output = self.pool(x)
        # print(output.shape)
        output = torch.flatten(output, 1)
        output = self.attention(output)
        output=self.fc(output)
        return output

class AvgMax_attention(nn.Module):
    def __init__(self):
        super(AvgMax_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.attention = attention_block(4096)
        self.fc = fc_linear(4096, 2048)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        output = torch.cat((avg_out,max_out),1)
        output = torch.flatten(output, 1)
        output = self.attention(output)
        output=self.fc(output)
        return output

class spatial_Max_attention(nn.Module):
    def __init__(self, reduction=16):
        super(spatial_Max_attention, self).__init__()
        self.sa=SpatialAttention()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.attention=attention_block(2048)
        self.fc = fc_linear(2048, 2048)

    def forward(self, x):
        output = self.sa(x)
        #view 相当于resize函数，改变形状
        output = self.pool(output)
        # print(output.shape)
        output = torch.flatten(output, 1)
        output = self.attention(output)
        output=self.fc(output)
        return output

class spatial_Avg_attention(nn.Module):
    def __init__(self, reduction=16):
        super(spatial_Avg_attention, self).__init__()
        self.sa = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.attention=attention_block(2048)
        self.fc = fc_linear(2048, 2048)

    def forward(self, x):
        output = self.sa(x)
        #view 相当于resize函数，改变形状
        output = self.pool(output)
        # print(output.shape)
        output = torch.flatten(output, 1)
        output = self.attention(output)
        output=self.fc(output)
        return output

class spatial_AvgMax_attention(nn.Module):
    def __init__(self):
        super(spatial_AvgMax_attention, self).__init__()
        self.sa = SpatialAttention()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.attention = attention_block(4096)
        self.fc = fc_linear(4096, 2048)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.sa(x)

        avg_out = self.avg_pool(output)
        max_out = self.max_pool(output)
        output = torch.cat((avg_out,max_out),1)
        output = torch.flatten(output, 1)
        # print(output.shape)
        output = self.attention(output)
        output=self.fc(output)
        return output


class spatial_Max(nn.Module):
    def __init__(self, reduction=16):
        super(spatial_Max, self).__init__()
        self.sa=SpatialAttention()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = fc_linear(2048, 2048)

    def forward(self, x):
        output = self.sa(x)
        #view 相当于resize函数，改变形状
        output = self.pool(output)
        # print(output.shape)
        output = torch.flatten(output, 1)
        output=self.fc(output)
        return output

class spatial_Avg(nn.Module):
    def __init__(self, reduction=16):
        super(spatial_Avg, self).__init__()
        self.sa = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = fc_linear(2048, 2048)

    def forward(self, x):
        output = self.sa(x)
        #view 相当于resize函数，改变形状
        output = self.pool(output)
        # print(output.shape)
        output = torch.flatten(output, 1)
        output=self.fc(output)
        return output

class spatial_AvgMax(nn.Module):
    def __init__(self):
        super(spatial_AvgMax, self).__init__()
        self.sa = SpatialAttention()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = fc_linear(4096, 2048)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.sa(x)
        avg_out = self.avg_pool(output)
        max_out = self.max_pool(output)
        output = torch.cat((avg_out,max_out),1)
        output = torch.flatten(output, 1)
        output=self.fc(output)
        return output