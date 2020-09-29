import torch.nn as nn
from torchvision import models
import torch
class img_res(nn.Module):
    def __init__(self):
        super(img_res,self).__init__()

        model_img=models.resnet50()
        model_img.avgpool = nn.Sequential()
        model_img.fc = nn.Sequential()
        self.model=model_img

    def forward(self, x):
        output=self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output=self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)
        return output

if __name__=='__main__':
    from torch.autograd import Variable

    input = Variable(torch.Tensor(2, 3,384, 128))  # BERT
    model=img_res()
    output=model(input)
    print(output.shape)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))