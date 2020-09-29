import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from models.CNN_text import ResNet_text_50
from torch.nn import init
from function import load_embedding
from models.Attention import *
import transformers as ppb

#权重初始化
######################################################################
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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Network(nn.Module):
    def __init__(self,args):
        super(Network,self).__init__()

        model_img=models.resnet50(args.pretrained)
        model_txt = ResNet_text_50(args)
        self.channel=2048

        if args.embedding_type=='BERT':
            # 导入预训练好的 DistilBERT 模型与分词器
            # model_class, tokenizer_class, pretrained_weights = (
            #     ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
            ## Want BERT instead of distilBERT? Uncomment the following line:
            model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
            # tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.text_embed = model_class.from_pretrained(pretrained_weights)
            self.text_embed.eval()
            # self.text_embed=nn.Sequential()
            self.BERT = True
            for p in self.text_embed.parameters():
                p.requires_grad = False
        else:
            text_embed = nn.Embedding(args.vocab_size, args.embedding_size)
            if args.embed_pretrained:
                word_embedding = load_embedding(args.embedding_dir)
                text_embed.weight.data.copy_(word_embedding)
            self.text_embed = text_embed
            if args.embed_pretrained:
                for p in self.text_embed.parameters():
                    p.requires_grad = False
            self.BERT = False


        model_img.avgpool=nn.Sequential()
        model_img.fc = nn.Sequential()
        model_txt.avgpool=nn.Sequential()
        # "Avg", "Max", "AvgMax", "Max_avg_att", "Avg_max_att", "AvgMax_share_cat", "AvgMax_share_add"
        if args.pool=='Avg':
            block_img=Avg()
            block_txt=Avg()
        elif args.pool=='Max':
            block_img=Max()
            block_txt=Max()
        elif args.pool=='AvgMax':
            block_img = AvgMax()
            block_txt = AvgMax()


        elif args.pool=="Max_avg_att":
            block_img = Max_avg_att()
            block_txt = Max_avg_att()
        elif args.pool == "Avg_max_att":
            block_img = Avg_max_att()
            block_txt = Avg_max_att()


        elif args.pool == "AvgMax_share_cat":
            block_img = AvgMax_share_cat()
            block_txt = AvgMax_share_cat()
        elif args.pool == "AvgMax_share_add":
            block_img = AvgMax_share_add()
            block_txt = AvgMax_share_add()


        elif args.pool=="Max_attention":
            block_img=Max_attention()
            block_txt=Max_attention()
        elif args.pool=="Avg_attention":
            block_img = Avg_attention()
            block_txt = Avg_attention()
        elif args.pool == "AvgMax_attention":
            block_img =AvgMax_attention()
            block_txt =AvgMax_attention()


        elif args.pool =="spatial_Max_attention":
            block_img = spatial_Max_attention()
            block_txt = spatial_Max_attention()
        elif args.pool =="spatial_Avg_attention":
            block_img = spatial_Avg_attention()
            block_txt = spatial_Avg_attention()
        elif args.pool =="spatial_AvgMax_attention":
            block_img = spatial_AvgMax_attention()
            block_txt = spatial_AvgMax_attention()

        elif args.pool =="spatial_Max":
            block_img = spatial_Max()
            block_txt = spatial_Max()
        elif args.pool =="spatial_Avg":
            block_img = spatial_Avg()
            block_txt = spatial_Avg()
        elif args.pool =="spatial_AvgMax":
            block_img = spatial_AvgMax()
            block_txt = spatial_AvgMax()

        self.model_img=model_img
        self.model_txt=model_txt

        self.block_img = block_img
        self.block_txt = block_txt


    def forward(self, img, txt,mask):
        if self.BERT:
            with torch.no_grad():
                txt = self.text_embed(txt, attention_mask=mask)
                txt = txt[0]
        else:
            txt = self.text_embed(txt)

        # print(txt.shape)
        txt = txt.unsqueeze(1)
        txt = txt.permute(0, 3, 1, 2)
        # print(txt.shape)
        ##img
        img_feature = self.model_img.conv1(img)
        img_feature = self.model_img.bn1(img_feature)
        img_feature = self.model_img.relu(img_feature)
        img_feature = self.model_img.maxpool(img_feature)

        img_feature = self.model_img.layer1(img_feature)
        img_feature = self.model_img.layer2(img_feature)
        img_feature = self.model_img.layer3(img_feature)
        img_feature = self.model_img.layer4(img_feature)
        # print(img_feature.shape)
        img_feature = self.block_img(img_feature)

        ##txt
        txt_feature = self.model_txt(txt)
        # print(txt_feature.shape)
        txt_feature = self.block_txt(txt_feature)
        return img_feature,txt_feature

if __name__ == '__main__':
    from train_config import parse_args

    args = parse_args()
    args.embedding_type='glove_768'
    args.embed_pretrained=False
    args.embedding_size=768
    args.embedding_dir = r"D:\python\baseline_person\data\glove_embedding\glove_200d_embedding.npy"
    # "Avg", "Max", "AvgMax", "Max_avg_att", "Avg_max_att", "AvgMax_share_cat", "AvgMax_share_add"
    args.pool='AvgMax_share_add'
    args.feature_size=2048

    model=Network(args)
    model=model.cuda()
    # print(model.text_embed.parameters())
    a=model.text_embed.parameters()
    img=Variable(torch.FloatTensor(2, 3, 384, 128)).cuda()
    txt = Variable(torch.ones(2, 120).long()).cuda()
    mask = Variable(torch.ones(2, 120).long()).cuda()
    img_feature, txt_feature=model(img,txt,mask)

    print(img_feature.shape)
    print(txt_feature.shape)

