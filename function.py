import torch.utils.data as data
import os
from  glove_token_process import CuhkPedes
import torch
import numpy as np
import random
from BERT_token_process import CUHKPEDES_BERT_token
from dataset_process.flickr30k_BERT_process import Flickr30k_BERT_token
from dataset_process.flowers_BERT_process import Flowers_BERT_token

def data__more(dir, dataset_name,batch_size, split, max_length,transform):
    print("The word length is", max_length)
    if dataset_name=='Flickr30k':
        print("The dataset is Flickr30k")
        data_split = Flickr30k_BERT_token(dir, split, max_length, transform)
    else:
        print("The dataset is Flowers")
        data_split = Flowers_BERT_token(dir, split, max_length,transform)
    print("the number of",split,":",len(data_split))
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=0)
    return loader


def data_config(dir, batch_size, split, max_length, embedding_type,transform):
    print("The word length is", max_length)
    if embedding_type=='BERT':
        print("The word embedding type is BERT")
        data_split = CUHKPEDES_BERT_token(dir, split, max_length, transform)
    else:
        print("The word embedding type is ",embedding_type)
        data_split = CuhkPedes(dir, split, max_length, embedding_type,transform)
    print("the number of",split,":",len(data_split))
    if split == 'train':
        shuffle = True
    else:
        shuffle = False
    loader = data.DataLoader(data_split, batch_size, shuffle=shuffle, num_workers=0)
    return loader

def optimizer_function(args,model,param=None):
    # 制定任意一层的学习率（多个参数组）：下面为两个参数组采用不同学习率

    ignored_params = (list(map(id, model.model_img.conv1.parameters()))
                      + list(map(id, model.model_img.bn1.parameters()))
                      + list(map(id, model.model_img.layer1.parameters()))
                      + list(map(id, model.model_img.layer2.parameters()))
                      + list(map(id, model.model_img.layer3.parameters()))
                      + list(map(id, model.model_img.layer4.parameters()))
                      + list(map(id, model.text_embed.parameters()))
                      )
    img_params = list(model.model_img.conv1.parameters())
    img_params.extend(list(model.model_img.bn1.parameters()))
    img_params.extend(list(model.model_img.layer1.parameters()))
    img_params.extend(list(model.model_img.layer2.parameters()))
    img_params.extend(list(model.model_img.layer3.parameters()))
    img_params.extend(list(model.model_img.layer4.parameters()))
    embed_parameters = list(model.text_embed.parameters())

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())  # ## 获取非指定层的参数id
    base_params = list(base_params)
    if param is not None:
        base_params.extend(list(param))  # 得到共享线性层的参数

    # parameters = [
    #     {'params': img_params,'initial_lr': 0.1},
    #     {'params': base_params,'initial_lr': 0.1},
    #     {'params': embed_parameters,'initial_lr': 0}
    # ]
    parameters = [
        {'params': img_params},
        {'params': base_params},
        {'params': embed_parameters}
    ]
    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.sgd_lr, weight_decay=args.sgd_weight_decay, momentum=args.sgd_momentum, nesterov=True)
        print("优化器为：SGD")
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=args.adam_lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)

        # seed
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
    return optimizer

def lr_scheduler(optimizer, args):

    if args.lr_decay_type=="ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min', factor=args.lr_decay_ratio,
                                                           patience=5, min_lr=args.end_lr)
        print("lr_scheduler is ReduceLROnPlateau")
    else:
        if '_' in args.epoches_decay:
            epoches_list = args.epoches_decay.split('_')
            epoches_list = [int(e) for e in epoches_list]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma=args.lr_decay_ratio)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, gamma=args.lr_decay_ratio,
            #                                                  last_epoch=143)
            print("lr_scheduler is MultiStepLR")
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay), gamma=args.lr_decay_ratio)
            print("lr_scheduler is StepLR")
    return scheduler

#加载网络参数
def load_checkpoint(model,resume):
    start_epoch=0
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        # checkpoint= torch.load(resume, map_location='cuda:0')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # print(resume)
        print('Load checkpoint at epoch %d.' % (start_epoch))
    return start_epoch,model

class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count

#保存训练节点
def save_checkpoint(state, epoch, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    filename = os.path.join(dst, str(epoch)) + '.pth.tar'
    torch.save(state, filename)

#### gradual warmup 逐渐预热学习率
def gradual_warmup(epoch,init_lr,optimizer,epochs):
    lr=init_lr
    if epoch < epochs:  # gradual warmup 逐渐预热学习率
        warmup_percent_done = (epoch+1) / epochs
        warmup_learning_rate = init_lr * warmup_percent_done  # gradual warmup_lr
        lr = warmup_learning_rate
    # else:
    #     learning_rate=np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
        # lr = lr ** 1.0001  # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)
    # print('learning_rate in epoch {}: {:.4f}s'.format(epoch,lr))
    # optimizer.param_groups[0]['lr']=0.1*lr
    # optimizer.param_groups[1]['lr']=lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# compute_topk(images_bank, text_bank, labels_bank,labels_bank, [1, 10], True)
def compute_topk(query, gallery, target_query, target_gallery, k=[1,10], reverse=False):
    result = []
    #torch.norm(input, p, dim, out=None,keepdim=False) → Tensor
    #dim：求指定维度上的范数，dim=0是对0维度上的一个向量求范数，返回结果数量等于其列的个数
    # p:范数计算中的幂指数值,默认为l2范数dim
    #keepdim（bool）– 保持输出的维度,keepdim=True时，输出与输入维度相同，仅仅是输出在求范数的维度上元素个数变为1
    #keepdim=True时，输出与输入维度相同，仅仅是输出在求范数的维度上元素个数变为1

    #图像特征和文本特征标准化
    query = query / (query.norm(dim=1,keepdim=True)+1e-12)
    gallery = gallery / (gallery.norm(dim=1,keepdim=True)+1e-12)
    sim_cosine = torch.matmul(query, gallery.t())
    #得到一个样本*样本数量的矩阵，表示图像和文本的余弦距离
    result.extend(topk(sim_cosine, target_gallery, target_query, k))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k, dim=0))
    return result


def topk(sim, target_gallery, target_query, k=[1,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    """
    a.topk()求a中的最大值或最小值，返回两个值，一个是a中的值（最大或最小），一个是这个值的索引。
    dim=1，为按行求最大最小值，largest为Ture，求最大值，largest=False，求最小值。
     k=1,max(k)=1
     k=[1,10] max(k)=10
     pred_index shape:[6148,10]
     pred_labels.t():[10,6148]
    """
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    # print("pred_labels shape", pred_labels.shape)
    # print("pred_labels top1",pred_labels)

    if dim == 1:
        pred_labels = pred_labels.t()
        # pred_labels=torch.transpose(pred_labels,1,0)
        # pred_labels = pred_labels.permute(1,0)
    # print("pred_labels.T shape",pred_labels.shape)
    # print("pred_labels.T top1", pred_labels)
    # print("target_gallery shape", target_gallery.shape)
    # print("target_gallery", target_gallery)

    """
    x.view(1, 8)    #输出维度：1*8
    x.view(-1, 4)  # -1表示维数自动判断，此输出的维度为：2*4
    target_query.view(1,-1) #target_query的形状是[6148],变成[1,6148]
    a.expand_as(b)把一个tensor a变成和函数括号内一样形状的tensor，用法与expand（）类似
    pred_labels.t():[10,6148]
    #correct.shape:【10,6148】
    """
    a=target_query.view(1,-1).expand_as(pred_labels)
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))
    # print("correct.shape",correct.shape)

    for topk in k:
        """
        torch.sum(input, dim, out=None) → Tensor
        input (Tensor) – 输入张量
        dim (int) – 缩减的维度 dim=0,按列求和，得到列数形状【6148】
        out (Tensor, optional) – 结果张量
        """
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        # print("correct_k",correct_k)


        result.append(correct_k * 100 / size_total)
    return result

#判断路径是否存在
def check_exists(root):
    if os.path.exists(root):
        return True
    return False

def load_embedding(path):
    word_embedding=torch.from_numpy(np.load(path))
    (vocab_size,embedding_size)=word_embedding.shape
    print('Load word embedding,the shape of word embedding is [{},{}]'.format(vocab_size,embedding_size))
    return word_embedding

def load_part_model(model,path):
    model_dict = model.state_dict()
    checkpoint = torch.load(path)
    pretrained_dict = checkpoint["state_dict"]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 取出预训练模型中与新模型的dict中重合的部分
    model_dict.update(pretrained_dict)  # 用预训练模型参数更新new_model中的部分参数
    model.load_state_dict(model_dict)  # 将更新后的model_dict加载进new model中
    return model

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


"""
注意到有两种图像我们不把他们考虑为正确匹配true-matches

一种是Junk_index1 错误检测的图像，主要是包含一些人的部件。
一种是Junk_index2 相同的人在同一摄像头下，按照reid的定义，我们不需要检索这一类图像。
"""
"""
所以，qf来自于query_feature[i]，query_feature来自于test.py中一批图片传入模型得到的特征数组query_feature.numpy()
同样，gf来自于gallery_feature，来自于test.py中一批图片传入模型得到的特征数组gallery_feature.numpy()
"""
"evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)"


def test_map(query_feature,query_label,gallery_feature, gallery_label):
    # 图像特征和文本特征标准化
    query_feature = query_feature / (query_feature.norm(dim=1, keepdim=True) + 1e-12)
    gallery_feature = gallery_feature / (gallery_feature.norm(dim=1, keepdim=True) + 1e-12)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    # print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],  gallery_feature, gallery_label)

        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    return CMC[0], CMC[9], CMC[19], ap / len(query_label)

def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)  ## 通过view改变tensor的形状，-1是自适应的调整，这里把所有数据放到一列上
    # print(gf.shape)
    # print(query.shape)

    score = torch.mm(gf, query)  # Cosine Distance 余弦距离等价于L2归一化后的内积
    # torch.mm表示两个张量的矩阵相乘，因为query只有一列，所以相乘的结果也只有一列
    score = score.squeeze(1).cpu()  # queeze()功能：去除size为1的维度，包括行和列。当维度大于等于2时，squeeze()无作用。
    # 其中squeeze(0)代表若第一维度值为1则去除第一维度，squeeze(1)代表若第二维度值为1则去除第二维度。
    # a.cpu()是把a放在cpu上
    score = score.numpy()  # 把tensor转换成numpy的格式，为了做后面的排序
    # predict index
    index = np.argsort(score)  ##from small to large
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。
    # 例如：x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    index = index[::-1]  # -1是指步长为-1，也就是从最后一个元素到第一个元素逆序输出
    # index = index[0:2000]
    # good index
    """
    np.argwhere( a ) 
    返回非0的数组元组的索引，其中a是要索引数组的条件。
    """
    # print(gl.shape)
    # print(ql.shape)
    gl=gl.cuda().data.cpu().numpy()
    ql=ql.cuda().data.cpu().numpy()
    query_index = np.argwhere(gl == ql)  ## 返回满足gl==ql的数组元组的索引，即query和gallery图像所属类别相同。


    # 我们可以使用 compute_mAP 来计算最后的结果. 在这个函数中，我们忽略了junk_index带来的影响。
    CMC_tmp = compute_mAP(index, query_index)
    return CMC_tmp


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc


    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc

if __name__ == '__main__':
    import torchvision.transforms as transforms
    # data_config(image_dir, anno_dir, batch_size, split, max_length, embedding_type, transform):
    max_length = 70
    transform_val_list = [
        # transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    split = 'train'
    image_dir = r'D:\python\language_search_master\data'
    anno_dir = r'D:\python\language_search_master\data\processed_data\BERT_encode'
    transform = transforms.Compose(transform_val_list)
    batch_size = 32
    max_length=80
    embedding_type='BERT_token'
    loader=data_config(image_dir, anno_dir, batch_size, split, max_length, embedding_type, transform)
    sample = next(iter(loader))
    img, caption, label, mask = sample
    print(img.shape)
    print(caption.shape)
    print(label.shape)
    print(mask.shape)

