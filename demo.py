import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import yaml
import time
# from tensorboard_logger import configure, log_value
from train_config import parse_args
from function import data_config, optimizer_function, load_checkpoint, lr_scheduler, AverageMeter, save_checkpoint, \
    gradual_warmup, fix_seed
from models.model import Network
from CMPM_CMPC import Loss
from tensorboardX import SummaryWriter
from torch import nn
import torch.distributed as dist



def valid(epoch, train_loader, network, compute_loss, args):
    valid_loss = AverageMeter()
    valid_cmpc_loss = AverageMeter()
    valid_cmpm_loss = AverageMeter()

    # switch to valid mode
    network.eval()

    with torch.no_grad():
        for step, (images, captions, labels, mask) in enumerate(train_loader):
            images = images.cuda()
            captions = captions.cuda()
            labels = labels.cuda()
            mask = mask.cuda()
            # if images.shape[0] < args.batch_size:  # skip the last batch
            #     continue

            # compute loss
            image_embeddings, text_embeddings = network(images, captions, mask)
            cmpm_loss, cmpc_loss, loss, image_precision, text_precision = compute_loss(
                image_embeddings, text_embeddings, labels)
            valid_cmpm_loss.update(cmpm_loss, images.shape[0])
            valid_cmpc_loss.update(cmpc_loss, images.shape[0])
            valid_loss.update(loss, images.shape[0])

            if step % 30 == 0:
                print(
                    "Valid Epoch:[{}/{}] iteration:[{}/{}] cmpm_loss:{:.4f} cmpc_loss:{:.4f} "
                        .format(epoch + 1, args.num_epoches, step, len(train_loader), valid_cmpm_loss.avg,
                                valid_cmpc_loss.avg,
                                ))

    return valid_cmpm_loss.avg, valid_cmpc_loss.avg, valid_loss.avg


def train(epoch, train_loader, network, opitimizer, compute_loss, args, checkpoint_dir):
    train_loss = AverageMeter()
    image_pre = AverageMeter()
    text_pre = AverageMeter()
    train_cmpc_loss = AverageMeter()
    train_cmpm_loss = AverageMeter()
    # switch to train mode
    network.train()

    for step, (images, captions, labels, mask) in enumerate(train_loader):
        images = images.cuda()
        captions = captions.cuda()
        labels = labels.cuda()
        mask = mask.cuda()
        opitimizer.zero_grad()
        # if images.shape[0] < args.batch_size:  # skip the last batch
        #     continue

        # compute loss
        image_embeddings, text_embeddings = network(images, captions, mask)
        cmpm_loss, cmpc_loss, loss, image_precision, text_precision = compute_loss(
            image_embeddings, text_embeddings, labels)
        train_cmpm_loss.update(cmpm_loss, images.shape[0])
        train_cmpc_loss.update(cmpc_loss, images.shape[0])

        train_loss.update(loss, images.shape[0])

        # 计算准确度
        image_pre.update(image_precision, images.shape[0])
        text_pre.update(text_precision, captions.shape[0])

        # 梯度下降
        loss.backward()
        opitimizer.step()
        if step % 300 == 0:
            print(
                "Train Epoch:[{}/{}] iteration:[{}/{}] cmpm_loss:{:.4f} cmpc_loss:{:.4f} "
                "image_pre:{:.4f} text_pre:{:.4f}"
                    .format(epoch + 1, args.num_epoches, step, len(train_loader), train_cmpm_loss.avg,
                            train_cmpc_loss.avg,
                            image_pre.avg, text_pre.avg))

    # save_checkpoint(state, epoch, dst, is_best)
    state = {"epoch": epoch + 1,
             "state_dict": network.state_dict(),
             "W": compute_loss.W
             }

    save_checkpoint(state, epoch + 1, checkpoint_dir)
    return train_cmpm_loss.avg, train_cmpc_loss.avg, train_loss.avg, image_pre.avg, text_pre.avg


def main(network, dataloader, compute_loss, optimizer, scheduler, start_epoch, args, checkpoint_dir):
    start = time.time()
    for epoch in range(start_epoch, args.num_epoches):
        print("**********************************************************")

        if epoch < args.warm_epoch:
            print('learning rate warm_up')
            if args.optimizer == 'sgd':
                optimizer = gradual_warmup(epoch, args.sgd_lr, optimizer, epochs=args.warm_epoch)
            else:
                optimizer = gradual_warmup(epoch, args.adam_lr, optimizer, epochs=args.warm_epoch)
        if args.embedding_type == "BERT" or args.embed_pretrained:
            optimizer.param_groups[2]['lr'] = 0
        if epoch < args.img_epoch:
            print("this is first step:")
            optimizer.param_groups[0]['lr'] = 0
        else:
            print("this is second step:")
        lr1 = optimizer.param_groups[0]['lr']
        lr2 = optimizer.param_groups[1]['lr']
        lr3 = optimizer.param_groups[2]['lr']
        print('learning_rate_1: {:.9f}'.format(lr1))
        print('learning_rate_2: {:.9f}'.format(lr2))
        print('learning_rate_3: {:.9f}'.format(lr3))

        train_cmpm_loss, train_cmpc_loss, train_loss, train_image_pre, train_text_pre = \
            train(epoch, dataloader['train'], network, optimizer, compute_loss, args, checkpoint_dir)
        print("----------------------------------------------------------------")

        valid_cmpm_loss, valid_cmpc_loss, valid_loss = \
            valid(epoch, dataloader['val'], network, compute_loss, args)

        scheduler.step()  # 训练的时候进行学习率规划，其定义在下面给出

        # 动态纪录数据
        writer.add_scalars('learning_rate', {'learning_rate_1': lr1, "learning_rate_2": lr2, "learning_rate_3": lr3},
                           epoch)
        writer.add_scalars('train_pre', {'image_pre': train_image_pre, "text_pre": train_text_pre}, epoch)
        writer.add_scalars('train_loss', {'cmpm_loss': train_cmpm_loss,
                                          "cmpc_loss": train_cmpc_loss, "train_loss": train_loss}, epoch)

        writer.add_scalars('valid_loss', {'cmpm_loss': valid_cmpm_loss,
                                          "cmpc_loss": valid_cmpc_loss, "train_loss": valid_loss}, epoch)
        writer.close()

        if args.CMPM:
            print("Train_cmpm_loss:{:.4f} Valid_cmpm_loss:{:.4f}".format(train_cmpm_loss, valid_cmpm_loss))

        if args.CMPC:
            print("Train_cmpc_loss:{:.4f} Valid_cmpc_loss:{:.4f}".format(train_cmpc_loss, valid_cmpc_loss))

        Epoch_time = time.time() - start
        start = time.time()
        print('Epoch_training complete in {:.0f}m {:.0f}s'.format(
            Epoch_time // 60, Epoch_time % 60))


if __name__ == '__main__':

    # 固定随机数
    seed = 53113
    fix_seed(seed)

    args = parse_args()
    name = args.name  # 模型名字

    # 数据保存
    checkpoint_dir = args.checkpoint_dir  # 模型保存路径
    checkpoint_dir = os.path.join(checkpoint_dir, name)
    log_dir = args.log_dir
    log_dir = os.path.join(log_dir, name)  # 动态保存数据路径
    run_dir = os.path.join('runs', args.name)

    # configure(log_dir)
    writer = SummaryWriter(run_dir)
    print(checkpoint_dir)
    # print(log_dir)

    opt_dir = os.path.join('log', name)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    print(opt_dir)
    # save args
    with open('%s/opts.yaml' % opt_dir, 'w') as fp:  # 先打开一个文件描述符
        yaml.dump(vars(args), fp, default_flow_style=False)

    # 加载GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []  # 定义列表
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            # 将gpu编号添加到列表末尾。
            gpu_ids.append(gid)

    cudnn.benchmark = True  # 可以提升一点训练速度,没什么额外开销,一般都会加。
    # set gpu ids
    # if len(gpu_ids) == 1:
    #     torch.cuda.set_device(gpu_ids[0])

    # 数据预处理
    transform_train_list = [
        # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        # torchvision.transforms.Resize(size, interpolation=2)：size : 获取输出图像的大小
        #                                                     interpolation : 插值，默认的  PIL.Image.BILINEAR， 一共有4中的插值方法
        # transforms.Resize((256, 128), interpolation=3),
        # transforms.Resize((224, 224)),
        transforms.Resize((384, 128), interpolation=3),
        transforms.Pad(10),  # 将给定的PIL.Image的所有边用给定的pad value填充。
        # transforms.RandomCrop((224, 224)),  # 切割中心点的位置随机选取
        transforms.RandomCrop((384, 128)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转给定的PIL.Image，概率为0.5，一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        # 把一个取值范围为[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray,转换为形状为
        # [C,H,W]，取值范围是[0,1.0]的torch.FloadTensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
        #                        hue=0),  # 色彩抖动
        # 给定均值mean和方差std,将会把Tensor正则化。即Normalized_image=(image-mean)/std。
    ]
    # if args.erasing_p > 0:
    #     transform_train_list = transform_train_list + [RandomErasing(probability=args.erasing_p)]
    transform_val_list = [
        # transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        # transforms.Resize((224, 224)),
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # 定义字典 data_transforms
    data_transforms = {
        # transforms.Compose：将多个transform组合起来使用。
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    dataloaders = {
    x: data_config(args.dir, args.batch_size, x, args.max_length, args.embedding_type, transform=data_transforms[x]) for
    x in
    ['train', 'val']}

    # 查看一个batch
    # images, captions, labels, captions_length = next(iter(dataloaders['val']))
    # print(labels)  # [batch_size,max_length]

    # 损失函数
    if args.CMPM:
        print("import CMPM")
    if args.CMPC:
        print("import CMPC")

    compute_loss = Loss(args).cuda()
    model = Network(args).cuda()

    rank = 1 * 4
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=4,
        rank=rank
    )

    if len(gpu_ids) > 0:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2,3,4,5])
    print("use ", args.pool)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # 计算参数数量
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # 断点加载
    if args.resume is not None:
        start_epoch, model = load_checkpoint(model, args.resume)
    else:
        print("Do not load checkpoint,Epoch start from 0")
        start_epoch = 0

    # 优化器
    opitimizer = optimizer_function(args, model, compute_loss.parameters())
    exp_lr_scheduler = lr_scheduler(opitimizer, args)
    main(model, dataloaders, compute_loss, opitimizer, exp_lr_scheduler, start_epoch, args, checkpoint_dir)









