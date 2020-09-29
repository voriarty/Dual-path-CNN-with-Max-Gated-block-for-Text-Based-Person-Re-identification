import torchvision.transforms as transforms
import torch
import yaml
from function import *
from test_config import parse_args
import time
from models.model import Network
import os
import shutil
import torch.backends.cudnn as cudnn
from tensorboard_logger import configure, log_value

def test(data_loader, network, args):

    # switch to evaluate mode
    network.eval()
    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    with torch.no_grad():
        for images, captions, labels,mask in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            mask = mask.cuda()

            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, mask)

            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index:index + interval] = labels

            index = index + interval

        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
        [ac_top1_t2i,ac_top5_t2i, ac_top10_t2i] = compute_topk(text_bank, images_bank, labels_bank, labels_bank, [1,10,20])
        # [ac_top1_i2t,ac_top5_i2t, ac_top10_i2t] = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,5,10])
        # ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i = compute_topk(images_bank, text_bank, labels_bank,
        #                                                                     labels_bank, [1, 10], True)
        return  ac_top1_t2i, ac_top5_t2i,ac_top10_t2i
        # return ac_top1_i2t,ac_top5_i2t, ac_top10_i2t

def main(model,args):
    test_transform = transforms.Compose([
        # transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
        transforms.Resize((384, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_loaders= data_config(args.dir, batch_size=args.batch_size, split='test', max_length=args.max_length, embedding_type=args.embedding_type,transform=test_transform)

    ac_t2i_top1_best = 0.0
    ac_t2i_top5_best = 0.0
    ac_t2i_top10_best = 0.0
    #返回指定路径下的文件和文件夹列表。
    best=0
    dst_best = args.checkpoint_dir + "/model_best" + ".pth.tar"
    # for i in range(1):
    for i in range(100,args.num_epoches):
        i=i+1
        model_file = os.path.join(args.model_path, str(i))+".pth.tar"
        # model_file = os.path.join(args.model_path, 'model_best.pth.tar')
        if os.path.isdir(model_file):
            continue
        start, network = load_checkpoint(model, model_file)
        ac_top1_t2i, ac_top5_t2i,ac_top10_t2i= test(test_loaders, network, args)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best=ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            best = i
            shutil.copyfile(model_file, dst_best)
        # print(model_file)
        # print(dst_best)
        print('Epoch:{} top1_t2i: {:.3f},top5_t2i: {:.3f}, top10_t2i: {:.3f}'
            .format(i,ac_top1_t2i, ac_top5_t2i,ac_top10_t2i))

    print('Epoch:{}:t2i_top1_best: {:.3f}, t2i_top5_best: {:.3f},t2i_top10_best: {:.3f}'.format(
            best,ac_t2i_top1_best,ac_t2i_top5_best, ac_t2i_top10_best))

if __name__ == '__main__':

    args = parse_args()


    # 加载GPU
    str_ids = args.gpus.split(',')
    gpu_ids = []  # 定义列表
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            # 将gpu编号添加到列表末尾。
            gpu_ids.append(gid)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True  # 可以提升一点训练速度,没什么额外开销,一般都会加。
    configure(args.log_test_dir)
    with open('%s/opts.yaml' % args.log_test_dir, 'w') as fp:  # 先打开一个文件描述符
        yaml.dump(vars(args), fp, default_flow_style=False)
    print(args.checkpoint_dir)
    model = Network(args).cuda()
    print("use ", args.pool)
    # model = nn.DataParallel(model)
    main(model,args)
