import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='command for train on CUHK-PEDES')
    parser.add_argument('--name', default='Experiment90', type=str, help='output model name')

    #dataset_Directory
    #CUHK-PEDES
    parser.add_argument('--dir', type=str,
                        default=r'D:\python\baseline_person\data',
                        help='directory to store dataset')  # 数据路径
    parser.add_argument('--dataset', type=str,
                        default="CUHKPEDES",
                        help='CUHKPEDES or Flickr30k or flowers')  # 数据路径

    ##save_Directory
    parser.add_argument('--checkpoint_dir', type=str,
                        default="checkpoint",
                        help='directory to store checkpoint')  # 保存节点路径
    parser.add_argument('--log_dir', type=str,
                        default="runs",
                        help='directory to store log')  # 保存日志路径

    #choose of model

    parser.add_argument('--pool', type=str, default='Max_attention',
                        help='one of "Avg", "Max","AvgMax",'
                             '"Max_avg_att","Avg_max_att",'
                             '"AvgMax_share_cat","AvgMax_share_add",'
                             '"Max_attention","Avg_attention","AvgMax_attention",'
                             '"spatial_Max_attention","spatial_Avg_attention","spatial_AvgMax_attention",'
                             '"spatial_Max","spatial_Avg","spatial_AvgMax"')

    #word_embedding
    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--embedding_type', type=str,
                        default='BERT',
                        help='type of word embedding BERT, glove_768,glove_300,glove_200,glove_100,glove_50')  # 词向量类型

    #glove setting
    parser.add_argument('--embedding_size', type=int, default=120)
    parser.add_argument('--vocab_size', type=int, default=7012)
    parser.add_argument('--embed_pretrained', action='store_false',
                        help='whether or not to restore the pretrained word_embedding')
    parser.add_argument('--embedding_dir', type=str,
                        default=r'D:\python\baseline_person\data\glove_embedding\glove_50d_embedding.npy',
                        help='directory to store embedding file')  # 存储embedding文件路径


    #CNN setting
    parser.add_argument('--num_classes', type=int, default=11003)
    # parser.add_argument('--num_classes', type=int, default=29783)
    parser.add_argument('--feature_size', type=int, default=2048)
    parser.add_argument('--pretrained', action='store_false',
                       help='whether or not to restore the pretrained visual model')
    parser.add_argument('--droprate', default=0, type=float, help='drop rate')

    #experiment setting
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoches', type=int, default=160)
    # parser.add_argument('--resume', default=r'D:\python\baseline_person\checkpoint\Experiment77\143.pth.tar',
    #                     help='whether or not to restore the pretrained whole model')
    parser.add_argument('--resume', action='store_true',
                        help='whether or not to restore the pretrained whole model')

    #loss function setting
    parser.add_argument('--CMPM', action='store_false')
    parser.add_argument('--CMPC', action='store_false')

    #Optimization setting
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='one of "sgd", "adam", "rmsprop", "adadelta", or "adagrad"')
    parser.add_argument('--wd', type=float, default=0.00004)

    #adam_setting
    parser.add_argument('--adam_lr', type=float, default=0.0002, help='the learning rate of adam')
    parser.add_argument('--adam_alpha', type=float, default=0.9)
    parser.add_argument('--adam_beta', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-8)

    #sgd_setting
    parser.add_argument('--sgd_lr', type=float, default=0.1, help='the learning rate')
    parser.add_argument('--sgd_weight_decay', type=float, default=5e-4)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)

    parser.add_argument('--lr_decay_type', type=str, default='MultiStepLR',
                        help='One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    # parser.add_argument('--end_lr', type=float, default=1e-6,
    #                     help='minimum end learning rate used by a polynomial decay learning rate')
    parser.add_argument('--epoches_decay', type=str, default='55_80_100_120_140', help='#epoches when learning rate decays')
    parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--img_epoch', type=int, default=80, help='img start to train"')

    # Default setting
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    return args
