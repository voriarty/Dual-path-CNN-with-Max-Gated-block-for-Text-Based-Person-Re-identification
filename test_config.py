import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='command for evaluate on CUHK-PEDES')

    # Directory
    #CUHK-PEDES
    parser.add_argument('--dir', type=str,
                        default=r'D:\python\baseline_person\data',
                        help='directory to store dataset')  # 数据路径
    parser.add_argument('--dataset', type=str,
                        default="CUHKPEDES",
                        help='CUHKPEDES or Flickr30k or flowers')  # 数据路径

    # choose of model

    parser.add_argument('--pool', type=str, default='Max_attention',
                        help='one of "Avg", "Max","AvgMax",'
                             '"Max_avg_att","Avg_max_att",'
                             '"AvgMax_share_cat","AvgMax_share_add",'
                             '"Max_attention","Avg_attention","AvgMax_attention",'
                             '"spatial_Max_attention","spatial_Avg_attention","spatial_AvgMax_attention",'
                             '"spatial_Max","spatial_Avg","spatial_AvgMax"')


    # CNN setting
    parser.add_argument('--num_classes', type=int, default=11003)
    # parser.add_argument('--num_classes', type=int, default=29783)
    parser.add_argument('--pretrained', action='store_true',
                        help='whether or not to restore the pretrained visual model')
    parser.add_argument('--droprate', default=0, type=float, help='drop rate')

    # BERT setting
    parser.add_argument('--max_length', type=int, default=120)
    parser.add_argument('--embedding_type', type=str,
                        default='BERT',
                        help='type of word embedding BERT, glove_768,glove_300,glove_200,glove_100,glove_50')  # 词向量类型

    # glove setting
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=7012)
    parser.add_argument('--embed_pretrained', action='store_true',
                        help='whether or not to restore the pretrained word_embedding')
    parser.add_argument('--embedding_dir', type=str,
                        default=r'D:\python\baseline_person\data\glove_embedding\glove_300d_embedding.npy',
                        help='directory to store embedding file')  # 存储embedding文件路径

    parser.add_argument('--model_path', type=str,
                        default=r"D:\python\baseline_person\checkpoint\Experiment90",
                        help='directory to load checkpoint')
    parser.add_argument('--checkpoint_dir', type=str,
                        default="checkpoint/Experiment90",
                        help='directory to store checkpoint')  # 保存节点路径
    parser.add_argument('--log_test_dir', type=str,
                        default="log_test/Experiment90",
                        help='directory to store test')  # 保存测试


    parser.add_argument('--feature_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoches', type=int, default=160)
    parser.add_argument('--gpus', type=str, default='0')

    args = parser.parse_args()
    return args
