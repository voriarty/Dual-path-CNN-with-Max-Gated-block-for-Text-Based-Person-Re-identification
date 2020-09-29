import pandas as pd
import transformers as ppb
import pickle
import numpy as np

# torch.cuda.set_device(0)
# txt_path=r'D:\python\baseline_person\data\CUHK-PEDES\train.csv'
# txt_path=r'D:\python\baseline_person\data\CUHK-PEDES\val.csv'
txt_path=r'D:\python\baseline_person\data\CUHK-PEDES\test.csv'
# save_path=r'D:\python\baseline_person\data\BERT_encode\BERT_id_train_120_new.npz'
# save_path=r'D:\python\baseline_person\data\BERT_encode\BERT_id_val_120_new.npz'
save_path=r'D:\python\baseline_person\data\BERT_encode\BERT_id_test_120_new.npz'
csv_data = pd.read_csv(txt_path, error_bad_lines=False, header=None)
print(csv_data.shape)
# print(csv_data[0].value_counts()) #查看每个label的个数
# print(csv_data[2][:20])
dataset=csv_data[2]  #得到所有的caption
# 导入预训练好的 DistilBERT 模型与分词器
# model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
# Want BERT instead of distilBERT? Uncomment the following line:
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights)
# model = nn.DataParallel(model).cuda()
"在我们把句子交给BERT之前，我们需要做一些简单的处理，把它们转换成所需的格式"
"""
# 分词
# 第一步要做的就是用 BERT 分词器（tokenizer）将这些单词（word）划分成词（token）。
# 第二步，我们再加入句子分类所需的特殊词（在句子开始加入 [CLS]，末端加入 [SEP]）。
# 第三步就是查表，将这些词（token）替换为嵌入表中对应的编号，我们可以从预训练模型中得到这张嵌入表。
# 只需要一行代码就可以完成分词器的所有工作：
# tokenizer.encode("a visually stunning rumination on love", add_special_tokens=True)
# 我们的第一步是对句子进行标记化——将它们按照BERT所熟悉的格式分成单词和子单词。，得到句子列表tokenized
"""
#将所有句子转化为了编号的列表。
tokenized = dataset.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
# print(tokenized.shape)
"""
把这些向量整理成相同的维度（在较短的句子后面填充上编号「0」）
"""
max_len = 120
padded=[]
for i in tokenized.values:
    if len(i)<max_len:
        i+=[0]*(max_len-len(i))
    else:
        i=i[:max_len]
    padded.append(i)
padded=np.array(padded)
print(padded.shape) #shape;[68108,max_len]
attention_mask = np.where(padded != 0, 1, 0)
print(attention_mask)
print(padded)
print(attention_mask.shape)
print(padded.shape)
print(csv_data[1].shape)
print(csv_data[0].shape)
dict={'caption_id':padded,'attention_mask':attention_mask,'images_path':csv_data[1],'labels':csv_data[0]}
with open(save_path,'wb') as f:
    pickle.dump(dict,f)