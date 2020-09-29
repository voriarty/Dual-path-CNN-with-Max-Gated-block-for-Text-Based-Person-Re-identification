from torchtext import data,datasets
import pickle
import torch
import numpy as np
TEXT=data.Field(sequential=True,tokenize='spacy',lower=True,fix_length=120)
LABLE=data.LabelField(sequential=False,)
Fields=[('target',LABLE),('image_path',LABLE),('text', TEXT)]
train,val,test=data.TabularDataset.splits(path=r'D:\python\baseline_person\data\CUHK-PEDES',train='train.csv',validation='val.csv',test='test.csv',
                                          format='csv',fields=Fields)
# print(vars(train.examples[0]))
# print(train[5].__dict__.keys())
# TEXT.build_vocab(train, vectors='glove.6B.300d')
# TEXT.build_vocab(train, vectors='glove.6B.200d')
# TEXT.build_vocab(train, vectors='glove.6B.100d')
TEXT.build_vocab(train, vectors='glove.6B.50d')

print(TEXT.vocab.vectors.shape)#torch.Size([7012, 300])
# print(len(train)) #68108
# print(TEXT.vocab.vectors)
np.save('../data/glove_embedding/glove_50d_embedding.npy',TEXT.vocab.vectors)
data_type=train
data_type=val
data_type=test
# save_path=r'D:\python\baseline_person\data\glove_encode\train_data_50d.pkl'
# save_path=r'D:\python\baseline_person\data\glove_encode\val_data_50d.pkl'
save_path=r'D:\python\baseline_person\data\glove_encode\test_data_50d.pkl'

max=120
with open(save_path,'wb') as f:
    labels=[]
    caption_id=[]
    images_path=[]
    count=0
    for i in range(len(data_type)):
        count+=1
        string_to_id=[TEXT.vocab.stoi[j] for j in vars(data_type.examples[i])['text']]
        # print(string_to_id)
        if len(string_to_id)<max:
            pad=[1]*(max-len(string_to_id))
            string_to_id.extend(pad)
        else:
            string_to_id=string_to_id[:max]
        if len(string_to_id)!=max:
            print('error')
            break
        labels.append(vars(data_type.examples[i])['target'])
        caption_id.append(string_to_id)
        path='CUHK-PEDES\imgs\\'+vars(data_type.examples[i])['image_path']
        # print(path)
        images_path.append(path)
    print(count)
    # print(labels)
    # print(caption_id)
    dict={'labels':labels,'caption_id':caption_id,'images_path':images_path}
    pickle.dump(dict,f)