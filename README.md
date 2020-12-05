# Dual-path-CNN-with-Max-Gated-block-for-Text-Based-Person-Re-identification
Dual-path CNN with Max Gated block for Text-Based Person Re-identification



### Model Structure
You may learn more from `model.py`. 


## Prerequisites

- Python 3.6
- GPU Memory >= 10G
- Numpy
- Pytorch 1.0+
- tensorboardX
- transformers
- torchtext


### Dataset & Preparation
CUHK-PEDES:You can communicate with shuangLi (lishuang@mit.edu) for the dataset.
you can get the image path, labels, and text description in 'train.csv','val.csv', and 'test.csv'.

BERT_token.py:you must tokenize the sentence before inputting the sentence to BERT.

if you want to embed the sentence with Glove, you can run 'glove_token.py'and download glove wordembedding.

### Train
Train a model by
```bash
python train.py
```
you can change hyper-parameter in 'train_config.py'.

`--name` the name of model.

#dataset_Directory

`--dir`  the path of training data

`--dataset` the type of dateset('CUHKPEDES or Flickr30k or flowers')

#save_Directory

`--checkpoint_dir` the path of the saving model.

`--log_dir`  the path of the saving for tensorboardX.

#choose of model
`--pool` the type of pooling (our best model is Max_attention).

#word_embedding
`--max_length` the max_length of the sentence

`--embedding_type` type of word embedding BERT, glove_768,glove_300,glove_200,glove_100,glove_50'

#glove setting

`--embedding_size` the max_length of the sentence for Glove.

`--vocab_size` the vocab_size of the Glove.

`--embed_pretrained` whether or not to restore the pretrained word_embedding

`--embedding_dir` the path to store embedding file.

#CNN setting

`--num_classes` the num_classes of the Classification layer.

`--feature_size` the feature size of the visual and textual description.

`--pretrained` whether or not to restore the pretrained visual model(ImageNet)

`--droprate` the drop rate

#experiment setting

`--batchsize` batch size.

`--num_epoches` num_epoches

`--resume` whether or not to restore the pretrained whole model

#loss function setting

`--CMPM` CPMM loss function setting.

`--CMPC` CMPC loss function setting.

#Optimization setting

`--optimizer` the type of the optimizer ('one of "sgd", "adam", "rmsprop", "adadelta", or "adagrad")

`--wd` 

#adam_setting

`--adam_lr` the learning of adam optimizer

`--adam_alpha`

`--adam_beta`

`--epsilon`

#sgd_setting

`--sgd_lr` the learning of sgd optimizer

`--sgd_weight_decay`

`--sgd_momentum`

`--lr_decay_type` 'One of "MultiStepLR" or "StepLR" or "ReduceLROnPlateau"'

`--lr_decay_ratio`

`--epoches_decay` epoches when learning rate decays

`--warm_epoch` the first K epoch that needs warm up

`--img_epoch` img start to train

`--gpus`

### Test

```bash
python test.py
```
