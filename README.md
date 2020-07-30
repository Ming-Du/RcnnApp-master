# Implementation of R-CNN

This is a Implementation of R-CNN. (R-CNN: Regions with Convolutional Neural Network Features)

## Prerequisites
- Python 3.6
- Pytorch 1.5.1
- sklearn 
- tensorboard
- tensorflow
- PIL

You can run the code in Windows/Linux with CPU/GPU. 

## CONNECT ME

244746808@qq.com

## Structure

The project is structured as follows:

17flowers    
find_tune_RCNN.py      
README.md           
train_alexnet.py
2flowers     
image_1306.jpg   test image       
resnet.py           
transform.py
alexnet.py   
make_dataset.py        
selectivesearch.py
config.py    
preprocessing_RCNN.py  
svm_RCNN.py
dataset.csv  
pre_train_model        
tools.py

```

## Getting started

### Supervised Train

In this step, we use make our own dataset for alexnet to train, i entrance the image and expand the original dataset. 

```
$ python make_dataset --datapath='./17flowers'
```

You can directly run it with default parameters.

### train alexnet or train resnet

In this step, we finetune the model on the 2flowers dataset.

```
$ python train_alexnet.py
```

Here we use the alexnet to train dataset, the last accuracy over 95%.

### fine tune

In this step, we fine tuen the alexnet ,change the class number to 3, background is one label.

```
$ python fine_rune_rcnn.py
```

Here we adopt the finetuned AlexNet with 20 epoch in last step as feature extracter.

### SVMS the last step we use svm the seprate the flower
```
$ python svm_RCNN.py

```


### Attention
 i don't come true the nms and the last step regression, i don't have time to do this

## References

- Selective-search: https://github.com/AlpacaDB/selectivesearch
- R-CNN with Tensorflow: https://github.com/bigbrother33/Deep-Learning

Many codes related to image proposals are copied directly from "R-CNN with Tensorflow", and this project has shown more details about R-CNN in chinese.
# RcnnApp-master
