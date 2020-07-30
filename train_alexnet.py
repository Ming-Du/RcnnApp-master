import os
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import torchvision as tv
from PIL import Image
from alexnet import *

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.autograd import Variable
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
from resnet import *
from torchvision.utils import make_grid

comment = 'train_alexnet'
writer = SummaryWriter(comment=comment)

class Flower(Dataset):
    def __init__(self, root, sep = ',', transform=None, train=True, test=False):
        # self.test = test
        # self.transform = transform
        df = pd.read_csv(root)
        df.columns = ['img_path', 'label']
        imgs = []
        for index, row in df.iterrows():
            imgs.append([row['img_path'], row['label']])
        self.imgs = imgs
        if test:
            i = 0
            arr = []
            while (i < int(len(imgs) / 3.0)):
                x = random.randint(1, len(imgs) - 1)
                arr.append(imgs[x])
                # arr.append(i)
                i += 1
            self.imgs = arr
        # if self.test:
        #     self.imgs = imgs
        # elif train:
        #     self.imgs = imgs[:int(0.7 * len(imgs))]
        # else:
        #     self.imgs = imgs[:int(0.7 * len(imgs))]
        if transform is None:
            normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            if test or not train:
                self.transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
    def __getitem__(self, item):
        img_path, label = self.imgs[item]
        image = Image.open(img_path)
        data = self.transform(image)
        # writer.add_image('image', data)
        return data, label
    def __len__(self):
        return len(self.imgs)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--num_worker', type=int, help='dataloader num workers', default=4)
parser.add_argument('--epoch', type=int, help='max epoch', default=5)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-4)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--visitual', type=int, help='train image visitual', default=0)
parser.add_argument('--lr_decay', type=float, help='lr decay', default=1e-4)
parser.add_argument('--train', type=str, help='train dataset root', default='dataset.csv')
parser.add_argument('--test', type=str,default='dataset.csv', help='test dataloader root path')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conv1(model, img, global_step):
    for name, layer in model._modules.items():
        if name == 'features':
            conv1 = layer[0]
            # print(conv1)
            # img = img.to(device)
            datas = conv1(img)
            count = 0
            for data in datas:
                x = data
                x1 = x.unsqueeze(0)

                img_transpose = x1.transpose(0, 1)
                img_grid = make_grid(img_transpose, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('image_batch_{}'.format(count), img_grid, global_step=global_step, dataformats='CHW')
                count += 1

def conv2(model, img, global_step):
    for name, layer in model._modules.items():
        if name == 'features':
            conv1 = layer[0]
            # img = img.to(device)
            output = conv1(img)
            output = layer[1](output)
            output = layer[2](output)
            output = layer[3](output) #conv2
            count = 0
            for data in output:
                x = data
                x1 = x.unsqueeze(0)
                img_transpose = x1.transpose(0, 1)
                img_grid = make_grid(img_transpose, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('conv2_batch_{}'.format(count), img_grid, global_step=global_step)
                count += 1

def conv3(model, img, global_step):
    for name, layer in model._modules.items():
        if name == 'features':
            output = layer[0](img)#conv1
            output = layer[1](output)
            output = layer[2](output)
            output = layer[3](output)#conv2
            output = layer[4](output)
            output = layer[5](output)
            output = layer[6](output) #conv3
            count = 0
            for data in output:
                x = data
                x1 = x.unsqueeze(0)
                img_transpose = x1.transpose(0, 1)
                img_grid = make_grid(img_transpose, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('conv3_batch_{}'.format(count), img_grid, global_step=global_step)
                count += 1

def conv4(model, img, global_step):
    for name, layer in model._modules.items():
        if name == 'features':
            output = layer[0](img)#conv1
            output = layer[1](output)
            output = layer[2](output)
            output = layer[3](output)#conv2
            output = layer[4](output)
            output = layer[5](output)
            output = layer[6](output) #conv3
            output = layer[7](output)
            output = layer[8](output)
            output = layer[9](output) #conv4
            count = 0
            for data in output:
                x = data
                x1 = x.unsqueeze(0)
                img_transpose = x1.transpose(0, 1)
                img_grid = make_grid(img_transpose, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('conv4_batch_{}'.format(count), img_grid, global_step=global_step)
                count += 1

def conv5(model, img, global_step):
    for name, layer in model._modules.items():
        if name == 'features':
            output = layer[0](img)#conv1
            output = layer[1](output)
            output = layer[2](output)
            output = layer[3](output)#conv2
            output = layer[4](output)
            output = layer[5](output)
            output = layer[6](output) #conv3
            output = layer[7](output)
            output = layer[8](output)
            output = layer[9](output) #conv4
            count = 0
            for data in output:
                x = data
                x1 = x.unsqueeze(0)
                img_transpose = x1.transpose(0, 1)
                img_grid = make_grid(img_transpose, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('conv5_batch_{}'.format(count), img_grid, global_step=global_step)
                count += 1


def train(**kwargs):
    model = Alexnet(num_classes=17)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    # writer.add_graph()
    train_data = Flower(root=args.train, sep = ' ', train=True)
    val_data = Flower(root=args.test, sep = ' ', train=False, test=True)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        running_loss = 0
        train_acc = 0
        for index, (data, label) in enumerate(train_dataloader):
            input = Variable(data)
            label = Variable(label)
            label = label.to(device)
            input = input.to(device)
            # if torch.cuda.is_available():
            #     input = input.cuda()
            #     label = label.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            # writer.add_image('image', input[0], index + epoch * len(train_dataloader))
            prediction = torch.max(score, 1)[1]
            train_accuracy = (prediction == label).sum()
            train_acc += train_accuracy.float() / args.batch_size
            running_loss += loss.item()
            print('loss = {}, train accuracy = {}, iteration/epoch = {}/{}'.format(loss.item(), train_acc / (index + 1), index, epoch))
            writer.add_scalar('Train/loss', loss.item(), index + epoch * (len(train_dataloader)))
            writer.add_scalar('train_acc', train_acc / (index + 1), index + epoch * len(train_dataloader))
            writer.add_scalar('avg_loss', running_loss / (index + 1), epoch * len(train_dataloader) + index)

            if args.visitual == 1:
                x = data.to(device)
                global_step = index + epoch * len(train_dataloader)
                conv1(model, x, global_step)
                conv2(model, x, global_step)
                conv3(model, x, global_step)
                conv4(model, x, global_step)
                conv5(model, x, global_step)
                # print(name, layer_name)
        torch.save(model.state_dict(), 'alexnet.pth')
        val_accuracy = val(model, val_dataloader)
        writer.add_scalar('accuracy', val_accuracy, epoch)
        print('accuracy = {}, epoch = {}'.format(val_accuracy, epoch))
        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram('{}/{}'.format(layer, attr), param, epoch)
            # writer.add_histogram()



def features_vis_conv1(model, x, global_step):
    for name, layer in model._modules.items():
        if name == 'conv1':
            batches = layer(x)
            count = 0
            for features in batches:
                feature_maps = features.unsqueeze(0)
                feature_maps = feature_maps.transpose(0, 1)
                img_grid = make_grid(feature_maps, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('resnet_feature_conv1_batch_{}'.format(count), img_grid, global_step)
                count += 1

def features_vis_conv2(model, x, global_step):
    for name, layer in model._modules.items():
        if name == 'conv1':
            features = layer(x)
        if name == 'conv2_x':
            batches = layer(features)
            count = 0
            for features in batches:
                feature_maps = features.unsqueeze(0)
                feature_maps = feature_maps.transpose(0, 1)
                img_grid = make_grid(feature_maps, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('resnet_feature_conv2_batch_{}'.format(count), img_grid, global_step)
                count += 1

def features_vis_conv3(model, x, global_step):
    for name, layer in model._modules.items():
        if name == 'conv1':
            features = layer(x)
        if name == 'conv2_x':
            features = layer(features)
        if name == 'conv3_x':
            batches = layer(features)
            count = 0
            for features in batches:
                feature_maps = features.unsqueeze(0)
                feature_maps = feature_maps.transpose(0, 1)
                img_grid = make_grid(feature_maps, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('resnet_feature_conv3_batch_{}'.format(count), img_grid, global_step)
                count += 1


def features_vis_conv4(model, x, global_step):
    for name, layer in model._modules.items():
        if name == 'conv1':
            features = layer(x)
        if name == 'conv2_x':
            features = layer(features)
        if name == 'conv3_x':
            features = layer(features)
        if name == 'conv4_x':
            batches = layer(features)
            count = 0
            for features in batches:
                feature_maps = features.unsqueeze(0)
                feature_maps = feature_maps.transpose(0, 1)
                img_grid = make_grid(feature_maps, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('resnet_feature_conv4_batch_{}'.format(count), img_grid, global_step)
                count += 1


def features_vis_conv5(model, x, global_step):
    for name, layer in model._modules.items():
        if name == 'conv1':
            features = layer(x)
        if name == 'conv2_x':
            features = layer(features)
        if name == 'conv3_x':
            features = layer(features)
        if name == 'conv4_x':
            features = layer(features)
        if name == 'conv5_x':
            batches = layer(features)
            count = 0
            for features in batches:
                feature_maps = features.unsqueeze(0)
                feature_maps = feature_maps.transpose(0, 1)
                img_grid = make_grid(feature_maps, normalize=True, scale_each=True, padding=8, nrow=8)
                writer.add_image('resnet_feature_conv5_batch_{}'.format(count), img_grid, global_step)
                count += 1

def train_resnet():
    model = resnet18(num_classes=17)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    # writer.add_graph()
    train_data = Flower(root=args.train, sep = ' ', train=True)
    val_data = Flower(root=args.train, sep = ' ', train=False)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        running_loss = 0
        train_acc = 0
        for index, (data, label) in enumerate(train_dataloader):
            input = Variable(data)
            label = Variable(label)
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            global_step = index + epoch * len(train_dataloader)
            running_loss += loss.item()
            features_vis_conv1(model, input, global_step)
            features_vis_conv2(model, input, global_step)
            features_vis_conv3(model, input, global_step)
            features_vis_conv4(model, input, global_step)
            features_vis_conv5(model, input, global_step)

            prediction = torch.max(score, 1)[1]
            train_accuracy = (prediction == label).sum()
            train_acc += train_accuracy.float() / args.batch_size
            print('loss = {}, train accuracy = {}, iteration/epoch = {}/{}'.format(loss.item(), train_acc / (index + 1), index, epoch))
            writer.add_scalar('Train/loss', loss.item(), index + epoch * (len(train_dataloader)))
            writer.add_scalar('train_accuarcy', train_acc / (index + 1), epoch * len(train_dataloader) + index)
            writer.add_scalar('avg_loss', running_loss / (index + 1), epoch * len(train_dataloader) + index)
        torch.save(model.state_dict(), 'resnet17.pth')
        val_accuracy = val(model, val_dataloader)
        writer.add_scalar('accuracy', val_accuracy, epoch)
        print('accuracy = {}, epoch = {}'.format(val_accuracy, epoch))
def val(model, dataloader):
    model.eval()
    a = 0.0
    num = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            
            val_input = Variable(data)
            val_label = Variable(label)
            if torch.cuda.is_available():
                val_input = val_input.cuda()
                val_label = val_label.cuda()
            score = model(val_input)
            softmax_score = torch.nn.functional.softmax(score)
            a = a + sum(np.argmax(np.array(softmax_score.data.cpu()), axis=1) == np.array(val_label.data.cpu()))
            num = num + len(label)
    model.train()
    accuracy = a /num
    return accuracy

if __name__ == '__main__':
    args = parser.parse_args()
    train()
    # train_resnet()
    writer.close()
    # train_resnet()