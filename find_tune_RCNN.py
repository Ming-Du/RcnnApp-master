from __future__ import division, print_function, absolute_import
import os
from preprocessing_RCNN import *
import config
from alexnet import *
import torch
from torch import nn, optim
from torch.autograd import Variable
from flower_dataloader import *
from torch.utils.tensorboard import SummaryWriter
import argparse

writer = SummaryWriter(comment='rcnn-fine-tune')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='batch size', default=16)
parser.add_argument('--num_worker', type=int, help='dataloader num workers', default=4)
parser.add_argument('--epoch', type=int, help='max epoch', default=30)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-6)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
parser.add_argument('--visitual', type=int, help='train image visitual', default=0)
parser.add_argument('--lr_decay', type=float, help='lr decay', default=1e-4)
parser.add_argument('--train', type=str, help='train dataset root', default='dataset.csv')
parser.add_argument('--test', type=str,default='dataset.csv', help='test dataloader root path')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def fine_tune_alexnet(num_class, X, Y, save_model_path, fine_tune_model_path):
    model = Alexnet(num_classes=17)
    model.load_state_dict(torch.load('./alexnet.pth'))
    model.classifier[6] = nn.Linear(4096, num_class)
    # model.classifier[6] = nn.Linear(num_class, num_class)
    print(model)
    # print(len(X), len(Y))
    # split_step = int(0.9 * len(X))
    train_data = TwoFlower(X, Y, train=True)
    val_data = TwoFlower(X, Y, train=False)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    # train_data = []
    # val_data = []
    # for i in range(split_step):
    #     train_data.append([X[i], Y[i]])
    # for i in range(split_step, len(X), 1):
    #     val_data.append([X[i], Y[i]])
    # train_data = np.vstack([X[0:split_step], Y[:split_step]])
    # val_data = np.vstack([X[split_step:len(X)], Y[split_step:len(X)]])
    # train_data = (X[0:split_step], Y[:split_step])
    # val_data = (X[split_step:len(X)], Y[split_step:len(X)])
    # print(len(train_data))
    # print(len(val_data))
    # print(model)
    # print(model.classifier[6])
    # print(model)
    # model.train()
    train(model, train_dataloader, test_dataloader, args.epoch, save_model_path, fine_tune_model_path)
    print('==' * 10)

def train(model, train_data, val_data, epoch, save_model_path, fine_tune_model_path):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
    for i in range(epoch):
        running_loss = 0
        # index = 0
        train_acc = 0
        for index, (data, label) in enumerate(train_data):
            data = Variable(data)
            label = Variable(label)
            
            # print(label.shape)
            N = label.size(0)
            # print(N)
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            score = model(data)
            # loss = criterion(score, torch.argmax(label, dim=1))

            # print(np.argmax(label.data.cpu()))
            # print(score.shape)
            # print(label.shape)
            log_prob = nn.functional.log_softmax(score, dim=1)

            # log_prob2 = nn.functional.log_softmax(score, dim=0)
            # print(log_prob2)
            # print(log_prob)
            # print(log_prob.shape)
            loss = -torch.sum(log_prob * label) / N
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            prediction = torch.max(score, 1)[1]
            train_accurracy = (prediction == torch.max(label, 1)[1]).sum().float() / args.batch_size
            train_acc += train_accurracy

            imgs = normalize_inverse(data, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            batch_index = 0
            for img in imgs:
                writer.add_image('batch_image_{}'.format(batch_index), img, global_step=666)
                batch_index += 1
            print('loss = {}, train acc = {}, iteration/epoch = {}/{}, epoch = {}'.format(loss, train_accurracy, index, i, epoch))
            writer.add_scalar("Training/loss", loss, index + i * len(train_data))
            writer.add_scalar('average loss', running_loss / (index + 1), index + i * len(train_data))
            writer.add_scalar('train_acc', train_acc / (index + 1), index + i * len(train_data))

            # writer.add_scalar('log prob', log_prob, index)
            index = index + 1
        torch.save(model.state_dict(), os.path.join(save_model_path, 'find_tune_flower.pth'))
        accuracy = eval(model, val_data)
        print('val accuracy is {} epoch ={}'.format(accuracy, i))
        writer.add_scalar('accuracy', accuracy, i)

def normalize_inverse(imgs, mean, std):
    '''
    :param img:  tensor[b, c, h, w]
    :param mean: [x, y, z]
    :param std: [x, y, z]
    :return: [0-255] pixel data
    '''
    datas = []
    for img in imgs:
        for i in range(len(img)):
            img[i] = img[i] * std[i] + mean[i]
        img = img.data.cpu()
        img = np.array(img * 255).clip(0, 255).squeeze().astype('uint8')
        datas.append(img)
    return datas

def eval(model, val_data):
    model.eval()
    a = 0.0
    num = 0.0
    for index, (data, label) in enumerate(val_data):
        with torch.no_grad():
            input = Variable(data)
            label = Variable(label)
            if torch.cuda.is_available():
                input = input.cuda()
                label = label.cuda()
            score = model(input)
            softmax_score = nn.functional.softmax(score)
            # print('eval label:', np.argmax(np.array(label.data.cpu()), axis=1))
            a = a + sum(np.argmax(np.array(softmax_score.data.cpu()), axis=1) == np.argmax(np.array(label.data.cpu()), axis=1))
            num = num + len(label)
    model.train()
    accuracy = a / num
    print('eval accuracy = {}'.format(accuracy))
    return accuracy

def findtune_data():
    data_set = config.FINE_TUNE_DATA
    if len(os.listdir(data_set)) == 0:
        print('Reading data')
        load_train_proposals(config.FINE_TUNE_LIST, num_class=2, save=True, threshold=0.2, is_svm=False, save_path=data_set)
    print('Loading data')
    X, Y = load_from_npy(data_set)
    fine_tune_alexnet(config.FINE_TUNE_CLASS, X, Y, config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH)

if __name__ == '__main__':
    args = parser.parse_args()
    findtune_data()