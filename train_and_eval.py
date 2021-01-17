import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.backends.cudnn as cudnn
import time
import sys
import os

import model
from utils.get_imagenet import get_train_dataloader, get_val_dataloader

parser = argparse.ArgumentParser(description='Train and evaluate models for CIFAR100 in pytorch')
parser.add_argument('--data-path', default='/gaoyangcheng/dataset/imagenet/', 
                    help='Path to dataset', type=str)
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset to train or evaluate')
parser.add_argument('--learning-rate', default=0.1, type=float, metavar='LR', 
                    help='initial learning rate(default: 0.1)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', 
                    help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--net', default='resnet18', type=str, metavar='NET',
                    help='net type (default: resnet18)')
parser.add_argument('--n-workers', default=0, type=int, metavar='Wks',
                    help='number of workers')

parser.add_argument('--path-save', default='/gaoyangcheng/dicts-vs-kmeans/model_path', type=str,
                    help='Path to save model')
parser.add_argument('--path-model', default='', type=str,
                    help='Path to model pretrained')
parser.add_argument('--eval', default=True, type=bool,
                    help='evaluate model')

def train(net, trainloader, criterion, optimizer):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, start=0):
        if torch.cuda.is_available():
            input, label = data[0].cuda(), data[1].cuda()
        else:
            input, label = data[0], data[1]
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if i % 10 == 9:
            print('[{}, {}] loss: {:.3f}'.format(epoch + 1, i + 1, loss.item()))

@torch.no_grad()
def evaluate(net, val_loader, criterion):
    net.eval()
    top5 = AverageMeter()
    top1 = AverageMeter()
    losses = AverageMeter()
    
    for i, data in enumerate(val_loader):
        if torch.cuda.is_available():
            input, label = data[0].cuda(), data[1].cuda()
        else:
            input, label = data[0], data[1]
        output = net(input)
        loss = criterion(output, label)
        prec1, prec5 = accuracy(output.data, label.data, topk=(1, 5))
        
        losses.update(prec1.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        print('Test: [{0}/{1}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            i, len(val_loader), 
            loss=losses,
            top1=top1, top5=top5))
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    #load dataset
    print('Dataset: {}'.format(args.dataset))
    
    if args.dataset == 'imagenet':
        train_loader = get_train_dataloader(data_path=args.data_path, batchsize=args.batch_size, num_workers=args.n_workers,
                                                                    distributed=False)
        val_loader = get_val_dataloader(data_path=args.data_path, batchsize=args.batch_size, num_workers = args.n_workers)
        num_classes = 1000
    #load model
    net_name = args.net
    print('Model: {}'.format(net_name))
    net = model.__dict__[net_name](pretrained=args.eval, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    #device
    if torch.cuda.is_available():
        net = net.cuda()
        criterion = criterion.cuda()
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    if args.eval: #evaluate
        print('Evaluating started...')
        print('Batchsize: {}'.format(args.batch_size))
        if args.path_model:
            PATH = os.path.join(args.path_model, '{}-{}-best.pth'.format(net_name, args.epochs))
            net.load_state_dict(torch.load(PATH))
        evaluate(net, val_loader, criterion)
    
    else: #train
        print('Hyperparameters:\nlr: {}, momentum: {}, weight_decay: {}'.format(args.learning_rate, args.momentum, args.weight_decay))
        print('Training started...')
        print('Epochs: {}'.format(args.epochs))\
        
        best_acc = 0.0
        time_start = time.time()
        for epoch in range(args.epochs):
            train(net, train_loader, criterion, optimizer)
            scheduler.step()
            acc = evaluate(net, val_loader, criterion)
            if epoch > 10 and best_acc < acc:
                best_acc = acc
                torch.save(net.state_dict(), os.path.join(args.path_save, 
                            '{}-{}-best.pth'.format(net_name, args.epochs)))
        time_end = time.time()
        print('Training finished.')
        print('TrainingTime: {:.3f}s.'.format(time_end - time_start))

        if not os.path.exists(args.path_save):
            os.mkdir(args.path_save)
        PATH = os.path.join(args.path_save, '{}-{}.pth'.format(net_name, args.epochs))
        print('Path: ' + PATH)
        torch.save(net.state_dict(), PATH)
        print('Path Saved.')

