from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
# from datasets import *
# from softplus_cifar_resnet import soft_tiny_ResNet18
# from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorch Training Script')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


# Specify dataset in PyTorch form
# Dataset object should be defined as in: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
train_dataset = ...
val_dataset = ...

val_len = int(len(val_dataset))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=2, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=2)

print('==> Building model..')

net = ...
# net = EfficientNet.from_pretrained('efficientnet-b0').cuda()
net = net.cuda()
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./tinyimgnet_checkpoint/tiny_imagenet_eff0.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(BackgroundGenerator(trainloader)),
                total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data

        pbar.set_description("Loss: {:.3f}, Acc: {:.3f}".format(train_loss / (batch_idx + 1), 100.*correct/total))
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = Variable(inputs).cuda(), Variable(targets).cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().data

        # progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('tinyimgnet_checkpoint'):
            os.mkdir('tinyimgnet_checkpoint')
        torch.save(state, './tinyimgnet_checkpoint/tiny_imagenet_eff0.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+350):
    train(epoch)
    test(epoch)
    if epoch == 150:
        args.lr = .01
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    if epoch == 250:
        args.lr = .001
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr