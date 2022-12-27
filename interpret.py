import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data
import argparse
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import time
from prefetch_generator import BackgroundGenerator
import os
from datasets import *
from visual import saveimg
from data_util import plot_saliency
from efficientnet_pytorch import EfficientNet
import explainer

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    # PATH to folder for saving visualization
    SAVE_PATH = ...
    os.makedirs(SAVE_PATH, exist_ok=True)

    MAX_SAMPLE = 100

    # Specify dataset in PyTorch form
    # Dataset object should be defined as in: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    test_dataset = ...  
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    # Specify your own CNN here, it should have output shape as [batchsize, num_of_classes]
    net = ...
    # net = inception_v3(pretrained=True)
    # net = EfficientNet.from_pretrained('efficientnet-b0')
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        net.cuda()
    
    net.eval()
    pbar = tqdm(enumerate(BackgroundGenerator(test_data_loader)),
                total=MAX_SAMPLE)
    start_time = time.time()

    
    for i, datas in pbar:

        if(i == MAX_SAMPLE):
            break

        x, label = datas
        x = Variable(x, requires_grad = True)

        if use_cuda:
            x = x.cuda()
            label = label.cuda()

        # Specify explainer and interpret
        more_explainer = explainer.SparseMoreauExplainer(lamb=1, LR=0.1, MAX_ITR=300, SIGMA=1e-1, samples=64, soft=5e-3)
        interpretation = more_explainer.explain(net, x, label)

        prepare_time = start_time-time.time()

        # save visualization
        saveimg(x.squeeze(), SAVE_PATH + '/original/{}.png'.format(i))
        plot_saliency(interpretation.detach(), SAVE_PATH + '/{}.png'.format(i))

        process_time = start_time-time.time()-prepare_time
        pbar.set_description("Effi: {:.2f}, samples: {}/{}:".format(
            process_time/(process_time+prepare_time), i, MAX_SAMPLE))
        start_time = time.time()
