from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils import data
import argparse
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import time
from prefetch_generator import BackgroundGenerator
import os

from efficientnet_pytorch import EfficientNet
from datasets import *
import explainer
import pickle
from data_util import *
from visual import saveimg
from ssim import SSIM

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

SAVE_PATH = ...
os.makedirs(SAVE_PATH, exist_ok=True)

topk_ratio = .4

def attack_experiment(model, dataloader, interpreter, attacker, max_samples=50):
    model.eval()

    pbar = tqdm(enumerate(BackgroundGenerator(dataloader)),
                    total=len(dataloader))
    avg_normalized_distance = 0
    avg_topk = 0
    avg_ssim = 0

    cnt = 0
    
    for i, datas in pbar:
        cnt += 1
        x, label = datas
        x = x.cuda()
        label = label.cuda()
        x = Variable(x, requires_grad = True)

        original_interpretation = interpreter.explain(model, x, label)

        # only non-zero pixels are counted
        TOPK = int(((original_interpretation.abs() > 0).sum() * topk_ratio).item())

        mask_original_interpretation = explainer.get_top_k_mask(original_interpretation, TOPK)

        # Ghorbani interpretation attack requires gradient, set it to true
        # This will increase GPU memory usage
        interpreter.set_requires_grad(True)

        # Ghorbani top-k interpretation attack, it's more powerful but takes longer time
        ghorbani_base_attack = attacker.explain(model, x, label, classifier_model = model)

        # Gaussian random attack, it's faster but less powerful
        # gaussian_random_attack = attacker.explain(x)

        # Save GPU memory
        interpreter.set_requires_grad(False)
        if (ghorbani_base_attack is not None):
            attacked_interpretation = interpreter.explain(model, ghorbani_base_attack, label)
            
            mask_attacked_interpretation = explainer.get_top_k_mask(attacked_interpretation,TOPK)

            ssim = SSIM()
            single_sample_ssim = ssim(original_interpretation, attacked_interpretation).item()
            single_sample_topk = explainer.top_k_intersection(mask_original_interpretation, mask_attacked_interpretation).type(x.type()) / TOPK
            single_sample_ned = explainer.normalized_eculidean_distance(original_interpretation, attacked_interpretation).item()

            avg_normalized_distance += single_sample_ned
            avg_topk += single_sample_topk.item()
            avg_ssim += single_sample_ssim

            print(single_sample_ned, single_sample_topk.item(), single_sample_ssim)
            
        pbar.set_description("Avg Robustness:{:.3f} ".format(
            avg_topk / (i + 1)
        ))
        
        os.makedirs(SAVE_PATH + '/original', exist_ok=True)
        os.makedirs(SAVE_PATH + '/original_attack', exist_ok=True)
        os.makedirs(SAVE_PATH + '/interpretation', exist_ok=True)
        os.makedirs(SAVE_PATH + '/interpretation_attack', exist_ok=True)

        saveimg(x.squeeze(), SAVE_PATH + '/original/{}.png'.format(i))
        saveimg(ghorbani_base_attack.squeeze(), SAVE_PATH + '/original_attack/{}.png'.format(i))
        plot_saliency(original_interpretation.detach(), SAVE_PATH + '/interpretation/{}.png'.format(i))
        plot_saliency(attacked_interpretation.detach(), SAVE_PATH + '/interpretation_attack/{}.png'.format(i))

        if cnt == max_samples:
            break


if __name__ == '__main__':

    # This setting is for efficientNet-b0, update it for your own network, it should be consistent with input size
    TOPK = ...
    # TOPK = int(topk_ratio * 3 * 224 * 224)

    # Specify SAVE_PATH for save visualization
    SAVE_PATH = ...
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Specify dataset in PyTorch form
    # Dataset object should be defined as in: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    test_dataset = ...
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    # Specify your own CNN here, it should have output shape as [batchsize, num_of_classes]
    net = ...
    # GPU is suggested
    net.cuda()

    # Specify interpreter
    MoreauInterpreter = explainer.SparseMoreauExplainer()

    # Specify attacker
    Attack_magnitude = 10
    Attacker = explainer.PureGhorbaniAttacker(Attack_magnitude, MoreauInterpreter, 10, Attack_magnitude / 100, TOPK)

    # Another attacker: Gaussian random attack
    # Attack_magnitude = 1e-2
    # Attacker = explainer.SimpleGaussianAttacker()

    attack_experiment(net, test_data_loader, MoreauInterpreter, Attacker)