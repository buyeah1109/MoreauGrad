import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def MoreauEnvelope(net, x, label, approxi, lamb=1, sigma=0.1, samples=64):
    noise_level = (x.max()-x.min()) * sigma
    x_batch = torch.vstack([Variable(x + torch.randn(x.shape).cuda() * noise_level, requires_grad=True) for i in range(samples)])
    output = net(x_batch + approxi)
    output = torch.mean(output, dim=0, keepdim=True)

    label_score = output[0][label]

    penalty = 0.5 * (torch.norm(approxi, p=2, dim = (1, 2, 3)) ** 2) / lamb 

    return label_score, penalty

def soft_threshold(approxi, threshold):
    larger = approxi > threshold
    smaller = approxi < -1 * threshold
    mask = torch.logical_or(larger, smaller)
    approxi = approxi * mask
    subtracted = larger * -1 * threshold
    added = smaller * threshold
    approxi = approxi + subtracted + added

    return approxi

def group_soft_threshold(x, threshold, group_dim):
    kernel = torch.ones((3, 3, group_dim, group_dim)).cuda()
    tmp = x ** 2
    tmp = F.conv2d(tmp, kernel, stride=group_dim)
    grouped_map = tmp  ** .5

    # print(grouped_map, grouped_map.shape)
    map_above_threshold = grouped_map > threshold
    grouped_map *= map_above_threshold
    is_zero = grouped_map == 0
    grouped_map += is_zero # prevent division by zero
    coeff = (1 - threshold / grouped_map)
    coeff *= map_above_threshold
    # print(coeff, coeff.shape)
    upsample = nn.Upsample(scale_factor=group_dim, mode='nearest')
    coeff = upsample(coeff)
    # print(coeff, coeff.shape)
    output = x * coeff
    return output