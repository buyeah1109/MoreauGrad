import torch
import matplotlib.pyplot as plt

def saveimg(input, SAVE_PATH, cmap=None):

    if input.shape[0] == 3:
        inp = torch.permute(torch.squeeze(input), dims=(1, 2, 0)).detach().cpu().numpy()
    else:
        inp = input.detach().cpu().numpy()
    inp = (inp - inp.min()) / (inp.max() - inp.min())

    plt.figure()
    plt.axis('off')
    if cmap is not None:
        plt.imshow(inp, cmap=cmap)
    else:
        plt.imshow(inp)
    plt.savefig(SAVE_PATH, bbox_inches='tight', pad_inches=0)
    plt.close()
