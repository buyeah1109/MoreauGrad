import torchvision.transforms as transforms
from data_util import plot_saliency
from efficientnet_pytorch import EfficientNet
import explainer
from PIL import Image

def sparse_interpret(net, input, label):
    sparse_explainer = explainer.SparseMoreauExplainer(MAX_ITR=200)
    sparse_explainer.set_requires_grad(False)
    interpretation = sparse_explainer.explain(net, input, label)
    plot_saliency(interpretation.detach(), './church_interpret.png')

def group_sparse_interpret(net, input, label):
    sparse_explainer = explainer.GroupSparseMoreauExplainer(MAX_ITR=200)
    sparse_explainer.set_requires_grad(False)
    interpretation = sparse_explainer.explain(net, input, label)
    plot_saliency(interpretation.detach(), './church_interpret_group.png')

if __name__ == '__main__':
    sample_image_path = './church_image.png'
    # Church is the 497-th class in ImageNet
    label = 497

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    sample_image = Image.open(sample_image_path).convert('RGB')
    sample_image = transform(sample_image).unsqueeze(0).cuda()

    net = EfficientNet.from_pretrained('efficientnet-b0').cuda()
    sparse_interpret(net, sample_image, label)
    group_sparse_interpret(net, sample_image, label)