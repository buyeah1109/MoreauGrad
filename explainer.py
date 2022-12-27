import torch
from torch.autograd import Variable
import math
from losses import *
from torch.nn.parameter import Parameter
from tqdm import tqdm
# from integrated import integrated_gradients
# from RelEx.models.saliency import RelEx


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


imagenet_scale= True
mnist_scale=False

def normalized_eculidean_distance(input1, input2):
	input1 = input1 / input1.norm(p=2)
	input2 = input2 / input2.norm(p=2)

	return torch.norm(input1 - input2, p=2)

def clamp_batch(x):
	if (imagenet_scale):
		pass
	elif (mnist_scale):
		x.clamp_(min=-.4242, max= 2.821)
	else:
		x[:,0] = torch.clamp(x[:,0], min=-2.429, max= 2.514)
		x[:,1] = torch.clamp(x[:,1], min=-2.418, max= 2.596)
		x[:,2] = torch.clamp(x[:,2], min=-2.221, max= 2.753)

def normalize_if_necessary(x, x_0, l2):
	x_flat = x.reshape(x.shape[0],-1)
	x_0_flat = x_0.reshape(x.shape[0],-1)
	diff_flat = x_flat - x_0_flat
	norms = diff_flat.norm(dim=1)
	x_flat[(norms > l2).nonzero()] =  (x_0_flat + (l2*diff_flat.t()/norms).t())[(norms > l2).nonzero()]
	clamp_batch(x)

def get_top_k_mask(x, k):
	flattened = x.reshape(x.shape[0], -1)
	x_sparsified = (flattened.scatter(
		1,
		torch.topk(flattened,x[0].nelement()-k,largest=False,dim=1)[1],
		0).reshape(x.shape) > 0).type(x.type())
	return x_sparsified

def top_k_intersection(x_1_mask, x_2_mask):
	x_l_flat = x_1_mask.reshape(x_1_mask.shape[0], -1)
	x_2_flat = x_2_mask.reshape(x_2_mask.shape[0], -1)
	return (x_l_flat*x_2_flat).sum(dim=1)

class Explainer: 
	def explain(self, model, x):
		pass

class Attacker: 
	def explain(self, model, x):
		pass

class BaseExplainer(Explainer):
	def __init__(self, requires_grad=False):
		self.requires_grad = requires_grad
	def explain(self, model, x, label):

		logits = model(x)
		max_logit = logits[0][label]

		x_grad = torch.autograd.grad(max_logit, x, create_graph=True, retain_graph=True)[0]

		x_grad = torch.abs(x_grad)
		return x_grad

class SparseMoreauExplainer(Explainer):
	def __init__(self, lamb=1, SIGMA=.1, LR=.1, MAX_ITR=300, samples=32, soft=5e-3, requires_grad=False):
		self.lamb = lamb
		self.sigma = SIGMA
		self.itr = MAX_ITR
		self.lr = LR
		self.samples = samples
		self.soft = soft
		self.requires_grad = requires_grad

	def set_requires_grad(self, requires=True):
		self.requires_grad = requires

	def explain(self, model, x, label):
		model.eval()
		approxi = torch.zeros(x.shape) 

		approxi = approxi.cuda()
		approxi = Parameter(approxi, requires_grad = True)
		avg_approxi = 0
		cnt = 0

		itr_bar = tqdm(range(self.itr))
		for n_itr in itr_bar:
			floss, ploss = MoreauEnvelope(model, x, label, approxi, self.lamb, sigma=self.sigma, samples=self.samples)
			loss = floss + ploss

			gradient = torch.autograd.grad(loss, approxi)[0]

			new_approxi = approxi - self.lr * gradient
			if self.soft > 0:
				new_approxi = soft_threshold(new_approxi, self.soft * self.lr)
			approxi = new_approxi

			if self.itr - (n_itr + 1) <= 100:
				cnt += 1
				avg_approxi += approxi.detach()

			itr_bar.set_description("loss: {:.2f}, floss: {:.6f}, ploss: {:.6f} ".format(
				loss.item(), floss.item(), self.lamb*2* ploss.item()
			))
		
		if self.requires_grad:
			result = model(x+approxi)
			result = result[0][label]		
			grad, = torch.autograd.grad(result, x, create_graph=self.requires_grad)
			return ((avg_approxi / cnt).abs() > 0) * grad.abs()
		
		else:
			return avg_approxi / cnt

class GroupSparseMoreauExplainer(Explainer):
	def __init__(self, lamb=1, SIGMA=.1, LR=.1, MAX_ITR=300, samples=32, soft=5e-2, gdim=14, requires_grad=False):
		self.lamb = lamb
		self.sigma = SIGMA
		self.itr = MAX_ITR
		self.lr = LR
		self.samples = samples
		self.soft = soft
		self.gdim = gdim
		self.requires_grad = requires_grad

	def set_requires_grad(self, requires=True):
		self.requires_grad = requires

	def explain(self, model, x, label):
		model.eval()
		approxi = torch.zeros(x.shape) 

		approxi = approxi.cuda()
		approxi = Parameter(approxi, requires_grad = True)
		avg_approxi = 0
		cnt = 0

		itr_bar = tqdm(range(self.itr))
		for n_itr in itr_bar:
			floss, ploss = MoreauEnvelope(model, x, label, approxi, self.lamb, sigma=self.sigma, samples=self.samples)
			loss = floss + ploss

			gradient = torch.autograd.grad(loss, approxi)[0]

			new_approxi = approxi - self.lr * gradient
			if self.soft > 0:
				new_approxi = group_soft_threshold(new_approxi, self.soft * self.lr, self.gdim)
			approxi = new_approxi

			if self.itr - (n_itr + 1) <= 100:
				cnt += 1
				avg_approxi += approxi.detach()

			itr_bar.set_description("loss: {:.2f}, floss: {:.6f}, ploss: {:.6f} ".format(
				loss.item(), floss.item(), self.lamb*2* ploss.item()
			))

		if self.requires_grad:
			result = model(x+approxi)
			result = result[0][label]		
			grad, = torch.autograd.grad(result, x, create_graph=self.requires_grad)
			return ((avg_approxi / cnt).abs() > 0) * grad.abs()
		
		else:
			return avg_approxi / cnt

class SmoothGradExplainer(Explainer):
	def __init__(self, stdev=0.15, n_samples=8192, batch_size = 64, requires_grad=False):
		self.stdev = stdev
		self.n_samples = n_samples
		self.batch_size = batch_size
		self.requires_grad = requires_grad

	def set_requires_grad(self, requires=True):
		self.requires_grad = requires

	def explain(self, net, x, label):
		total_samples = self.n_samples
		batchsize = self.batch_size
		batches = total_samples / batchsize
		grads = 0
		for batch_cnt in range(int(batches)):
			sigma = self.stdev * (torch.max(x) - torch.min(x))
			x_stack = torch.vstack(tensors=[x for i in range(batchsize)])
			noise_stack = torch.vstack(tensors=[torch.randn(x.shape).cuda() * sigma for i in range(batchsize)])
			pred = net(x_stack + noise_stack)
			pred = torch.sum(pred, dim=0, keepdim=True)
			label_score = pred[0][label]
			x_grad, = torch.autograd.grad(label_score, x_stack, create_graph=self.requires_grad)
			x_grad = torch.sum(x_grad, dim=0, keepdim=True)
			grads += x_grad

			del x_grad, x_stack, noise_stack

		return (grads / total_samples).abs()

# class IntegratedGradExplainer(Explainer):
# 	def __init__(self, n_samples=50, ref = None, requires_grad=False):
# 		self.n_samples = n_samples
# 		self.ref = ref
# 		self.requires_grad = requires_grad

# 	def set_requires_grad(self, requires=True):
# 		self.requires_grad = requires

# 	def explain(self, net, x, label):
# 		intg_grad = integrated_gradients(net, x, label, steps=self.n_samples, baseline = self.ref, requires_grad=self.requires_grad)

# 		return intg_grad.abs()

# class RelEx_Loss(nn.Module):
#     def __init__(self, lambda1, lambda2):
#         super().__init__()

#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.eps = 1e-7  # to defense log(0)

#     def forward(self, scores, m):
#         foregnd_scores, backgnd_scores = scores
#         foregnd_term = -torch.log(foregnd_scores)
#         m_l1_term = self.lambda1 * torch.abs(m).view(m.size(0), -1).sum(dim=1)
#         backgnd_term = -self.lambda2 * torch.log(1 - backgnd_scores + self.eps)
#         return foregnd_term + m_l1_term + backgnd_term

# class RelExExplainer(Explainer):
# 	def __init__(self, requires_grad=False):
# 		self.requires_grad = requires_grad
# 		self.criterion = RelEx_Loss(lambda1=1e-4, lambda2=1.)

# 	def set_requires_grad(self, requires=True):
# 		self.requires_grad = requires

# 	def explain(self, net, x, label):
# 		relex = RelEx(net, device = torch.device('cuda'))
# 		sal, accu = relex(x, label)

# 		return sal.abs()

class L0Explainer(Explainer):

	def __init__(self, l_0, flatten = 0, requires_grad=False):
		self.l_0 = l_0
		self.flatten = flatten
		self.requires_grad = requires_grad

	def explain(self, model, x, label):
		# if (not x.requires_grad):
		# 	x = Variable(x, requires_grad=True)
		logits = model(x)
		logits = logits.sum(dim=0, keepdim=True)
		max_logit = logits[0][label]
		x_grad, = torch.autograd.grad(max_logit.sum(), x, create_graph=self.requires_grad)
		x_grad = torch.abs(x_grad)
		if (self.flatten != 0):
			maxes = torch.topk(x_grad.reshape(x_grad.shape[0],-1),self.flatten,dim=1)[0].min(dim=1)[0]
			x_grad = (torch.where(x_grad.reshape(x_grad.shape[0],-1).t() < maxes, x_grad.reshape(x_grad.shape[0],-1).t(), maxes)/maxes).t().reshape(x_grad.shape)
		else:
			x_grad = (x_grad.reshape(x_grad.shape[0],-1).t()/x_grad.reshape(x_grad.shape[0],-1).max(dim=1)[0]).t().reshape(x_grad.shape)
		x_sparsified = x_grad.reshape(
			x_grad.shape[0],-1).scatter(
				1,
				torch.topk(x_grad.reshape(x_grad.shape[0],-1),x_grad[0].nelement()-self.l_0,largest=False,dim=1)[1],
				0).reshape(x_grad.shape)

		return x_sparsified

class SparsifiedSmoothGradExplainer(Explainer):
	def __init__(self, t=.1, flatten=.01, sigma=.1, n_samples=256, batch_size=128, requires_grad=False):
		self.samples = n_samples
		self.batchsize = batch_size
		self.sigma = sigma
		self.tao = t
		self.flatten = flatten
		self.requires_grad = requires_grad

	def set_requires_grad(self, requires=True):
		self.requires_grad = requires
	
	def explain(self, net, x, label):
		batches = self.samples / self.batchsize
		grads = 0
		sz = 1 * x.shape[1] * x.shape[2] * x.shape[3]
		for batch_cnt in range(int(batches)):
			noise_level = self.sigma * (torch.max(x) - torch.min(x))
			explainer = L0Explainer(int(self.tao*sz), flatten=int(self.flatten*sz), requires_grad=self.requires_grad)

			x_noise_stack = torch.vstack(tensors=[x + (torch.randn(x.shape).cuda() * noise_level) for i in range(self.batchsize)])
			x_grad = explainer.explain(net, x_noise_stack, label)

			x_grad = torch.sum(x_grad, dim=0, keepdim=True)
			grads += x_grad
			del x_grad

		avg_grad = grads / self.samples
		return avg_grad.abs()

class PureGhorbaniAttacker(Attacker):
	def __init__(self, l2, explainer, iterations, stepsize, k):
		self.explainer = explainer
		self.iterations = iterations
		self.l2 = l2
		self.stepsize = stepsize
		self.k = k
	def obtain_gradient(self, model, x, target_mask, label):
		x = Variable(x, requires_grad = True)
		raw_saliency = self.explainer.explain(model, x, label)

		raw_saliency = raw_saliency.abs()
		flattened_raw_saliency = raw_saliency.reshape(raw_saliency.shape[0], -1)
		flattened_normalized_saliency = (flattened_raw_saliency.t()/flattened_raw_saliency.sum(dim=1)).t()
		target_function_itemized = (flattened_normalized_saliency*target_mask).sum(dim=1)
		target_function = target_function_itemized.sum()
		
		ret, = torch.autograd.grad(target_function, x)

		return ret, target_function_itemized

	def explain(self, model, x, label, classifier_model=None):
		if (classifier_model == None):
			classifier_model = model
		y_org = label
		orig_grad = torch.abs(self.explainer.explain(model, x, label))
		sz = x[0].nelement()
		top_k_mask = get_top_k_mask(orig_grad,self.k).reshape(x.shape[0],-1)
		x_curr = x.clone()
		best_scores = math.inf * torch.ones(x.shape[0]).type(x.type()).cuda()
		best_x = x.clone()
		fail_mask =  torch.ones(x.shape[0]).type('torch.ByteTensor').cuda()
		for i in range(self.iterations):
			current_x_valid = classifier_model(Variable(x_curr)).max(1)[1].data == y_org

			grad, val = self.obtain_gradient(model, x_curr, top_k_mask, label)
			# print(val)
			best_mask = (current_x_valid * (val < best_scores)).type('torch.ByteTensor').cuda()
			fail_mask = fail_mask * (1-best_mask)
			best_scores[best_mask.nonzero()] = val[best_mask.nonzero()]
			#print(best_scores.sum())
			best_x[best_mask.nonzero()] = x_curr[best_mask.nonzero()]

			x_curr = x_curr.clone() - self.stepsize*torch.sign(grad)
			normalize_if_necessary(x_curr, x, self.l2)
		return best_x

class SimpleGaussianAttacker(Attacker):
	def __init__(self, sigma):
		self.sigma = sigma
	
	def explain(self, x):
		noise = self.sigma * (x.max() - x.min()) * torch.randn(x.shape).cuda()
		noise = Variable(noise, requires_grad=True)
		return x + noise
