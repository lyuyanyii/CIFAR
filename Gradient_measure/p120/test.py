
from meghair.utils.io import load_network
import numpy as np
import pickle
import cv2
from megskull.graph import Function
import matplotlib.pyplot as plt
from megskull.network.visitor import NetworkVisitor
import matplotlib.pyplot as plt
import megskull.opr.all as O

class NN:
	def __init__(self, name):
		with open("{}".format(name), "rb") as f:
			self.net = load_network(f)
		self.test_func = Function().compile([self.net.outputs[0], self.net.loss_var])
		with open("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/meanstd.data", "rb") as f:
			self.mean, self.std = pickle.load(f)
			self.mean = np.array(self.mean).reshape(3, 32, 32)
			self.std = np.array(self.std).reshape(3, 32, 32)
	
	def test(self, batch, labels, isWhite = False):
		batch = np.array(batch)
		#checking images
		if batch.shape[-1] == 3:
			batch = batch.transpose(0, 3, 1, 2)
		if not isWhite:
			batch = (batch - self.mean) / self.std
		batch = np.array(batch, dtype = np.float32)
		outputs = self.test_func(data = batch, label = labels)
		return outputs

def load_CIFAR_data(path = "/unsullied/sharefs/liuyanyi02/lyy/CIFAR/cifar-10-batches-py/test_batch"):
	with open(path, "rb") as f:
		dic = pickle.load(f, encoding = "bytes")
	data = dic[b'data']
	label = dic[b'labels']
	return data, label

def test():
	data, labels = load_CIFAR_data()
	p120 = NN("data/p120.data")
	net = p120.net
	loss = net.loss_var
	visitor = NetworkVisitor(loss)
	inp = []
	for i in visitor.all_oprs:
		if "data" in i.name:
			inp.append(i)
		if "conv" in i.name and ":" not in i.name:
			inp.append(i)
	print(inp)
	grad = []
	out = []
	for i in inp:
		grad.append(O.Grad(loss, i))
		out.append(i)
	F = Function()
	F._env.flags.train_batch_normalization = True
	func = F.compile(grad)
	F = Function()
	F._env.flags.train_batch_normalization = True
	func1 = F.compile(out)
	
	batch = data[:128]
	batch = batch.reshape(128, 3, 32, 32)
	mean, std = p120.mean, p120.std
	batch = (batch - mean) / std
	label = labels[:128]
	grad_out = func(data = batch, label = label)
	lay_out = func1(data = batch, label = label)
	idx = 0
	grad_list = []
	for i, j in zip(grad_out, lay_out):
		print(i.shape, idx)
		idx += 1
		f = i.flatten()
		print("grad")
		print(f)
		print(np.mean(f), np.std(f))
		grad_list.append(np.std(f))
		print("val")
		h = j.flatten()
		print(h)
		print(np.mean(h), np.std(j))
	pickle.dump(grad_list, open("p120_norelu_grad.data", "wb"))
	"""
	print(grad_out[-1])
	print(grad_out.shape)
	grad_out = grad_out.flatten()
	print(np.mean(grad_out))
	print(np.std(grad_out))
	"""

def crop_test():
	data, labels = load_CIFAR_data("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/cifar-10-batches-py/data_batch_1")	
	P3 = NN("resnet20_acc91.7")
	for i in range(10):
		img, lab = data[i], labels[i]
		img = np.array(img, dtype = np.uint8).reshape(3, 32, 32)
		img = img.transpose(1, 2, 0)
		scores, losses = [], []
		plt.figure(i)
		for j in range(100):
			l = np.random.randint(16, 26)
			ax, ay = np.random.randint(32 - l + 1, size = (2, ))
			bx, by = ax + l, ay + l
			img_c = img[ax:bx, ay:by, :]
			img_c = cv2.resize(img_c, (32, 32))
			score, loss = P3.test([img_c], [lab])
			score = score[0][lab]
			scores.append(score)
			losses.append(loss[0])
		scores = sorted(scores, reverse = True)
		losses = sorted(losses)
		plt.plot(range(len(scores)), scores, 'g')
		plt.plot(range(len(losses)), losses, 'b')
		print(scores)
		print(losses)
		print("****************************************")
		input()
		#cv2.imshow('x', img)
		#cv2.waitKey(0)
		#plt.show()
def orth_test():
	#p30 = NN("../plain30/data/plain30.data")
	#p30 = NN("../../resnet20/data/resnet20_acc91.7")
	#p30 = NN("../plain30_orth/data/plain30_orth.data")
	p30 = NN("../../densenetl100k24/data/densenetl100k24.data")
	#p30 = NN("../../lrvswc/wc/data/lr.data")
	net = p30.net
	loss = net.loss_var
	visitor = NetworkVisitor(loss)
	"""
	for i in visitor.all_oprs:
		print(i)
	"""
	for i in visitor.all_oprs:
		if ":W" in i.name:
			W = i.eval()
			print(W.shape)
			if W.ndim != 4:
				continue
			#W = W.reshape(W.shape[0], -1)
			W = W.sum(axis = 3).sum(axis = 2)
			print(W.shape)
			#plt.plot(range(len(W[:, 0])), abs(W[:, 1]))
			W = np.array(W)
			W = W.T
			W = W / ((W**2).sum(axis = 0)**0.5)
			A = np.dot(W.T, W)
			print(list(np.round(A * 1000).astype(np.int32) / 1000))
			I = np.identity(A.shape[0])
			print(np.round(((A - I)**2).mean(), decimals = 5))
			input()
			#with open("p30wcW.data", "wb") as f:
			#	pickle.dump(W, f)
			#plt.show()
def trans_test():
	p30 = NN("/home/liuyanyi02/CIFAR/slimming/resnet20/data/slm_res20.data")
	net = p30.net
	loss = net.loss_var
	visitor = NetworkVisitor(loss)
	W0 = None
	for i in visitor.all_oprs:
		if ":W" in i.name:
			W1 = i.eval()
			W1 = W1.sum(axis = 3).sum(axis = 2)
			#W1 = W1 / ((W1**2).sum(axis = 0))
			W1 = W1.T
			if W0 is None:
				W0 = W1
				continue
			if W0.shape != W1.shape:
				W0 = W1
				continue
			A = np.dot(W0, W1)
			print(np.round(A * 1000).astype(np.int32) / 1000)
			I = np.identity(A.shape[0])
			print(i)
			print(((A - I)**2).mean())
			input()

def orth_test1():
	p30 = NN("../plain30_xcep/data/plain30_xcep.data")
	net = p30.net
	visitor = NetworkVisitor(net.loss_var)
	for i in visitor.all_oprs:
		print(i)
		print(i.partial_shape)
		"""
		if i.partial_shape[2:] == (1, 1):
			print(i)
			W = i.eval()
			W = W.reshape(W.shape[0:2])
			print(W)
			W = W / ((W**2).sum(axis = 0)**0.5)
			A = np.dot(W.T, W)
			print("*********")
			print(A)
			print("*********")
			I = np.identity(A.shape[0])
			print(((A - I)**2).mean())
			input()
		"""

if __name__ == '__main__':
	#crop_test()
	#occ_test()
	#linear_test()
	#mean_test()
	#orth_test()
	#orth_test1()
	#trans_test()
	test()
