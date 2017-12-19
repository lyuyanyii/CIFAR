from meghair.utils.io import load_network
import numpy as np
import pickle
import cv2
from megskull.graph import Function
import matplotlib.pyplot as plt
from megskull.network.visitor import NetworkVisitor
import matplotlib.pyplot as plt

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
	r120 = NN("data/r120_G2.data")
	data, label = load_CIFAR_data(path = "/unsullied/sharefs/liuyanyi02/lyy/CIFAR/cifar-10-batches-py/data_batch_1")
	batch = data[:10]
	batch = batch.reshape(10, 3, 32, 32)
	label = label[:10]
	outputs = r120.test(batch, label)
	for i in outputs:
		print(i)
		input()

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
