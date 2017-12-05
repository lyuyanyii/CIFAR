
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

def myw_test():
	d40_MY = NN("./data/r20_MY.data")
	net = d40_MY.net
	outputs = []
	visitor = NetworkVisitor(net.loss_var)
	for i in visitor.all_oprs:
		if "fc1" in i.name and ":W" not in i.name and ":b" not in i.name:
			outputs.append(i)
	func = Function().compile(outputs)
	data, labels = load_CIFAR_data()
	batch = data[:128]
	batch = batch.reshape(128, 3, 32, 32)
	mean, std = d40_MY.mean, d40_MY.std
	batch = (batch - mean) / std
	outputs_weights = func(data = batch)
	for i in outputs_weights:
		print(i.shape)
		w = i[0]
		w = w.reshape(-1, 4, 4)
		print(w)
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
if __name__ == '__main__':
	myw_test()	
