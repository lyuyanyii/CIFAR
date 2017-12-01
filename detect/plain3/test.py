from meghair.utils.io import load_network
import numpy as np
import pickle
import cv2
from megskull.graph import Function
import matplotlib.pyplot as plt

class NN:
	def __init__(self, name):
		with open("./data/{}".format(name), "rb") as f:
			self.net = load_network(f)
		self.test_func = Function().compile([self.net.outputs[0], self.net.loss_var])
		with open("/home/liuyanyi02/CIFAR/meanstd.data", "rb") as f:
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

def load_CIFAR_data(path = "/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch"):
	with open(path, "rb") as f:
		dic = pickle.load(f, encoding = "bytes")
	data = dic[b'data']
	label = dic[b'labels']
	return data, label

def crop_test():
	data, labels = load_CIFAR_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/data_batch_1")	
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

def occ_test():
	data, labels = load_CIFAR_data()
	resnet = NN("resnet20_acc91.7")
	for i in range(10):
		img, lab = data[i], labels[i]
		img = np.array(img).reshape(3, 32, 32)
		pic = np.array(img)
		mean, std = resnet.mean, resnet.std
		img = (img - mean) / std
		scores = []
		losses = []
		for l in range(16):
			ax, ay = 15 - l, 15 - l
			bx, by = 17 + l, 17 + l
			mask = np.ones((3, 32, 32))
			mask[:, ax:bx, ay:by] = 0
			score, loss = resnet.test([img * mask], [lab], isWhite = True)
			pic1 = pic * mask
			pic1 = pic1.transpose(1, 2, 0).astype(np.uint8)
			print("score = ", score[0][lab])
			print("loss = ", loss[0])
			cv2.imshow('x', pic1)
			cv2.waitKey(0)
			scores.append(score[0][lab])
			losses.append(loss[0])
		print(scores)
		print(losses)

def linear_test():
	data, labels = load_CIFAR_data()
	resnet = NN("resnet20_acc91.7")
	clsid = 0
	for ti in range(50):
		i = np.random.randint(len(labels))
		for tj in range(1):
			j = np.random.randint(len(labels))
			"""
			while True:
				j = np.random.randint(len(labels))
				if labels[i] == labels[j]:
					break
			"""
			img1, lab1 = data[i], labels[i]
			img2, lab2 = data[j], labels[j]
			mean, std = resnet.mean, resnet.std
			img1 = np.array(img1).reshape(3, 32, 32)
			img1 = (img1 - mean) / std
			img2 = np.array(img2).reshape(3, 32, 32)
			img2 = (img2 - mean) / std
			scores1 = []
			scores2 = []
			losses = []
			for k in np.linspace(0, 1, 10, endpoint = False):
				img = img1 * (1 - k) + 0 * k
				score, loss = resnet.test([img], [lab1], isWhite = True)
				scores1.append(score[0][lab1])
				scores2.append(score[0][lab2])
				losses.append(loss[0])
			img1 = data[i]
			img1 = np.array(img1, dtype = np.uint8).reshape(3, 32, 32)
			img2 = data[j]
			img2 = np.array(img2, dtype = np.uint8).reshape(3, 32, 32)
			img1 = img1.transpose(1, 2, 0)
			img2 = img2.transpose(1, 2, 0)
			img = 0.5 * img1 + 0.5 * img2
			img = img.astype(np.uint8)
			#cv2.imshow('x', img)
			#cv2.waitKey(0)
			img_ori = img1
			#cv2.imwrite("0.5.png", img)
			#cv2.imwrite("1.png", img_ori)
			print(lab1, lab2)
			print(scores1)
			print(scores2)
			print(losses)
			input()

def mean_test():
	data, labels = load_CIFAR_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/data_batch_1")
	resnet = NN("resnet20_acc91.7")
	clsid = 0
	idx = [i for i in range(len(labels)) if labels[i] == clsid]
	imgs = [data[i] for i in idx]
	labels = [labels[i] for i in idx]
	imgs = np.array(imgs)
	img0 = imgs[0]
	imgs = imgs[1:]
	img1 = np.mean(imgs, axis = 0)
	img = img0 * 0.6 + img1 * 0.4
	img = img.reshape(3, 32, 32)
	scores, loss = resnet.test([img], [clsid])
	score = scores[0][clsid]
	loss = loss[0]
	print(score, loss)
	print(scores[0])

if __name__ == '__main__':
	#crop_test()
	#occ_test()
	#linear_test()
	mean_test()
