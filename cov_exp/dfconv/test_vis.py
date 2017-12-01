from megskull.graph import FpropEnv
from meghair.utils.io import load_network
from megskull.graph import Function
import numpy as np
import cv2
from train import MyMomentum

net = load_network(open("./data/dfconv.data", "rb"))
#test_func = Function().compile(net.outputs[0])

from megskull.network import NetworkVisitor
visitor = NetworkVisitor(net.loss_var)
offsets = []
locs = []
for i in visitor.all_oprs:
	print(i, i.name)
	if "Astype" in i.name:
		locs.append(i)
	if i.name == "conv2offsetx":
		offsets.append(i)
		print("A")
print(len(locs))
locs = locs[::4]
outs = [net.outputs[0]] + locs
test_func = Function().compile(outs)
outs1 = offsets
offs_func = Function().compile(outs1)

def load_data(name):
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = "bytes")
	return dic

dic = load_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch")
data = dic[b'data']
label = dic[b'labels']

data = data[0:128]
data = data.astype(np.float32)
imgs = data.copy()
label = label[0:128]
import pickle
with open("meanstd.data", "rb") as f:
	mean, std = pickle.load(f)
data = (data - mean) / std
data = np.resize(data, (128, 3, 32, 32))
data = data.astype(np.float32)
imgs = np.resize(imgs, (128, 3, 32, 32))
"""
import cv2
for i in range(10):
	img = data[i].transpose(1, 2, 0)
	img = img[:,::-1,:]
	cv2.imshow('x', img)
	cv2.waitKey(0)
"""
#data = data.astype(np.float32)
#data = (data - 128) / 256

outputs = test_func(data = data)
outputs1 = offs_func(data = data)
for i in range(128):
	nex = []
	cur = [32 * 16 + 15]
	for it in reversed(range(1, len(outputs))):
		for j in cur:
			x = (j // 32) * 3 + 1
			y = (j % 32) * 3 + 1
			#print(it, x, y)
			for dx in [-1, 0, 1]:
				for dy in [-1, 0, 1]:
					nx, ny = x + dx, y + dy
					nex.append(outputs[it][i][0][nx * 32 * 3 + ny][2])
		cur = nex
		nex = []
	for it in range(len(outputs1)):
		print(outs1[it].name)
		print(outs1[it].partial_shape)
		print(outputs1[it][i][0])
		print(np.sum(outputs1[it][i][0]))
	img = imgs[i].transpose(1, 2, 0).astype(np.uint8)
	mask = np.zeros((32, 32))
	for j in cur:
		x = (j // 32)
		y = j % 32
		print(x, y)
		mask[x][y] = 1
		img[x][y] = (255, 0, 0)
	print(mask[10:20, 10:20])
	#cv2.imshow('x', img)
	#cv2.waitKey(0)
	input()
	

"""
pred = np.concatenate(pred_lis, axis = 0)
print(np.array(pred).shape)
score = np.max(np.array(pred), axis = 1)
print(np.argmin(score))
pred = np.argmax(np.array(pred), axis = 1)
label = np.array(label[:len(pred)])
acc = (np.array(pred) == np.array(label)).mean()
print((acc))
"""
