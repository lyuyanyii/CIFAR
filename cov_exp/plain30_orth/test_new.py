from megskull.graph import FpropEnv
from meghair.utils.io import load_network
from megskull.graph import Function
import numpy as np

net = load_network(open("./data/plain30_orth.data", "rb"))
test_func = Function().compile(net.outputs[0])

def load_data(name):
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = "bytes")
	return dic

dic = load_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch")
data = dic[b'data']
label = dic[b'labels']

data = data.astype(np.float32)
import pickle
with open("meanstd.data", "rb") as f:
	mean, std = pickle.load(f)
data = (data - mean) / std
data = np.resize(data, (10000, 3, 32, 32))
data = data.astype(np.float32)
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

pred = test_func(data = data)
print(np.array(pred).shape)
score = np.max(np.array(pred), axis = 1)
print(np.argmin(score))
pred = np.argmax(np.array(pred), axis = 1)
acc = (np.array(pred) == np.array(label)).mean()
print((acc))

