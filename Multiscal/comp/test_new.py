from megskull.graph import FpropEnv
from meghair.utils.io import load_network
from megskull.graph import Function
import numpy as np
import cv2

net = load_network(open("./data/comp.data", "rb"))
test_func = Function().compile(net.outputs[0])

from megskull.network import NetworkVisitor
visitor = NetworkVisitor(net.loss_var)

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

def work(batch):
	nbatch = []
	for img in batch:
		imgs = [img]                                             
		patch_size = 32
		l = int(patch_size * 0.8)                                
		s = [[0, l], [patch_size - l, patch_size]]               
		for i in range(4):
			xx = s[i % 2]
			yy = s[i // 2]                                       
			for j in range(3):                                   
				fmap = img[j]                                    
				fmap = fmap[xx[0]:xx[1], yy[0]:yy[1]]            
				fmap = cv2.resize(fmap, (patch_size, patch_size))
				imgs.append([fmap])                              
		img = np.concatenate(imgs, axis = 0)
		nbatch.append(img)
	return nbatch

batch_size = 128
data = work(data)
pred = test_func(data = data)
"""
pred_lis = []
for i in range(data.shape[0] // batch_size):
	pred = test_func(data = data[i*batch_size : (i + 1)*batch_size])
	pred_lis.append(pred)
pred = np.concatenate(pred_lis, axis = 0)
"""
print(np.array(pred).shape)
score = np.max(np.array(pred), axis = 1)
print(np.argmin(score))
pred = np.argmax(np.array(pred), axis = 1)
label = np.array(label[:len(pred)])
acc = (np.array(pred) == np.array(label)).mean()
print((acc))
