from meghair.utils.io import load_network
from megskull.graph import Function
import numpy as np
from numpy.linalg import svd

net = load_network(open("./data/plain30.data", "rb"))
func = Function().compile(net.outputs)

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


idx = np.random.randint(data.shape[0], size = 3000)
data = [data[i] for i in idx]

def pca_ener(A):
	A = A.T
	A = A - np.mean(A, axis = 0)
	CovM = np.dot(A.T, A) / A.shape[0]
	U, s, V = svd(CovM)
	s = np.cumsum(s)
	return s

out = func(data = data)[1:]
ener = []
for conv_eval in out:
	conv_eval = conv_eval.transpose(1, 0, 2, 3)
	ener.append(pca_ener(conv_eval.reshape(conv_eval.shape[0], -1)))

with open("pca_ener.data", "wb") as f:
	pickle.dump(ener, f)
