import os
import pickle
import argparse
from multiprocessing import Process as P
from multiprocessing import Queue, Lock
from dpflow import control, OutputPipe
import numpy as np
from meghair.utils.misc import add_rand_seed_entropy, stable_rand_seed

def augment(img, x, y, ox, oy):
	padding1 = np.zeros((3, 32, 4))
	img = np.concatenate([padding1, img, padding1], axis = 2)
	padding2 = np.zeros((3, 4, 40))
	img = np.concatenate([padding2, img, padding2], axis = 1)
	#x, y = np.random.randint(9, size = 2)
	img = img[:, x:x + 32, y : y + 32]
	ax, ay, bx, by = ox - 8, oy - 8, ox + 8, oy + 8
	ax = max(0, ax)
	ay = max(0, ay)
	bx = min(32, bx)
	by = min(32, by)
	mask = np.ones(img.shape)
	mask[:, ax:ay, bx:by] = 0
	img *= mask
	return img

def worker(data, pname, que, lock, is_train, mean, std):
	p = OutputPipe(pname, buffer_size = 20)
	if is_train:
		worker_id = int(que.get())
		que.put(worker_id + 1)
		add_rand_seed_entropy(worker_id)
		add_rand_seed_entropy(pname)
		np.random.seed(stable_rand_seed(worker))
	with control(io = [p]):
		while True:
			#np.random.set_state(que.get())
			if is_train:
				idx = np.random.randint(len(data[0]))
				x, y = np.random.randint(9, size = 2)
				ox, oy = np.random.randint(32, size = 2)
				flag = np.random.randint(2)
			else:
				idx = int(que.get())
				que.put(idx + 1)
			#que.put(np.random.get_state())
			#que.put((int(idx) + 1) % len(data[0]))
			img = np.array(data[0][int(idx)])
			img = img.astype(np.float32)
			img = (img - mean) / std
			img = np.resize(img, (3, 32, 32))
			#img = (img - 128) / 256
			if is_train:
				img = augment(img, x, y, ox, oy)
				if flag == 1:
					img = img[:,:,::-1]
					print("reversed")
			#a = msgpack.packb([img, data[1][int(idx)]], default = m.encode)
			#b = msgpack.unpackb(a, object_hook = m.decode)
			#print(np.array(b[0]).shape, b[1])
			p.put_pyobj([np.array(img), int(data[1][int(idx)])])
			print("put {} cutout-data {} successfully".format({True:"train", False:"valid"}[is_train], int(idx)))

def load_data(name):
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = 'bytes')
	return dic


parser = argparse.ArgumentParser()
parser.add_argument("t", type = int)
#parser.add_argument("m")
args = parser.parse_args()
np.random.seed(0)

if True:
	dics = []
	for i in range(1, 6):
		dics.append(load_data("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/cifar-10-batches-py/data_batch_{}".format(i)))

	data = np.array(dics[0][b'data'])
	labels = np.array(dics[0][b'labels'])
	for i in range(1, 5):
		data = np.concatenate([data, np.array(dics[i][b'data'])], axis = 0)
		labels = np.concatenate([labels, np.array(dics[i][b'labels'])], axis = 0)
	mean = np.mean(data, axis = 0)
	std = np.std(data, axis = 0)
	#p = np.arange(50000, dtype = np.int32)
	#np.random.shuffle(p)
	#data = np.array([data[i] for i in p]) 
	#labels = np.array([labels[i] for i in p])

	import pickle
	with open("meanstd.data", "wb") as f:
		pickle.dump([mean, std], f)
	train_dataset = [data, labels]

	test_dic = load_data("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/cifar-10-batches-py/test_batch")
	test_data = test_dic[b'data']
	test_label = test_dic[b'labels']
	#test_data = (test_data - mean) / std
	valid_dataset = [test_data, test_label]
	#valid_dataset = [data[49500:], labels[49500:]]
	
	lis = []
	que = Queue(1)
	que.put(0)
	lock = Lock()

	p = "lyy.CIFAR10.train.cutout"
	for i in range(args.t):
		proc = P(target = worker, args = (train_dataset, p, que, lock, True, mean, std))
		proc.start()
		lis.append(proc)
	
	que_val = Queue(1)
	que_val.put(0)
	p_val = "lyy.CIFAR10.valid"
	for i in range(args.t // 10 + 1):
		proc = P(target = worker, args = (valid_dataset, p_val, que_val, lock, False, mean, std))
		proc.start()
		lis.append(proc)
	
	for i in lis:
		i.join()

else:
	test_dic = load_data("~/CIFAR/cifar-10-batches-py/test_batch")
