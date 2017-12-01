import argparse
from meghair.train.env import TrainingEnv, Action
from megskull.opr.loss import WeightDecay

from megskull.graph import FpropEnv
import megskull
from dpflow import InputPipe, control
import time

from network import make_network
import numpy as np
from megskull.utils.meta import override
from megskull.optimizer import Momentum
from megbrain.opr import concat
import cv2
from megskull.graph import Function

from init import init

def load_CIFAR_data(path = "/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch"):
	with open(path, "rb") as f:
		dic = pickle.load(f, encoding = "bytes")
	data = dic[b'data']
	label = dic[b'labels']
	return data, label

class CIFAR_test:
	def __init__(self):
		self.data, self.labels = load_CIFAR_data
		with open("/home/liuyanyi02/CIFAR/meanstd.data", "rb") as f:
			self.mean, self.std = pickle.load(f)
		data = self.data
		data = np.array(data)
		data = (data - mean) / std
		data = data.reshape(data.shape[0], 3, 32, 32)
		self.data = data
		self.labels = np.array(self.labels)
	def test(val_func):
		batch_size = 1000
		lis = []
		for i in range(self.data.shape[0] // batch_size):
			pred = val_func(data = self.data[i*batch_size : (i+1)*batch_size])
			lis.append(pred)
		if self.data.shape[0] % batch_size != 0:
			i = self.data.shape[0] // batch_size
			pred = val_func(data = self.data[i*batch_size :])
			lis.append(pred)
		pred = np.concatenate(lis, axis = 0)
		pred = np.argmax(pred, axis = 1)
		acc = (pred == labels).mean()
		return acc

minibatch_size = 128
patch_size = 32
net_name = "p30WN"
path = "/unsullied/sharefs/liuyanyi02/LYY/data/"

def get_minibatch(p, size):
	data = []
	labels = []
	for i in range(size):
		(img, label) = p.get()
		data.append(img)
		labels.append(label)
	return {"data": np.array(data).astype(np.float32), "label":np.array(labels)}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with TrainingEnv(name = "lyy.{}.test".format(net_name), part_count = 2, custom_parser = parser) as env:
		net = make_network(minibatch_size = minibatch_size)
		preloss = net.loss_var
		net.loss_var = WeightDecay(net.loss_var, {"*conv*:W": 1e-4, "*fc*:W": 1e-4, "*bnaff*:k": 1e-4, "*offset*":1e-4})
		
		tr_p = InputPipe("lyy.CIFAR10.train", buffer_size = 1000)
		with control(io = [tr_p]):
			net = init(net, get_minibatch(tr_p, minibatch_size))

		train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
	
		lr = 0.1
		optimizer = Momentum(lr, 0.9)
		optimizer(train_func)
		
	
		dic = {
			"loss": net.loss_var,
			"pre_loss": preloss,
			"outputs": net.outputs[0]
		}
		train_func.compile(dic)
		valid_func = Function().compile(net.outputs[0])
		
		env.register_checkpoint_component("network", net)
		env.register_checkpoint_component("opt_state", train_func.optimizer_state)
	
		va_p = InputPipe("lyy.CIFAR10.valid", buffer_size = 1000)
		epoch = 0
		EPOCH_NUM = 50000 // minibatch_size
		i = 0
		max_acc = 0
		ORI_IT = 64000
		BN_IT = 10000
		TOT_IT = ORI_IT + BN_IT

		C = CIFAR_test()
	
		his = []
		his_test = []
		import time
		with control(io = [tr_p]):
			with control(io = [va_p]):
		
				a = time.time()
				while i <= TOT_IT:
					i += 1
	
					token1 = time.time()
					data = get_minibatch(tr_p, minibatch_size)
					time_data = time.time() - token1
					
					token2 = time.time()
					out = train_func(data = data['data'], label = data["label"])
					time_train = time.time() - token2
					if time_data > (time_train + time_data) * 0.2:
						print("Loading data may spends too much time {}".format(time_data / (time_train + time_data)))
					loss = out["pre_loss"]
					pred = np.array(out["outputs"]).argmax(axis = 1)
					acc = (pred == np.array(data["label"])).mean()
					his.append([loss, acc])
					print("Minibatch = {}, Loss = {}, Acc = {}".format(i, loss, acc))
					#Learning Rate Adjusting
					if i == ORI_IT // 2 or i == ORI_IT // 4 * 3:
						optimizer.learning_rate /= 10
					if i == ORI_IT:
						optimizer.learning_rate = 1e-5
					if i % (EPOCH_NUM) == 0:
						epoch += 1
						"""
						data_val = get_minibatch(va_p, minibatch_size)
						out_val = valid_func(data = data_val["data"], label = data_val["label"])
						pred = np.argmax(np.array(out_val["outputs"]), axis = 1)
						acc = (np.array(pred) == np.array(data_val["label"])).mean()
						"""
						acc = C.test(valid_func)
						his_test.append([i, acc])
	
						print("Epoch = {}, Acc = {}, Max_acc = {}".format(epoch, acc, max_acc))
						b = time.time()
						b = b + (b - a) / i * (TOT_IT - i)
						print("Expected finish time {}".format(time.asctime(time.localtime(b))))
	
						if acc > max_acc and i > ORI_IT:
							max_acc = acc
						env.save_checkpoint(path + "{}.data".format(net_name))
						print("**************************")
						import pickle
						with open("hisloss.data", "wb") as f:
							pickle.dump(his, f)
						with open("histest.data", "wb") as f:
							pickle.dump(his_test, f)
