import sys
sys.path.append("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/latest_tools")
from th import TensorboardHelper as TB

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
import pickle

"""
class MyMomentum(Momentum):
	@override(Momentum)
	def get_gradient(self, param):
		grad = super().get_gradient(param)
		if ("offset" in param.name):
			return grad
"""
def load_CIFAR_data(path = "/unsullied/sharefs/liuyanyi02/lyy/CIFAR/cifar-10-batches-py/test_batch"):
	with open(path, "rb") as f:
		dic = pickle.load(f, encoding = "bytes")
	data = dic[b'data']
	label = dic[b'labels']
	return data, label

class CIFAR_test:
	def __init__(self):
		self.data, self.labels = load_CIFAR_data()
		with open("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/meanstd.data", "rb") as f:
			self.mean, self.std = pickle.load(f)
		data = self.data
		data = np.array(data)
		data = (data - self.mean) / self.std
		data = data.reshape(data.shape[0], 3, 32, 32)
		self.data = data.astype(np.float32)
		self.labels = np.array(self.labels)
	def test(self, val_func):
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
		acc = (pred == self.labels).mean()
		return acc

minibatch_size = 128
patch_size = 32
net_name = "d40_orth"
path = ""

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
	tb = TB("tbdata/")

	with TrainingEnv(name = "lyy.{}.test".format(net_name), part_count = 2, custom_parser = parser) as env:
		net = make_network(minibatch_size = minibatch_size)
		preloss = net.loss_var
		net.loss_var = WeightDecay(net.loss_var, {"*conv*:W": 1e-4, "*fc*:W": 1e-4, "*bnaff*:k": 1e-4, "*offset*":1e-4})

		train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
	
		lr = 0.1
		optimizer = Momentum(lr, 0.9)
		optimizer(train_func)
		
		#train_func.comp_graph.share_device_memory_with(valid_func.comp_graph)
	
		dic = {
			"loss": net.loss_var,
			"pre_loss": preloss,
			"outputs": net.outputs[0]
		}
		train_func.compile(dic)
		valid_func = Function().compile(net.outputs[0])
		
		env.register_checkpoint_component("network", net)
		env.register_checkpoint_component("opt_state", train_func.optimizer_state)
	
		tr_p = InputPipe("lyy.CIFAR10.train", buffer_size = 1000)
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
					tb.tick()
	
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
					tb.add_scalar("loss", loss)
					tb.add_scalar("traing_acc", acc)
					print("Minibatch = {}, Loss = {}, Acc = {}".format(i, loss, acc))
					#Learning Rate Adjusting
					if i == ORI_IT // 2 or i == ORI_IT // 4 * 3:
						optimizer.learning_rate /= 10
					if i == ORI_IT:
						optimizer.learning_rate = 1e-5
					if i % (EPOCH_NUM) == 0:
						epoch += 1
						acc = C.test(valid_func)
						his_test.append([i, acc])
	
						print("Epoch = {}, Acc = {}, Max_acc = {}".format(epoch, acc, max_acc))
						b = time.time()
						b = b + (b - a) / i * (TOT_IT - i)
						print("Expected finish time {}".format(time.asctime(time.localtime(b))))
	
						tb.add_scalar("test_acc", acc)
						if acc > max_acc and i > ORI_IT:
							max_acc = acc
						env.save_checkpoint(path + "{}.data".format(net_name))
						print("**************************")
						import pickle
						with open("hisloss.data", "wb") as f:
							pickle.dump(his, f)
						with open("histest.data", "wb") as f:
							pickle.dump(his_test, f)
						tb.flush()

