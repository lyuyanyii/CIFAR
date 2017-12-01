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

class MyMomentum(Momentum):
	@override(Momentum)
	def get_gradient(self, param):
		grad = super().get_gradient(param)
		if ("offset" in param.name):
			#grad /= int(param.name[7:9]) * 10
			"""
			ndim = grad.shape[1] // 2
			gradx = grad[:, :ndim]
			grady = grad[:, ndim:]
			l2 = (gradx**2 + grady**2)**0.5
			gradx /= l2
			grady /= l2
			grad = concat([gradx, grady], axis = 1)
			"""
		return grad

#import msgpack
#import msgpack_numpy as m

minibatch_size = 128
patch_size = 32
net_name = "comp"

def get_minibatch(p, size):
	data = []
	labels = []
	for i in range(size):
		#a = p.get()
		#(img, label) = msgpack.unpackb(a, object_hook = m.decode)
		(img, label) = p.get()
		"""
		imgs = [img]
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
		imgs = np.concatenate(imgs, axis = 0)
		"""
		data.append(img)
		labels.append(label)
	return {"data": np.array(data).astype(np.float32), "label":np.array(labels)}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with TrainingEnv(name = "lyy.{}.test".format(net_name), part_count = 2, custom_parser = parser) as env:
		print("A")
		net = make_network(minibatch_size = minibatch_size)
		preloss = net.loss_var
		net.loss_var = WeightDecay(net.loss_var, {"*conv*:W": 1e-4, "*fc*:W": 1e-4, "*bnaff*:k": 1e-4, "*offset*":1e-4})
	
		train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
		valid_func = env.make_func_from_loss_var(net.loss_var, "val", train_state = False)
	
		lr = 0.1
		optimizer = Momentum(lr, 0.9)
		#optimizer.learning_rate = 0.01
		optimizer(train_func)
		
		train_func.comp_graph.share_device_memory_with(valid_func.comp_graph)
	
		dic = {
			"loss": net.loss_var,
			"pre_loss": preloss,
			"outputs": net.outputs[0]
		}
		train_func.compile(dic)
		valid_func.compile(dic)
		
		env.register_checkpoint_component("network", net)
		env.register_checkpoint_component("opt_state", train_func.optimizer_state)
	
		tr_p = InputPipe("lyy.CIFAR10_Multiscal.train", buffer_size = 1000)
		va_p = InputPipe("lyy.CIFAR10_Multiscal.valid", buffer_size = 1000)
		epoch = 0
		EPOCH_NUM = 49500 // minibatch_size
		i = 0
		max_acc = 0
		ORI_IT = 64000
		BN_IT = 10000
		TOT_IT = ORI_IT + BN_IT
	
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
					if time_data > (time_train + time_data) * 0.1:
						print(time_data / (time_train + time_data))
					loss = out["pre_loss"]
					pred = np.array(out["outputs"]).argmax(axis = 1)
					acc = (pred == np.array(data["label"])).mean()
					his.append([loss, acc])
					print("minibatch = {}, loss = {}, acc = {}".format(i, loss, acc))
					#Learning Rate Adjusting
					if i == ORI_IT // 2 or i == ORI_IT // 4 * 3:
						optimizer.learning_rate /= 10
					if i == ORI_IT:
						optimizer.learning_rate = 1e-5
						env.save_checkpoint("{}.data.ORI".format(net_name))
					if i % (EPOCH_NUM) == 0:
						epoch += 1
						data_val = get_minibatch(va_p, minibatch_size)
						out_val = valid_func(data = data_val["data"], label = data_val["label"])
						pred = np.argmax(np.array(out_val["outputs"]), axis = 1)
						acc = (np.array(pred) == np.array(data_val["label"])).mean()
						his_test.append([i, acc])
	
						print("epoch = {}, acc = {}, max_acc = {}".format(epoch, acc, max_acc))
						b = time.time()
						b = b + (b - a) / i * (TOT_IT - i)
						print("Expected finish time {}".format(time.asctime(time.localtime(b))))
	
						if acc > max_acc and i > ORI_IT:
							max_acc = acc
							env.save_checkpoint("{}.data.bestmodel".format(net_name))
						env.save_checkpoint("{}.data".format(net_name))
						print("**************************")
						import pickle
						with open("hisloss.data", "wb") as f:
							pickle.dump(his, f)
						with open("histest.data", "wb") as f:
							pickle.dump(his_test, f)