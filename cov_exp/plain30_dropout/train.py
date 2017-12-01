import argparse
from meghair.train.env import TrainingEnv, Action
from megskull.opr.loss import WeightDecay

from megskull.graph import FpropEnv
import megskull
from dpflow import InputPipe, control
import time

from network import make_network
import numpy as np

#import msgpack
#import msgpack_numpy as m

minibatch_size = 128
patch_size = 32
net_name = "plain30_dropout"

def get_minibatch(p, size):
	data = []
	labels = []
	for i in range(size):
		#a = p.get()
		#(img, label) = msgpack.unpackb(a, object_hook = m.decode)
		(img, label) = p.get()
		data.append(img)
		labels.append(label)
	return {"data": np.array(data).astype(np.float32), "label":np.array(labels)}

if __name__ == '__main__':
	with TrainingEnv(name = "lyy.{}.test".format(net_name), part_count = 2) as env:
		net = make_network(minibatch_size = minibatch_size)
		preloss = net.loss_var
		net.loss_var = WeightDecay(net.loss_var, {"*conv*:W": 1e-4, "*fc*:W": 1e-4, "*bnaff*:k": 1e-4})
	
		train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
		valid_func = env.make_func_from_loss_var(net.loss_var, "val", train_state = False)
	
		lr = 0.1
		optimizer = megskull.optimizer.NesterovMomentum(lr, 0.9)
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
	
		tr_p = InputPipe("lyy.CIFAR10.train", buffer_size = 1000)
		va_p = InputPipe("lyy.CIFAR10.valid", buffer_size = 1000)
		epoch = 0
		EPOCH_NUM = 49500 // minibatch_size
		i = 0
		max_acc = 0
		ORI_IT = 64000
		BN_IT = 10000
		TOT_IT = ORI_IT + BN_IT
	
		his = []
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
						data_val = get_minibatch(va_p, 500)
						out_val = valid_func(data = data["data"], label = data["label"])
						pred = np.argmax(np.array(out_val["outputs"]), axis = 1)
						acc = (np.array(pred) == np.array(data["label"])).mean()
	
						print("epoch = {}, acc = {}, max_acc = {}".format(epoch, acc, max_acc))
						b = time.time()
						b = b + (b - a) / i * TOT_IT
						print("Expected finish time {}".format(time.asctime(time.localtime(b))))
	
						if acc > max_acc and i > ORI_IT:
							max_acc = acc
							env.save_checkpoint("{}.data.bestmodel".format(net_name))
						env.save_checkpoint("{}.data".format(net_name))
						print("**************************")
						import pickle
						with open("hisloss.data", "wb") as f:
							pickle.dump(his, f)
