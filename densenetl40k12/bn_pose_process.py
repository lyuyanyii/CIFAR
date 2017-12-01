from meghair.train.env import TrainingEnv
from meghair.utils.io import load_network
from megskull.opr.regularizer import BatchNormalization
from megskull.graph import Function
import numpy as np
from dpflow import InputPipe, control
from train import get_minibatch

def _get_dataset_mean_var(data, func, i):
	total_mean = None
	total_Esqr = None

	p = InputPipe("lyy.CIFAR10.train", buffer_size = 1000)
	with control(io = [p]):
		data = get_minibatch(p, 30)
		output = func(data = data["data"])
	
	total_mean = output[int(i) * 2].mean(axis = 0)
	total_Esqr = output[int(i) * 2 + 1].mean(axis = 0)

	return total_mean, total_Esqr - total_mean**2

def bn_post_process(model_file:str, save_model_file:str, data):
	with TrainingEnv(name = model_file + "bn_post_proc", part_count = 2) as env:
		net = load_network(open(model_file, "rb"))
		#loss_func = env.make_func_from_loss_var(net.loss_var, "val", train_state = False)

		bn_oprs = [opr for opr in net.loss_visitor.all_oprs if isinstance(opr, BatchNormalization)]
		bn_inputs = [opr.inputs[0] for opr in bn_oprs]

		mean_Esqr_nodes = []
		for i in bn_inputs:
			if i.partial_shape.ndim == 2:
				mean = i.mean(axis = 0).reshape((1, -1))
				mean.vflags.data_parallel_reduce_method = 'sum'
				Esqr = (i**2).mean(axis = 0).reshape((1, -1))
				Esqr.vflags.data_parallel_reduce_method = 'sum'
			if i.partial_shape.ndim == 4:
				mean = i.mean(axis = 3).mean(axis = 2).mean(axis = 0).reshape((1, -1))
				mean.vflags.data_parallel_reduce_method = 'sum'
				Esqr = (i**2).mean(axis = 3).mean(axis = 2).mean(axis = 0).reshape((1, -1))
				Esqr.vflags.data_parallel_reduce_method = 'sum'
			mean_Esqr_nodes.append(mean)
			mean_Esqr_nodes.append(Esqr)

		func = Function().compile(mean_Esqr_nodes)

		for i in range(len(bn_oprs)):
			opr = bn_oprs[i]
			layer_mean, layer_var = _get_dataset_mean_var(data, func, i)
			if layer_mean.ndim == 0:
				layer_mean = layer_mean.reshape((1, ))
			if layer_var.ndim == 0:
				layer_var = layer_var.reshape((1, ))

			state = opr.State(
				channels = layer_mean.shape[0],
				val = [layer_mean, layer_var, 1]
				)
			state.owner_opr_type = type(opr)

			opr.set_opr_state(state)

			opr.freezed = True

		env.register_checkpoint_component("network", net)
		env.save_checkpoint(save_model_file)

def load_data(name):
	return None
	"""
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = "bytes")
	data = dic[b'data']
	label = dic[b'labels']

	data = data.astype(np.float32)
	with open("meanstd.data", "rb") as f:
		mean, std = pickle.load(f)
	data = (data - mean) / std
	data = np.resize(data, (10000, 3, 32, 32))
	data = data.astype(np.float32)
	return data
	"""


bn_post_process("data/densenetl40k12.data", "densenetl40k12.data_bned", load_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/test_batch"))

