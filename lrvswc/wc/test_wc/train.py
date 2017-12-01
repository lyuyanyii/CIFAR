import argparse
from meghair.train.env import TrainingEnv, Action
from megskull.opr.loss import WeightDecay

from megskull.graph import FpropEnv
import megskull
from dpflow import InputPipe, control
import time

import numpy as np
from megskull.utils.meta import override
from megskull.optimizer import Momentum
from megbrain.opr import concat
import cv2

import megskull.opr.all as O
import numpy as np
from megskull.opr.helper.param_init import ConstantParamInitializer as C

from megskull.optimizer import NaiveSGD, OptimizableFunc
from megskull.network import Network
from My import MyWeightDecay
from megskull.opr.helper.elemwise_trans import ReLU, Identity

minibatch_size = 10
patch_size = 32
net_name = "test_wc"


inp = O.DataProvider("a", shape = (minibatch_size, 3))
out = O.FullyConnected(
	"fc", inp, output_dim = 3,
	W = C(1),
	nonlinearity = Identity()
	)
W = out.inputs[1]
loss = O.ZeroGrad(out.sum())
network = Network(outputs = [loss])
network.loss_var = loss

"""
func = OptimizableFunc.make_from_loss_var(loss)
NaiveSGD(1)(func)
func.compile(loss)

print(func())
print(np.array(a.eval(), dtype = np.float32))

loss.Mul_Wc(10)

print(func())
print(np.array(a.eval()))

print(func())
"""
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	with TrainingEnv(name = "lyy.{}.test".format(net_name), part_count = 2, custom_parser = parser) as env:
		print("A")
		net = network
		preloss = net.loss_var
		net.loss_var = MyWeightDecay(net.loss_var, {"*":0.0001})

		train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
	
		NaiveSGD(1)(train_func)

		#train_func.comp_graph.share_device_memory_with(valid_func.comp_graph)
	
		train_func.compile(net.loss_var)
		
		env.register_checkpoint_component("network", net)
		env.register_checkpoint_component("opt_state", train_func.optimizer_state)
	
		data = np.ones((10, 3))
		print(train_func(a = data))
		print(W.eval())

		net.loss_var.owner_opr.Mul_Wc(10000)
		print(train_func(a = data))
		print(W.eval())

		print(train_func(a = data))
