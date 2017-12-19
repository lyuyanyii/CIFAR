import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat 
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith
from ntools.megtools.tools.info import ComplexityInfo as CInfo

global idx
idx = 0

"""
base model: 264m
"""

def bn_relu_conv(inp, ker_shape, stride, padding, out_chl, has_relu, has_bn, has_conv = True, group = None):
	global idx
	idx += 1
	if has_bn:
		l1 = BN("bn{}".format(idx), inp, eps = 1e-9)
		l1 = ElementwiseAffine("bnaff{}".format(idx), l1, shared_in_channels = False, k = C(1), b = C(0))
	else:
		l1 = inp
	
	if has_relu:
		l2 = arith.ReLU(l1)
	else:
		l2 = l1
	
	if not has_conv:
		return l2

	if group is None:
		l3 = Conv2D(
			"conv{}".format(idx), l2, kernel_shape = ker_shape, stride = stride, padding = padding,
			output_nr_channel = out_chl,
			nonlinearity = Identity()
			)
	else:
		l3 = Conv2D(
			"conv{}".format(idx), l2, kernel_shape = ker_shape, stride = stride, padding = padding,
			output_nr_channel = out_chl,
			nonlinearity = Identity(),
			group = group,
			)
	
	return l3

def dense_block(inp, k, l, j):
	lay = inp
	for i in range(l):
		idx = j * l + i + 1
		chl = lay.partial_shape[1]
		p = idx & -idx
		p = max(0, chl - p * k)
		sh_lay = lay[:, p:, :, :]
		cur_lay = bn_relu_conv(sh_lay, 3, 1, 1, k, True, True)
		lay = Concat([lay, cur_lay], axis = 1)
	k *= 2
	return lay

def transition(inp, i):
	l1 = bn_relu_conv(inp, 1, 1, 0, inp.partial_shape[1], True, True, has_conv = i != 2)
	global idx
	idx += 1
	if i != 2:
		l2 = Pooling2D(
			"Pooling{}".format(idx), l1, window = 2, mode = "AVERAGE"
			)
	else:
		l2 = Pooling2D(
			"Pooling{}".format(idx), l1, window = 8, stride = 8, mode = "AVERAGE"
			)
	return l2

def make_network(minibatch_size = 64):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	k, l = 18, (40 - 4) // 3
	lay = bn_relu_conv(inp, 3, 1, 1, k, False, False)

	for i in range(3):
		lay = transition(dense_block(lay, k, l, i), i)
	
	#global average pooling
	print(lay.partial_shape)
	feature = lay.mean(axis = 2).mean(axis = 2)
	#feature = Pooling2D("glbpoling", lay, window = 8, stride = 8, mode = "AVERAGE")
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred])
	network.loss_var = CrossEntropyLoss(pred, label)

	info = CInfo()
	info.get_complexity(network.outputs).as_table().show()

	return network

if __name__ == '__main__':
	make_network()
