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

global idx
idx = 0

"""
def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	l1 = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = G(mean = 0, std = ((1 + int(isrelu)) / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = {True:ReLU(), False:Identity()}[isrelu]
		)
	l2 = BN("bn{}".format(idx), l1, eps = 1e-9)
	l2 = ElementwiseAffine("bnaff{}".format(idx), l2, shared_in_channels = False, k = C(1), b = C(0))
	return l2
"""

def bn_relu_conv(inp, ker_shape, stride, padding, out_chl, has_relu, has_bn, has_conv = True):
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

	l3 = Conv2D(
		"conv{}".format(idx), l2, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = G(mean = 0, std = (1 / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		b = C(0),
		nonlinearity = Identity()
		)
	
	return l3

def dense_block(inp, k, l):
	lay = inp
	for i in range(l):
		cur_lay = bn_relu_conv(lay, 3, 1, 1, k, True, True)
		lay = Concat([lay, cur_lay], axis = 1)
	return lay

def transition(inp, i):
	l1 = bn_relu_conv(inp, 1, 1, 0, inp.partial_shape[1], True, True, i != 2)
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


"""
def res_layer(inp, chl):
	pre = inp
	inp = conv_bn(inp, 3, 1, 1, chl, True)
	inp = conv_bn(inp, 3, 1, 1, chl, False)
	inp = arith.ReLU(inp + pre)
	return inp

def res_block(inp, chl, n):
	stride = 2
	if chl == 16:
		stride = 1
	pre = inp
	inp = conv_bn(inp, 3, stride, 1, chl, True)
	inp = conv_bn(inp, 3, 1, 1, chl, False)
	inp = inp + conv_bn(pre, 1, stride, 0, chl, False)
	inp = arith.ReLU(inp)
	
	for i in range(n - 1):
		inp = res_layer(inp, chl)
	
	return inp
"""

def make_network(minibatch_size = 64):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)

	k, l = 12, (40 - 4) // 3
	for i in range(3):
		lay = transition(dense_block(lay, k, l), i)
	
	#global average pooling
	print(lay.partial_shape)
	feature = lay.mean(axis = 2).mean(axis = 2)
	#feature = Pooling2D("glbpoling", lay, window = 8, stride = 8, mode = "AVERAGE")
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		W = G(mean = 0, std = (1 / feature.partial_shape[1])**0.5),
		b = C(0),
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred])
	network.loss_var = CrossEntropyLoss(pred, label)
	return network

if __name__ == '__main__':
	make_network()
