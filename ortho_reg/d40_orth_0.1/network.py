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
import megskull.opr.all as O

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
		return l2, None

	l3 = Conv2D(
		"conv{}".format(idx), l2, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		nonlinearity = Identity()
		)
	w = l3.inputs[1]
	assert ":W" in w.name
	
	return l3, w

def dense_block(inp, k, l):
	lay = inp
	lis_w = []
	for i in range(l):
		cur_lay, w = bn_relu_conv(lay, 3, 1, 1, k, True, True)
		lis_w.append(w)
		lay = Concat([lay, cur_lay], axis = 1)
	return lay, lis_w

def transition(inp, i):
	l1, w = bn_relu_conv(inp, 1, 1, 0, inp.partial_shape[1], True, True, i != 2)
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
	return l2, [w]

def make_network(minibatch_size = 64):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	lay, w = bn_relu_conv(inp, 3, 1, 1, 16, False, False)
	lis_w = [w]

	k, l = 12, (40 - 4) // 3
	for i in range(3):
		#lay = transition(dense_block(lay, k, l), i)
		lay, lis_new = dense_block(lay, k, l)
		lis_w += lis_new
		lay, lis_new = transition(lay, i)
		lis_w += lis_new
	
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

	lmd = 0.1
	for w in lis_w:
		if w is None:
			continue
		print(w.partial_shape)
		w = w.reshape(w.partial_shape[0], -1).dimshuffle(1, 0)
		w = w / ((w**2).sum(axis = 0)).dimshuffle('x', 0)
		A = O.MatMul(w.dimshuffle(1, 0), w)
		network.loss_var += lmd * ((A - np.identity(A.partial_shape[0]))**2).sum()

	return network

if __name__ == '__main__':
	make_network()
