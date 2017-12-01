import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine 
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith
from megskull.network import NetworkVisitor
import megskull.opr.all as O

global idx
idx = 0

def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	l1 = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		#W = G(mean = 0, std = ((1) / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		#b = C(0),
		nonlinearity = Identity()
		)
	w = l1.inputs[1]
	assert ":W" in w.name
	l2 = BN("bn{}".format(idx), l1, eps = 1e-9)
	l2 = ElementwiseAffine("bnaff{}".format(idx), l2, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		l2 = arith.ReLU(l2)
	return l2, w

def res_layer(inp, chl):
	pre = inp
	inp, w = conv_bn(inp, 3, 1, 1, chl, True)
	inp, w = conv_bn(inp, 3, 1, 1, chl, False)
	inp = arith.ReLU(inp + pre)
	return inp, [w, w]

def res_block(inp, chl, n):
	lis_w = []
	stride = 2
	if chl == 16:
		stride = 1
	pre = inp
	inp, w = conv_bn(inp, 3, stride, 1, chl, True)
	lis_w.append(w)
	inp, w = conv_bn(inp, 3, 1, 1, chl, False)
	lis_w.append(w)
	res_path, w = conv_bn(pre, 1, stride, 0, chl, False)
	inp = inp + res_path
	lis_w.append(w)
	inp = arith.ReLU(inp)
	
	for i in range(n - 1):
		inp, lis_new = res_layer(inp, chl)
		lis_w += lis_new
	
	return inp, lis_w

def make_network(minibatch_size = 128, debug = False):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size), dtype = np.float32)
	label = DataProvider("label", shape = (minibatch_size, ), dtype = np.int32)

	lay, w = conv_bn(inp, 3, 1, 1, 16, True)
	lis_w = [w]

	n = 3
	lis = [16, 32, 64]
	for i in lis:
		lay, lis_new  = res_block(lay, i, n)
		lis_w += lis_new
	
	#global average pooling
	#feature = lay.mean(axis = 2).mean(axis = 2)
	feature = Pooling2D("pooling", lay, window = 8, stride = 8, padding = 0, mode = "AVERAGE")
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		#W = G(mean = 0, std = (1 / 64)**0.5),
		#b = C(0),
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred])
	network.loss_var = CrossEntropyLoss(pred, label)

	lmd = 1
	for w in lis_w:
		w = w.reshape(w.partial_shape[0], -1).dimshuffle(1, 0)
		w = w / ((w**2).sum(axis = 0)).dimshuffle('x', 0)
		A = O.MatMul(w.dimshuffle(1, 0), w)
		network.loss_var += lmd * ((A - np.identity(A.partial_shape[0]))**2).mean()
	
	if debug:
		visitor = NetworkVisitor(network.loss_var)
		for i in visitor.all_oprs:
			print(i)
			print(i.partial_shape)
			print("input = ", i.inputs)
			print("output = ", i.outputs)
			print()

	return network

if __name__ == "__main__":
	make_network(debug = True)
