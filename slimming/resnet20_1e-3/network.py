import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine 
		)
import megskull.opr.all as O
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith
from megskull.network import NetworkVisitor

global idx
idx = 0
global lis_k
lis_k = []

def bn_relu_conv(inp, ker_shape, stride, padding, out_chl, has_relu, has_bn, has_conv = True):
	global idx
	global lis_k
	idx += 1
	if has_bn:
		l1 = BN("bn{}".format(idx), inp, eps = 1e-9)
		l1 = ElementwiseAffine("bnaff{}".format(idx), l1, shared_in_channels = False, k = C(0.5), b = C(0))
		assert l1.inputs[1].name == "bnaff{}".format(idx) + ":k"
		lis_k.append(O.Abs(l1.inputs[1]))
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
		#W = G(mean = 0, std = (1 / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		#b = C(0),
		nonlinearity = Identity()
		)
	
	return l3


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
	l2 = BN("bn{}".format(idx), l1, eps = 1e-9)
	l2 = ElementwiseAffine("bnaff{}".format(idx), l2, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		l2 = arith.ReLU(l2)
	return l2

def res_layer(inp, chl):
	pre = inp
	inp = bn_relu_conv(inp, 3, 1, 1, chl, True, True)
	inp = bn_relu_conv(inp, 3, 1, 1, chl, True, True)
	inp = inp + pre
	return inp

def res_block(inp, chl, n):
	stride = 2
	if chl == 16:
		stride = 1
	pre = inp
	inp = bn_relu_conv(inp, 3, stride, 1, chl, True, True)
	inp = bn_relu_conv(inp, 3, 1, 1, chl, True, True)
	inp = inp + bn_relu_conv(pre, 1, stride, 0, chl, True, True)
	inp = arith.ReLU(inp)
	
	for i in range(n - 1):
		inp = res_layer(inp, chl)
	
	return inp

def make_network(minibatch_size = 128, debug = False):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size), dtype = np.float32)
	label = DataProvider("label", shape = (minibatch_size, ), dtype = np.int32)

	lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)

	n = 3
	lis = [16, 32, 64]
	for i in lis:
		lay = res_block(lay, i, n)
	
	#global average pooling
	#feature = lay.mean(axis = 2).mean(axis = 2)
	feature = Pooling2D("pooling", lay, window = 8, stride = 8, padding = 0, mode = "AVERAGE")
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		#W = G(mean = 0, std = (1 / 64)**0.5),
		#b = C(0),
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred] + lis_k)
	network.loss_var = CrossEntropyLoss(pred, label)
	lamb = 1e-3
	for i in lis_k:
		network.loss_var += lamb * i.sum()
	
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
