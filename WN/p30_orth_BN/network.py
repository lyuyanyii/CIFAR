import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat,
			MatMul
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

def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	l1 = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = G(mean = 0, std = ((1 + int(isrelu)) / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = Identity()
		)
	l2 = BN("bn{}".format(idx), l1, eps = 1e-9)
	l2 = ElementwiseAffine("bnaff{}".format(idx), l2, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		l2 = arith.ReLU(l2)
	return l2, l1

def make_network(minibatch_size = 128):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	#lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)
	lay, conv = conv_bn(inp, 3, 1, 1, 16, True)
	out = [conv]
	for chl in [32, 64, 128]:
		for i in range(10):
			lay, conv = conv_bn(lay, 3, 1, 1, chl, True)
			out.append(conv)
		if chl != 128:
			lay = Pooling2D("pooling{}".format(chl), lay, window = 2, mode = "MAX")

	
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
	
	network = Network(outputs = [pred] + out)
	network.loss_var = CrossEntropyLoss(pred, label)
	#conv1 = out[0]
	#print(conv1.inputs[1].partial_shape)
	lmd = 0.01
	for conv_lay in out:
		w = conv_lay.inputs[1]
		#w = w.reshape(w.partial_shape[0], -1).dimshuffle(1, 0)
		#w = w.dimshuffle(1, 0, 2, 3)
		w = w.reshape(w.partial_shape[0], -1).dimshuffle(1, 0)
		w = w / ((w**2).sum(axis = 0)).dimshuffle('x', 0)
		A = MatMul(w.dimshuffle(1, 0), w)
		network.loss_var += lmd * ((A - np.identity(A.partial_shape[0]))**2).sum()

	return network

if __name__ == '__main__':
	make_network()
