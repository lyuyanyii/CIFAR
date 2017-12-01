import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat 
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider, ConstProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith
from scipy.stats import ortho_group

global idx
idx = 0

def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu, mode = None):
	global idx
	idx += 1
	print(inp.partial_shape, ker_shape, out_chl)
	if ker_shape == 1:
		W = ortho_group.rvs(out_chl)
		W = W[:, :inp.partial_shape[1]]
		W = W.reshape(W.shape[0], W.shape[1], 1, 1)
		W = ConstProvider(W)
		b = ConstProvider(np.zeros(out_chl))
	else:
		W = G(mean = 0, std = ((1 + int(isrelu)) / (ker_shape**2 * inp.partial_shape[1]))**0.5)
		b = C(0)
	l1 = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		group = mode,
		W = W,
		b = b,
		nonlinearity = Identity()
		)
	l2 = BN("bn{}".format(idx), l1, eps = 1e-9)
	l2 = ElementwiseAffine("bnaff{}".format(idx), l2, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		l2 = arith.ReLU(l2)
	return l2, l1

def xcep_layer(inp, chl):
	inp, conv1 = conv_bn(inp, 3, 1, 1, chl, True, mode = 'chan')
	print(inp.partial_shape)
	inp, conv2 = conv_bn(inp, 1, 1, 0, chl, True)
	return inp, conv1, conv2

def make_network(minibatch_size = 128):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	#lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)
	lay, conv = conv_bn(inp, 3, 1, 1, 16, True)
	out = [conv]
	for chl in [32 * 3, 64 * 3, 128 * 3]:
		for i in range(10):
			lay, conv1, conv2 = xcep_layer(lay, chl)
			out.append(conv1)
			out.append(conv2)
		if chl != 128 * 3:
			lay = Pooling2D("pooling{}".format(chl), lay, window = 2, mode = "MAX")

	
	#global average pooling
	print(lay.partial_shape)
	feature = lay.mean(axis = 2).mean(axis = 2)
	#feature = Pooling2D("glbpoling", lay, window = 8, stride = 8, mode = "AVERAGE")
	W = ortho_group.rvs(feature.partial_shape[1])
	W = W[:, :10]
	W = ConstProvider(W)
	b = ConstProvider(np.zeros((10, )))
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		W = W,
		b = b,
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred] + out)
	network.loss_var = CrossEntropyLoss(pred, label)
	return network

if __name__ == '__main__':
	make_network()