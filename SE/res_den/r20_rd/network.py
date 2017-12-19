import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine 
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity, Sigmoid
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

def bn_relu_conv(inp, ker_shape, stride, padding, out_chl, isrelu, isbn):
	global idx
	idx += 1
	if isbn:
		inp = BN("bn{}".format(idx), inp, eps = 1e-9)
		inp = ElementwiseAffine("bnaff{}".format(idx), inp, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		inp = arith.ReLU(inp)
	inp = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		#W = G(mean = 0, std = ((1) / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		#b = C(0),
		nonlinearity = Identity()
		)
	return inp

def den_lay(inp, chl):
	out = []
	stage = 8
	for i in range(stage):
		lay = bn_relu_conv(inp, 3, 1, 1, chl // stage, True, True)
		out.append(lay)
		inp = O.Concat([inp, lay], axis = 1)
	return O.Concat(out, axis = 1)

def res_layer(inp, chl):
	pre = inp
	#inp = conv_bn(inp, 3, 1, 1, chl, True)
	#inp = conv_bn(inp, 3, 1, 1, chl, False)
	inp = den_lay(inp, chl)
	inp = den_lay(inp, chl)
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
	
	network = Network(outputs = [pred])
	network.loss_var = CrossEntropyLoss(pred, label)
	
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
