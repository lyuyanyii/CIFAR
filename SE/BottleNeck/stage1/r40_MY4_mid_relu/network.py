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

def res_layer(inp, chl, stride = 1, proj = False):
	pre = inp
	inp = conv_bn(inp, 1, stride, 0, chl // 4, True)
	name = inp.name
	#Global Average Pooling
	SE = inp.mean(axis = 3).mean(axis = 2)
	sum_lay = 0
	out_lay = 0
	width = 4
	lay = FullyConnected(
		"fc0({})".format(name), SE, output_dim = chl // 4,
		nonlinearity = ReLU()
		)
	#fc1
	lay = FullyConnected(
		"fc1({})".format(name), lay, output_dim = chl // 4 * width,
		nonlinearity = Identity()
		)
	lay = lay.reshape(inp.shape[0], chl // 4, width)
	lay = Softmax("softmax({})".format(name), lay, axis = 2)
	for i in range(width):
		if i == 0:
			inp_lay = inp
		else:
			inp_lay = O.Concat([inp[:, width:, :, :], inp[:, :width, :, :]], axis = 1)
		inp_lay = inp_lay * lay[:, :, i].dimshuffle(0, 1, 'x', 'x')
	inp = O.ReLU(inp_lay)
	inp = conv_bn(inp, 3, 1, 1, chl // 4, True)
	inp = conv_bn(inp, 1, 1, 0, chl, False)
	if proj:
		pre = conv_bn(pre, 1, stride, 0, chl, False)
	inp = arith.ReLU(inp + pre)
	return inp

def res_block(inp, chl, i, n):
	stride = 2
	if i == 0:
		stride = 1
	inp = res_layer(inp, chl, stride = stride, proj = True)
	
	for i in range(n - 1):
		inp = res_layer(inp, chl)
	
	return inp

def make_network(minibatch_size = 128, debug = False):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size), dtype = np.float32)
	label = DataProvider("label", shape = (minibatch_size, ), dtype = np.int32)

	lay = conv_bn(inp, 3, 1, 1, 16, True)

	n = 4
	lis = [16 * 4, 32 * 4, 64 * 4]
	for i in range(len(lis)):
		lay = res_block(lay, lis[i], i, n)
	
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
