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
from ntools.megtools.tools.info import ComplexityInfo as CInfo
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

def conv_norm(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	inp = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		nonlinearity = Identity()
		)
	mean = inp.mean(axis = 3).mean(axis = 2)
	std = ((inp - mean.dimshuffle(0, 1, 'x', 'x'))**2).mean(axis = 3).mean(axis = 2)**0.5
	inp = (inp - mean.dimshuffle(0, 1, 'x', 'x')) / std.dimshuffle(0, 1, 'x', 'x')
	inp = ElementwiseAffine("aff{}".format(idx), inp, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		inp = O.ReLU(inp)
	return inp

"""
def res_layer(inp, chl, stride = 1, proj = False):
	pre = inp
	inp = conv_bn(inp, 1, stride, 0, chl // 4, True)
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
"""
def make_network(minibatch_size = 128, debug = False):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size), dtype = np.float32)
	label = DataProvider("label", shape = (minibatch_size, ), dtype = np.int32)

	lay = conv_bn(inp, 3, 1, 1, 16, True)

	lis = [16, 32, 64]
	for i in range(len(lis)):
		#lay = res_block(lay, lis[i], i, n)
		for j in range(10):
			lay = conv_norm(lay, 3, 1, 1, lis[i], True)
		if i < len(lis) - 1:
			lay = conv_norm(lay, 2, 2, 0, lis[i + 1], True)
	
	feature = lay
	#global average pooling
	#feature = lay.mean(axis = 2).mean(axis = 2)
	#feature = Pooling2D("pooling", lay, window = 8, stride = 8, padding = 0, mode = "AVERAGE")
	"""
	lay = lay.reshape(lay.shape[0], lay.shape[1], lay.shape[2] * lay.shape[3])
	print(lay.partial_shape)
	a = O.ParamProvider(
		"a", np.random.randn(lay.partial_shape[2], 10) * (1 / lay.partial_shape[2])**0.5)
	a = a.dimshuffle('x', 0, 1)
	a = a.broadcast((lay.partial_shape[0], a.partial_shape[1], a.partial_shape[2]))
	print(a.partial_shape)
	b = O.ParamProvider(
		"b", np.random.randn(lay.partial_shape[2], 10) * (1 / lay.partial_shape[2])**0.5)
	b = b.dimshuffle('x', 0, 1)
	b = b.broadcast((lay.partial_shape[0], b.partial_shape[1], b.partial_shape[2]))
	print(b.partial_shape)
	fca = O.BatchedMatMul(lay, a)
	fcb = O.BatchedMatMul(lay, b)
	fc = O.BatchedMatMul(fca.dimshuffle(0, 2, 1), fcb) / 64
	outs = []
	for i in range(10):
		outs.append(fc[:, i, i].dimshuffle(0, 'x'))
	fc = O.Concat(outs, axis = 1)
	pred = Softmax("pred", fc)
	"""
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		#W = G(mean = 0, std = (1 / 64)**0.5),
		#b = C(0),
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred])
	#info = CInfo()
	#info.get_complexity(network.outputs).as_table().show()
	network.loss_var = CrossEntropyLoss(pred, label)
	
	"""
	if debug:
		visitor = NetworkVisitor(network.loss_var)
		for i in visitor.all_oprs:
			print(i)
			print(i.partial_shape)
			print("input = ", i.inputs)
			print("output = ", i.outputs)
			print()
	"""

	return network

if __name__ == "__main__":
	make_network(debug = True)
