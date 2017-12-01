import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat 
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider, ParamProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith
import megskull.opr.all as O

from meghair.utils.io import load_network
from megskull.network import NetworkVisitor

global idx
idx = 0

def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	l1 = Conv2D(
		"encoder_conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = G(mean = 0, std = ((1 + int(isrelu)) / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = Identity()
		)
	l2 = BN("encoder_bn{}".format(idx), l1, eps = 1e-9)
	l2 = ElementwiseAffine("bnaff{}".format(idx), l2, shared_in_channels = False, k = C(1), b = C(0))
	if isrelu:
		l2 = arith.ReLU(l2)
	return l2, l1

def deconv_bn_relu(name, inp, kernel_shape = None, stride = None, padding = None, output_nr_channel = None, isbnrelu = True):
	lay = O.Deconv2DVanilla(name, inp, 
		kernel_shape = kernel_shape,
		stride = stride,
		padding = padding,
		output_nr_channel = output_nr_channel)
	if isbnrelu:
		lay = BN(name + "bn", lay, eps = 1e-9)
		lay = ElementwiseAffine(name + "bnaff", lay, shared_in_channels = False, k = C(1), b = C(0))
		lay = arith.ReLU(lay)
	return lay

def make_network(minibatch_size = 128):
	pre_net = load_network("half.data")

	inp = pre_net.outputs[-1]
	visitor = NetworkVisitor(inp).all_oprs
	for i in visitor:
		if isinstance(i, BN):
			i.set_freezed()
		if isinstance(i, ParamProvider):
			i.set_freezed()
		if isinstance(i, DataProvider):
			dp = i
	lay = O.ZeroGrad(inp)
	chl = inp.partial_shape[1]

	p = []
	for tt in range(3):
		lay = deconv_bn_relu("encoder_deconv_{}0".format(tt), lay, kernel_shape = 3, stride = 1, padding = 1, output_nr_channel = chl)
		lay = deconv_bn_relu("encoder_deconv_{}1".format(tt), lay, kernel_shape = 3, stride = 1, padding = 1, output_nr_channel = chl)
		p.append(lay)
		if tt != 2:
			lay = deconv_bn_relu("encoder_deconv_{}2".format(tt), lay, kernel_shape = 2, stride = 2, padding = 0, output_nr_channel = chl // 2)
		chl = chl // 2
	lay = deconv_bn_relu("outputs", lay, kernel_shape = 3, stride = 1, padding = 1, output_nr_channel = 3, isbnrelu = False)
	loss = ((lay - dp)**2).sum(axis = 3).sum(axis = 2).sum(axis = 1).mean()
	network = Network(outputs = [lay, inp] + p)
	network.loss_var = loss
	return network

if __name__ == '__main__':
	make_network()
