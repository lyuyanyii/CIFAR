import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat,
			ParamProvider
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
from megskull.opr.regularizer import Dropout
import megskull.opr.arith as arith
import megbrain as mgb

class MyDropout(Dropout):
	def _init_output_mgbvar(self, env):
		vin = env.get_mgbvar(self._var_input)
		if env.flags.enable_dropout:
			mask = 0xFFFFFFFF
			a, b = map(int, env.rng.randint(mask + 1, size=2))
			seed = ((a & mask) << 32) | b
			#logger.info('{}: enable dropout {}: seed={} incl_prob={}'.format(
			#	env, self.name, hex(seed), self.incl_prob))

			rv = env.mgb_opr.uniform_rng(shape=vin.shape, seed=seed)
			incl_mask = rv < self._incl_prob_ss
			vin *= (incl_mask).astype(vin.dtype)
		else:
			vin *= self._incl_prob_ss
		env.set_mgbvar(self._var_output, vin)

def gaussian(shape, mean = 0, std = 1):
	return np.random.randn(shape[0], shape[1], shape[2], shape[3]) * std + mean

def conv_dropout(name, inp, ker_shape, stride, padding, out_chl):
	W = ParamProvider(name + 'W', initializer = gaussian((out_chl, inp.partial_shape[1], ker_shape, ker_shape), mean = 0, std = (2 / (ker_shape**2 * inp.partial_shape[1]))**0.5))
	W = MyDropout(name + 'W' + 'dropout', W)
	return Conv2D(
		name, inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = W,
		nonlinearity = Identity()
		)

global idx
idx = 0

def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	l1 = conv_dropout("conv{}".format(idx), inp, ker_shape, stride, padding, out_chl)
	"""
	l1 = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = G(mean = 0, std = ((1 + int(isrelu)) / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = Identity()
		)
	"""
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
	feature = MyDropout("dropout", feature)
	#feature = Pooling2D("glbpoling", lay, window = 8, stride = 8, mode = "AVERAGE")
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		W = G(mean = 0, std = (1 / feature.partial_shape[1])**0.5),
		b = C(0),
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred] + out)
	network.loss_var = CrossEntropyLoss(pred, label)
	return network

if __name__ == '__main__':
	make_network()
