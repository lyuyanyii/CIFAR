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
import megbrain as mgb
from megskull.utils.logconf import get_logger

logger = get_logger(__name__)

class Shake_fprop(O.NonTrainableMLPOperatorNodeBase):
	
	opr_attribute = O.NonTrainableMLPOperatorNodeBase.OprAttribute(
		impure=True
	)

	_alpha = None

	def __init__(self, name, inpvar, *, alpha = None):
		inpvar = O.as_varnode(inpvar)
		self._alpha = alpha
		if name is None:
			name = "ShakeFprop({})".format(inpvar.name)
		super().__init__(name, inpvar)
	
	def _init_output_mgbvar(self, env):
		vin = env.get_mgbvar(self._var_input)
		if env.flags.enable_dropout:
			alpha = env.get_mgbvar(self._alpha)
			vin *= alpha
		else:
			vin *= 0.5
		env.set_mgbvar(self._var_output, vin)

global idx
idx = 0

def relu_conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu = True, isbn = True):
	global idx
	idx += 1
	if isrelu:
		inp = arith.ReLU(inp)
	inp = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		nonlinearity = Identity()
		)
	if isbn:
		inp = BN("bn{}".format(idx), inp, eps = 1e-9)
		inp = ElementwiseAffine("bnaff{}".format(idx), inp, shared_in_channels = False, k = C(1), b = C(0))
	return inp

def skip(inp, isdown, chl):
	if isdown == -1:
		return inp
	global idx
	l1 = inp
	if isdown != 0:
		l1 = Pooling2D("pooling1_{}".format(idx), inp, window = 1, stride = 2, mode = "AVERAGE")
	l1 = relu_conv_bn(l1, 1, 1, 0, chl // 2, isrelu = False, isbn = False)

	l2 = inp
	if isdown != 0:
		l2 = Pooling2D("pooling2_{}".format(idx), inp[:, :, 1:, 1:], window = 1, stride = 2, mode = "AVERAGE")
	l2 = relu_conv_bn(l2, 1, 1, 0, chl // 2, isrelu = False, isbn = False)

	lay = O.Concat([l1, l2], axis = 1)
	lay = BN("bn_down_{}".format(isdown), lay, eps = 1e-9)
	lay = ElementwiseAffine("bnaff_down_{}".format(isdown), lay, shared_in_channels = False, k = C(1), b = C(0))
	return lay

def Shake_bprop(inp, beta, alpha):
	inp = O.SetGrad(inp, None)
	inp.set_grad_var(O.GradWrt(inp) / alpha * beta)
	return inp

def get_seed():
	mask = 0xFFFFFFFF
	a, b = map(int, np.random.randint(mask + 1, size=2))
	seed = ((a & mask) << 32) | b
	return seed

def res_layer(inp, chl, stride = 1, isdown = -1):
	l1 = inp
	l1 = relu_conv_bn(l1, 3, stride, 1, chl)
	l1 = relu_conv_bn(l1, 3, 1, 1, chl)

	alpha = O.UniformRNG(inp.shape[0], seed = get_seed()).dimshuffle(0, 'x', 'x', 'x').broadcast(l1.shape)
	beta = O.UniformRNG(inp.shape[0], seed = get_seed()).dimshuffle(0, 'x', 'x', 'x').broadcast(l1.shape)
	alpha1 = alpha
	alpha2 = 1 - alpha
	beta1 = beta
	beta2 = 1 - beta

	#l1 = Shake_fprop(None, l1, alpha = alpha1)
	#l1 = Shake_bprop(l1, beta1, alpha1)

	l2 = inp
	l2 = relu_conv_bn(l2, 3, stride, 1, chl)
	l2 = relu_conv_bn(l2, 3, 1, 1, chl)
	#l2 = Shake_fprop(None, l2, alpha = alpha2)
	#l2 = Shake_bprop(l2, beta2, alpha2)

	SS_params = [l1.partial_shape, alpha1, alpha2, beta1, beta2]
	return skip(inp, isdown, chl) + l1, SS_params

def res_block(inp, chl, n, i):
	stride = 2
	if i == 0:
		stride = 1

	SS_list = []
	inp, SS = res_layer(inp, chl, stride = stride, isdown = i)
	SS_list.append(SS)
	
	for i in range(n - 1):
		inp, SS = res_layer(inp, chl)
		SS_list.append(SS)
	
	return inp, SS_list

def make_network(minibatch_size = 128, debug = False):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size), dtype = np.float32)
	label = DataProvider("label", shape = (minibatch_size, ), dtype = np.int32)

	lay = relu_conv_bn(inp, 3, 1, 1, 16, isrelu = False, isbn = False)

	n = 4
	chl = 96
	SS_list = []
	for i in range(3):
		lay, SS_list_new = res_block(lay, chl, n, i)
		SS_list += SS_list_new
		chl *= 2
	
	#global average pooling
	feature = lay.mean(axis = 2).mean(axis = 2)
	#feature = Pooling2D("pooling", lay, window = 8, stride = 8, padding = 0, mode = "AVERAGE")
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
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

	return network, SS_list

if __name__ == "__main__":
	make_network(debug = True)

