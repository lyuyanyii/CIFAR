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
from megskull.opr.helper.mgb_forward import MGBOprForwarderBase
from megskull.utils.meta import override
import megbrain.opr as O
import megbrain as mgb

class MyConv(MGBOprForwarderBase):
	_meta_data_type_type = mgb.opr_param_defs.Convolution.DataType
	_meta_conv_mode_type = mgb.opr_param_defs.Convolution.Mode
	_meta_sparse_type = mgb.opr_param_defs.Convolution.Sparse
	_meta_format_type = mgb.opr_param_defs.Convolution.Format
	conv_mode = _meta_conv_mode_type.CROSS_CORRELATION
	data_type = _meta_data_type_type.FLOAT
	_sparse_type = _meta_sparse_type.GROUP
	format = _meta_format_type.NCHW
	"""?x? conv, weights are different bewteen minibatch-dim"""
	def __init__(self, inp, W, *, name = None):
		if name is None:
			name = "MyConv({}, {})".format(inp.name, W.name)
		super().__init__(inputs = [inp, W], name = name)
	
	@override(MGBOprForwarderBase, check_sig=False)
	def _mgb_func(self, env, vin, W):
		vin = vin.reshape(1, vin.shape[0] * vin.shape[1], vin.shape[2], vin.shape[3])
		param_for_opr_impl = mgb.opr_param_defs.Convolution(
			mode = self.conv_mode,
			pad_h = 0, pad_w = 0,
			stride_h = 1, stride_w = 1,
			dilate_h = 1, dilate_w = 1,
			data_type = self.data_type, format = self.format,
			sparse = self._sparse_type)
		return O.convolution(
			vin, W, param = param_for_opr_impl,
			strategy = env.flags.conv_execution_strategy,
			workspace_limit=env.flags.conv_execution_workspace_limit
			)

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

def res_layer(inp, chl):
	pre = inp
	inp = conv_bn(inp, 3, 1, 1, chl, True)
	inp = conv_bn(inp, 3, 1, 1, chl, False)
	name = inp.name
	#Global Average Pooling
	SE = inp.mean(axis = 3).mean(axis = 2)
	group = 1
	#fc0
	SE = FullyConnected(
		"fc0({})".format(name), SE, output_dim = chl,
		nonlinearity = ReLU()
		)
	#fc1
	SE = FullyConnected(
		"fc1({})".format(name), SE, output_dim = (chl // group)**2 * group,
		nonlinearity = Sigmoid()
		)
	SE = SE.reshape(inp.shape[0] * group, chl // group, chl // group, 1, 1)
	SE /= SE.sum(axis = 4).sum(axis = 3).sum(axis = 2).dimshuffle(0, 1, 'x', 'x', 'x')
	inp = MyConv(inp, SE)
	"""
	#inp = inp * SE.dimshuffle(0, 1, 'x', 'x')
	inp = inp.reshape(1, inp.shape[0] * inp.shape[1], inp.shape[2], inp.shape[3])
	inp = Conv2D(
		"conv({})".format(name), inp, kernel_shape = 1, stride = 1, padding = 0,
		#output_nr_channel = chl,
		W = SE,
		nonlinearity = Identity(),
		#group = group
		)
	"""
	inp = inp.reshape(pre.shape)
	inp = arith.ReLU(inp + pre)
	return inp

def res_block(inp, chl, n):
	stride = 2
	if chl == 16:
		stride = 1
	pre = inp
	inp = conv_bn(inp, 3, stride, 1, chl, True)
	inp = conv_bn(inp, 3, 1, 1, chl, False)
	inp = inp + conv_bn(pre, 1, stride, 0, chl, False)
	inp = arith.ReLU(inp)
	
	for i in range(n - 1):
		inp = res_layer(inp, chl)
	
	return inp

def make_network(minibatch_size = 128, debug = False):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size), dtype = np.float32)
	label = DataProvider("label", shape = (minibatch_size, ), dtype = np.int32)

	lay = conv_bn(inp, 3, 1, 1, 16, True)

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
