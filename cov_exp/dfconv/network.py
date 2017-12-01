import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat,
		Floor, Ceil, ones, Cumsum, Min, Max,
		AdvancedIndexing, Astype, Linspace, IndexingRemap,
		Equal, ZeroGrad,
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider, ConstProvider
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

def dfconv(inp, chl, isrelu, ker_shape = 3, stride = 1, padding = 1, dx = [-1, 0, 1], dy = [-1, 0, 1]):
	global idx
	name = "conv{}".format(idx)
	offsetlay = Conv2D(
		name + "conv1", inp, kernel_shape = 3, stride = 1, padding = 1,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = ((1) / (3**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = Identity()
		)
	offsetlay = BN(name + "BN1", offsetlay, eps = 1e-9)
	offsetlay = arith.ReLU(offsetlay)
	offsetlay = Conv2D(
		name + "conv2", inp, kernel_shape = 3, stride = 1, padding = 1,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = ((1) / (3**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = Identity()
		)
	offsetlay = BN(name + "BN2", offsetlay, eps = 1e-9)

	offsetx = inp.partial_shape[2] * Conv2D(
		name + "offsetx", offsetlay, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = (1 / (ker_shape**2 * inp.partial_shape[2]))**0.5),
		nonlinearity = Identity()
		)
	offsety = inp.partial_shape[3] * Conv2D(
		name + "offsety", offsetlay, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = (1 / (ker_shape**2 * inp.partial_shape[3]))**0.5),
		nonlinearity = Identity()
		)

	"""
	gamma = 0.0001
	ndim = ker_shape**2 * offsetx.partial_shape[2] * offsetx.partial_shape[3]
	offsetx = FullyConnected(
		name + "offsetx", offsetx, output_dim = ndim,
		W = G(mean = 0, std = (1 / ndim)**0.5),
		b = C(0),
		nonlinearity = Identity()
		)
	offsetx = offsetx.reshape(offsety.shape)
	offsety = FullyConnected(
		name + "offsety", offsety, output_dim = ndim,
		W = G(mean = 0, std = (1 / ndim)**0.5),
		b = C(0),
		nonlinearity = Identity()
		)
	offsety = offsety.reshape(offsetx.shape)
	"""
	
	outputs = []
	for sx in range(2):
		for sy in range(2):
			if sx == 0:
				ofx = Floor(offsetx)
				bilx = offsetx - ofx
			else:
				ofx = Ceil(offsetx)
				bilx = ofx - offsetx
			if sy == 0:
				ofy = Floor(offsety)
				bily = offsety - ofy
			else:
				ofy = Ceil(offsety)
				bily = ofy - offsety

			"""
			No padding
			padding1 = ConstProvider(np.zeros((inp.partial_shape[0], inp.partial_shape[1], 1, inp.partial_shape[3])))
			padding2 = ConstProvider(np.zeros((inp.partial_shape[0], inp.partial_shape[1], inp.partial_shape[2] + 2, 1)))
			arg_fea = Concat([padding1, inp, padding1], axis = 2)
			arg_fea = Concat([padding2, arg_fea, padding2], axis = 3)
			"""
			arg_fea = inp

			#one_mat = ConstProvider(np.ones((inp.partial_shape[2], inp.partial_shape[3])), dtype = np.int32)
			one_mat = ConstProvider(1, dtype = np.int32).add_axis(0).broadcast((ofx.partial_shape[2], ofx.partial_shape[3]))
			affx = (Cumsum(one_mat, axis = 0) - 1) * stride
			affy = (Cumsum(one_mat, axis = 1) - 1) * stride

			ofx = ofx + affx.dimshuffle('x', 'x', 0, 1)
			ofy = ofy + affy.dimshuffle('x', 'x', 0, 1)
			one_mat = ConstProvider(np.ones((ker_shape, ofx.partial_shape[2], ofx.partial_shape[3])))
			#ofx[:, :ker_shape, :, :] -= 1
			#ofx[:, ker_shape*2:, :, :] += 1
			ofx += Concat([one_mat * i for i in dx], axis = 0).dimshuffle('x', 0, 1, 2)
			#ofy[:, ::3, :, :] -= 1
			#ofy[:, 2::3, :, :] += 1
			one_mat = ones((1, ofx.partial_shape[2], ofx.partial_shape[3]))
			one_mat = Concat([one_mat * i for i in dy], axis = 0)
			one_mat = Concat([one_mat] * ker_shape, axis = 0)
			ofy += one_mat.dimshuffle('x', 0, 1, 2)
			ofx = Max(Min(ofx, arg_fea.partial_shape[2] - 1), 0)
			ofy = Max(Min(ofy, arg_fea.partial_shape[3] - 1), 0)

			def DeformReshape(inp, ker_shape):
				inp = inp.reshape(inp.shape[0], ker_shape, ker_shape, inp.shape[2], inp.shape[3])
				inp = inp.dimshuffle(0, 3, 1, 4, 2)
				inp = inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2], inp.shape[3] * inp.shape[4])
				return inp

			ofx = DeformReshape(ofx, ker_shape)
			ofy = DeformReshape(ofy, ker_shape)
			bilx = DeformReshape(bilx, ker_shape)
			bily = DeformReshape(bily, ker_shape)

			of = ofx * arg_fea.shape[2] + ofy
			arg_fea = arg_fea.reshape(arg_fea.shape[0], arg_fea.shape[1], -1)
			of = of.reshape(ofx.shape[0], -1)
			of = of.dimshuffle(0, 'x', 1)
			#of = Concat([of] * arg_fea.partial_shape[1], axis = 1)
			of = of.broadcast((of.shape[0], arg_fea.shape[1], of.shape[2]))
			arx = Linspace(0, arg_fea.shape[0], arg_fea.shape[0], endpoint = False)
			arx = arx.add_axis(1).add_axis(2).broadcast(of.shape)
			ary = Linspace(0, arg_fea.shape[1], arg_fea.shape[1], endpoint = False)
			ary = ary.add_axis(0).add_axis(2).broadcast(of.shape)
			of = of.add_axis(3)
			arx = arx.add_axis(3)
			ary = ary.add_axis(3)
			idxmap = Astype(Concat([arx, ary, of], axis = 3), np.int32)
			"""
			sample = []
			for i in range(arg_fea.partial_shape[0]):
				for j in range(arg_fea.partial_shape[1]):
					sample.append(arg_fea[i][j].ai[of[i][j]].dimshuffle('x', 0))
			sample = Concat(sample, axis = 0)
			"""
			sample = IndexingRemap(arg_fea, idxmap).reshape(inp.shape[0], inp.shape[1], bilx.shape[1], -1)
			bilx = bilx.dimshuffle(0, 'x', 1, 2).broadcast(sample.shape)
			bily = bily.dimshuffle(0, 'x', 1, 2).broadcast(sample.shape)
			sample *= bilx * bily
			
			outputs.append(sample)
	
	output = outputs[0]
	for i in outputs[1:]:
		output += i
	
	return conv_bn(output, ker_shape, 3, 0, chl, isrelu)

def dfpooling(name, inp, window = 2, padding = 0, dx = [0, 1], dy = [0, 1]):
	#inp = ConstProvider([[[[1, 2], [3, 4]]]], dtype = np.float32)
	"""
	Add a new conv&bn to insure that the scale of the feature map is variance 1.
	"""
	ker_shape = window
	stride = window	
	offsetlay = Conv2D(
		name + "conv", inp, kernel_shape = 3, stride = 1, padding = 1,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = ((1) / (3**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = Identity()
		)
	offsetlay = BN(name + "BN", offsetlay, eps = 1e-9)

	offsetx = inp.partial_shape[2] * Conv2D(
		name + "conv1x", offsetlay, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = 1 / (ker_shape**2 * inp.partial_shape[2])),
		nonlinearity = Identity()
		)
	offsety = inp.partial_shape[3] * Conv2D(
		name + "conv1y", offsetlay, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = G(mean = 0, std = 1 / (ker_shape**2 * inp.partial_shape[3])),
		nonlinearity = Identity()
		)

	gamma = 0.0001
	ndim = ker_shape**2 * offsetx.partial_shape[2] * offsetx.partial_shape[3]
	offsetx = FullyConnected(
		name + "offsetx", offsetx, output_dim = ndim,
		W = G(mean = 0, std = gamma / ndim),
		b = C(0),
		nonlinearity = Identity()
		)
	offsetx = offsetx.reshape(offsety.shape)
	offsety = FullyConnected(
		name + "offsety", offsety, output_dim = ndim,
		W = G(mean = 0, std = gamma / ndim),
		b = C(0),
		nonlinearity = Identity()
		)
	offsety = offsety.reshape(offsetx.shape)
	print(offsety.partial_shape)

	#offsetx = ZeroGrad(offsetx)
	#offsety = ZeroGrad(offsety)
	outputs = []
	for sx in range(2):
		for sy in range(2):
			if sx == 0:
				ofx = Floor(offsetx)
				bilx = 1 - (offsetx - ofx)
			else:
				ofx = Ceil(offsetx)
				bilx = 1 - (ofx - offsetx)
			if sy == 0:
				ofy = Floor(offsety)
				bily = 1 - (offsety - ofy)
			else:
				ofy = Ceil(offsety)
				bily = 1 - (ofy - offsety)
			"""
			No padding
			padding1 = ConstProvider(np.zeros((inp.partial_shape[0], inp.partial_shape[1], 1, inp.partial_shape[3])))
			padding2 = ConstProvider(np.zeros((inp.partial_shape[0], inp.partial_shape[1], inp.partial_shape[2] + 2, 1)))
			arg_fea = Concat([padding1, inp, padding1], axis = 2)
			arg_fea = Concat([padding2, arg_fea, padding2], axis = 3)
			"""
			arg_fea = inp

			#one_mat = ConstProvider(np.ones((inp.partial_shape[2], inp.partial_shape[3])), dtype = np.int32)
			one_mat = ConstProvider(1, dtype = np.int32).add_axis(0).broadcast((ofx.shape[2], ofx.shape[3]))
			affx = (Cumsum(one_mat, axis = 0) - 1) * stride
			affy = (Cumsum(one_mat, axis = 1) - 1) * stride

			ofx = ofx + affx.dimshuffle('x', 'x', 0, 1)
			ofy = ofy + affy.dimshuffle('x', 'x', 0, 1)
			one_mat = ConstProvider(np.ones((ker_shape, ofx.partial_shape[2], ofx.partial_shape[3])))
			#ofx[:, :ker_shape, :, :] -= 1
			#ofx[:, ker_shape*2:, :, :] += 1
			ofx += Concat([one_mat * i for i in dx], axis = 0).dimshuffle('x', 0, 1, 2)
			#ofy[:, ::3, :, :] -= 1
			#ofy[:, 2::3, :, :] += 1
			one_mat = ones((1, ofx.partial_shape[2], ofx.partial_shape[3]))
			one_mat = Concat([one_mat * i for i in dy], axis = 0)
			one_mat = Concat([one_mat] * ker_shape, axis = 0)
			ofy += one_mat.dimshuffle('x', 0, 1, 2)
			ofx = Max(Min(ofx, arg_fea.partial_shape[2] - 1), 0)
			ofy = Max(Min(ofy, arg_fea.partial_shape[3] - 1), 0)

			def DeformReshape(inp, ker_shape):
				inp = inp.reshape(inp.shape[0], ker_shape, ker_shape, inp.shape[2], inp.partial_shape[3])
				inp = inp.dimshuffle(0, 3, 1, 4, 2)
				inp = inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2], inp.shape[3] * inp.shape[4])
				return inp

			ofx = DeformReshape(ofx, ker_shape)
			ofy = DeformReshape(ofy, ker_shape)
			bilx = DeformReshape(bilx, ker_shape)
			bily = DeformReshape(bily, ker_shape)

			of = ofx * arg_fea.partial_shape[2] + ofy
			arg_fea = arg_fea.reshape(arg_fea.shape[0], arg_fea.shape[1], -1)
			of = of.reshape(ofx.shape[0], -1)
			of = of.dimshuffle(0, 'x', 1)
			#of = Concat([of] * arg_fea.partial_shape[1], axis = 1)
			of = of.broadcast((of.shape[0], arg_fea.shape[1], of.shape[2]))
			arx = Linspace(0, arg_fea.shape[0], arg_fea.shape[0], endpoint = False)
			arx = arx.add_axis(1).add_axis(2).broadcast(of.shape)
			ary = Linspace(0, arg_fea.shape[1], arg_fea.shape[1], endpoint = False)
			ary = ary.add_axis(0).add_axis(2).broadcast(of.shape)
			of = of.add_axis(3)
			arx = arx.add_axis(3)
			ary = ary.add_axis(3)
			idxmap = Astype(Concat([arx, ary, of], axis = 3), np.int32)
			"""
			sample = []
			for i in range(arg_fea.partial_shape[0]):
				for j in range(arg_fea.partial_shape[1]):
					sample.append(arg_fea[i][j].ai[of[i][j]].dimshuffle('x', 0))
			sample = Concat(sample, axis = 0)
			"""
			sample = IndexingRemap(arg_fea, idxmap).reshape(inp.shape[0], inp.shape[1], bilx.shape[1], -1)
			bilx = bilx.dimshuffle(0, 'x', 1, 2).broadcast(sample.shape)
			bily = bily.dimshuffle(0, 'x', 1, 2).broadcast(sample.shape)
			sample *= bilx * bily
			
			outputs.append(sample)
	
	output = outputs[0]
	for i in outputs[1:]:
		output += i
	
	return Pooling2D(name, output, window = 2, mode = "AVERAGE")


def make_network(minibatch_size = 128):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	#lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)
	lay, conv = conv_bn(inp, 3, 1, 1, 16, True)
	out = [conv]
	"""
	for chl in [32, 64, 128]:
		for i in range(10):
			lay, conv = conv_bn(lay, 3, 1, 1, chl, True)
			out.append(conv)
		if chl != 128:
			lay = dfpooling("pooling{}".format(chl), lay)
	"""
	chl = 32
	for i in range(3):
		lay, conv = dfconv(lay, chl, True)


	
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
	return network

if __name__ == '__main__':
	make_network()
