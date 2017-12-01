import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat,
		Floor, Ceil, ones, Cumsum, Min, Max,
		AdvancedIndexing, Astype, Linspace, IndexingRemap,
		Equal,
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider, ConstProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith
import pickle

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

dic = {}

def dfconv(inp, chl, isrelu, flag, ker_shape = 3, stride = 1, padding = 1, dx = [-1, 0, 1], dy = [-1, 0, 1]):
	global idx
	#idx += 1
	gamma = 0.1
	offsetx = gamma * inp.partial_shape[2] * Conv2D(
		"conv{}_offsetx".format(idx + 1), inp, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = C(0),
		nonlinearity = Identity()
		)
	offsety = gamma * inp.partial_shape[3] * Conv2D(
		"conv{}_offsety".format(idx + 1), inp, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = C(0),
		nonlinearity = Identity()
		)

	outputs = []
	for sx in range(2):
		for sy in range(2):
			if sx == 0:
				ofx = Floor(offsetx)
				bilx = offsetx - ofx + Equal(Floor(offsetx), Ceil(offsetx))
			else:
				ofx = Ceil(offsetx)
				bilx = ofx - offsetx
			if sy == 0:
				ofy = Floor(offsety)
				bily = offsety - ofy + Equal(Floor(offsety), Ceil(offsety))
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

			"""
			if flag:
				#one_mat = ConstProvider(np.ones((inp.partial_shape[2], inp.partial_shape[3])), dtype = np.int32)
				one_mat = ConstProvider(1, dtype = np.int32).add_axis(0).broadcast((ofx.partial_shape[2], ofx.partial_shape[3]))
				affx = (Cumsum(one_mat, axis = 0) - 1) * stride
				affy = (Cumsum(one_mat, axis = 1) - 1) * stride

				affx = affx.dimshuffle('x', 'x', 0, 1).broadcast(list(ofx.partial_shape))
				affy = affy.dimshuffle('x', 'x', 0, 1).broadcast(list(ofy.partial_shape))
				affx = ConstProvider(affx.eval())
				affy = ConstProvider(affy.eval())
				ofx = ofx + affx
				ofy = ofy + affy
				one_mat = ConstProvider(np.ones((ker_shape, ofx.partial_shape[2], ofx.partial_shape[3])))
				#ofx[:, :ker_shape, :, :] -= 1
				#ofx[:, ker_shape*2:, :, :] += 1
				affx1 = Concat([one_mat * i for i in dx], axis = 0).dimshuffle('x', 0, 1, 2).broadcast(list(ofx.partial_shape))
				affx1 = ConstProvider(affx1.eval())
				ofx += affx1
				#ofy[:, ::3, :, :] -= 1
				#ofy[:, 2::3, :, :] += 1
				one_mat = ones((1, ofx.partial_shape[2], ofx.partial_shape[3]))
				one_mat = Concat([one_mat * i for i in dy], axis = 0)
				one_mat = Concat([one_mat] * ker_shape, axis = 0)
				affy1 = one_mat.dimshuffle('x', 0, 1, 2).broadcast(list(ofy.partial_shape))
				affy1 = ConstProvider(affy1.eval())
				ofy += affy1
				
				dic["affx"] = affx
				dic["affx1"] = affx1
				dic["affy"] = affy
				dic["affy1"] = affy1
			else:
				ofx = ofx + dic["affx"] + dic["affx1"]
				ofy = ofy + dic["affy"] + dic["affy1"]
			"""
			ofx = Max(Min(ofx, arg_fea.partial_shape[2] - 1), 0)
			ofy = Max(Min(ofy, arg_fea.partial_shape[3] - 1), 0)

			def DeformReshape(inp, ker_shape):
				inp = inp.reshape(inp.partial_shape[0], ker_shape, ker_shape, inp.partial_shape[2], inp.partial_shape[3])
				inp = inp.dimshuffle(0, 3, 1, 4, 2)
				inp = inp.reshape(inp.partial_shape[0], inp.partial_shape[1] * inp.partial_shape[2], inp.partial_shape[3] * inp.partial_shape[4])
				return inp

			ofx = DeformReshape(ofx, ker_shape)
			ofy = DeformReshape(ofy, ker_shape)
			bilx = DeformReshape(bilx, ker_shape)
			bily = DeformReshape(bily, ker_shape)

		
			of = ofx * arg_fea.partial_shape[2] + ofy
			arg_fea = arg_fea.reshape(arg_fea.partial_shape[0], arg_fea.partial_shape[1], -1)
			of = of.reshape(ofx.partial_shape[0], -1)
			of = of.dimshuffle(0, 'x', 1)
			#of = Concat([of] * arg_fea.partial_shape[1], axis = 1)
			of = of.broadcast((of.partial_shape[0], arg_fea.partial_shape[1], of.partial_shape[2]))
			if flag:
				arx = Linspace(0, arg_fea.partial_shape[0], arg_fea.partial_shape[0], endpoint = False)
				arx = arx.add_axis(1).add_axis(2).broadcast(of.shape)
				ary = Linspace(0, arg_fea.partial_shape[1], arg_fea.partial_shape[1], endpoint = False)
				ary = ary.add_axis(0).add_axis(2).broadcast(of.shape)
				arx = arx.add_axis(3)
				ary = ary.add_axis(3)
				
				dic["arx"] = arx
				dic["ary"] = ary
			else:
				arx = dic["arx"]
				ary = dic["ary"]

			of = of.add_axis(3)
			idxmap = Astype(Concat([arx, ary, of], axis = 3), np.int32)
			idxmap = np.zeros(list(idxmap.partial_shape), dtype = np.int32)
			"""
			sample = []
			for i in range(arg_fea.partial_shape[0]):
				for j in range(arg_fea.partial_shape[1]):
					sample.append(arg_fea[i][j].ai[of[i][j]].dimshuffle('x', 0))
			sample = Concat(sample, axis = 0)
			"""
			sample = IndexingRemap(arg_fea, idxmap).reshape(inp.partial_shape[0], inp.partial_shape[1], bilx.partial_shape[1], -1)
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

	ker_shape = window
	stride = window
	gamma = 0.1
	offsetx = gamma * inp.partial_shape[2] * Conv2D(
		name + "offsetx", inp, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = C(0),
		nonlinearity = Identity()
		)
	offsety = gamma * inp.partial_shape[3] * Conv2D(
		name + "offsety", inp, kernel_shape = ker_shape, stride = stride, 
		padding = padding,
		output_nr_channel = ker_shape**2,
		W = C(0),
		nonlinearity = Identity()
		)
	outputs = []
	for sx in range(2):
		for sy in range(2):
			if sx == 0:
				ofx = Floor(offsetx)
				bilx = offsetx - ofx + Equal(Floor(offsetx), Ceil(offsetx))
			else:
				ofx = Ceil(offsetx)
				bilx = ofx - offsetx
			if sy == 0:
				ofy = Floor(offsety)
				bily = offsety - ofy + Equal(Floor(offsety), Ceil(offsety))
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
				inp = inp.reshape(inp.partial_shape[0], ker_shape, ker_shape, inp.partial_shape[2], inp.partial_shape[3])
				inp = inp.dimshuffle(0, 3, 1, 4, 2)
				inp = inp.reshape(inp.partial_shape[0], inp.partial_shape[1] * inp.partial_shape[2], inp.partial_shape[3] * inp.partial_shape[4])
				return inp

			ofx = DeformReshape(ofx, ker_shape)
			ofy = DeformReshape(ofy, ker_shape)
			bilx = DeformReshape(bilx, ker_shape)
			bily = DeformReshape(bily, ker_shape)

			of = ofx * arg_fea.partial_shape[2] + ofy
			arg_fea = arg_fea.reshape(arg_fea.partial_shape[0], arg_fea.partial_shape[1], -1)
			of = of.reshape(ofx.partial_shape[0], -1)
			of = of.dimshuffle(0, 'x', 1)
			#of = Concat([of] * arg_fea.partial_shape[1], axis = 1)
			of = of.broadcast((of.partial_shape[0], arg_fea.partial_shape[1], of.partial_shape[2]))
			arx = Linspace(0, arg_fea.partial_shape[0], arg_fea.partial_shape[0], endpoint = False)
			arx = arx.add_axis(1).add_axis(2).broadcast(of.shape)
			ary = Linspace(0, arg_fea.partial_shape[1], arg_fea.partial_shape[1], endpoint = False)
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
			sample = IndexingRemap(arg_fea, idxmap).reshape(inp.partial_shape[0], inp.partial_shape[1], bilx.partial_shape[1], -1)
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
	idxmap = np.zeros((128, 3, 32, 32, 4), dtype = np.int32)
	sample = IndexingRemap(inp, idxmap)
	network = Network(outputs = [sample])
	sample = FullyConnected("fc", sample, output_dim = 1)
	network.loss_var = sample.sum()
	return network

	#lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)
	lay, conv = conv_bn(inp, 3, 1, 1, 32, True)
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
		lay, conv = dfconv(lay, chl, True, i == 0)

	
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
