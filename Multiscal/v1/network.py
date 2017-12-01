import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout, ElementwiseAffine, Concat,
		Floor, Ceil, ones, Cumsum, Min, Max,
		AdvancedIndexing, Astype, Linspace, IndexingRemap,
		Equal, ZeroGrad, Resize,
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

def b_resize(name, inp, rate = 0.8):
	#inp = ConstProvider([[[[1, 2], [3, 4]]]], dtype = np.float32)
	f_size = inp.partial_shape[2]
	l = int(f_size * rate)
	s = [[0, l], [f_size - l, f_size]]
	ar0 = Linspace(0, inp.shape[0], inp.shape[0], endpoint = False)
	ar0 = ar0.add_axis(1).add_axis(2).add_axis(3).broadcast(inp.shape).add_axis(4)
	ar1 = Linspace(0, inp.shape[1], inp.shape[1], endpoint = False)
	ar1 = ar1.add_axis(0).add_axis(2).add_axis(3).broadcast(inp.shape).add_axis(4)

	fmaps = [inp]
	for i in range(4):
		xx = s[i % 2]
		yy = s[i // 2]
		#x = Linspace(xx[0], xx[1], f_size, endpoint = False)
		#y = Linspace(yy[0], yy[1], f_size, endpoint = False)
		x = ConstProvider(np.linspace(xx[0], xx[1], f_size, endpoint = False))
		y = ConstProvider(np.linspace(yy[0], yy[1], f_size, endpoint = False))
		fx, fy = Floor(x), Floor(y)
		cx, cy = Ceil(x), Ceil(y)
		nfmaps = []
		for sx in range(2):
			for sy in range(2):
				ix = fx if sx == 0 else cx
				iy = fy if sy == 0 else cy
				bx = (cx - x + Equal(fx, cx) if sx == 0 else x - fx)
				by = (cy - y + Equal(fy, cy) if sy == 0 else y - fy)
				arx = ix.add_axis(0).add_axis(0).add_axis(3).broadcast(inp.shape).add_axis(4)
				ary = iy.add_axis(0).add_axis(0).add_axis(0).broadcast(inp.shape).add_axis(4)
				idxmap = Astype(Concat([ar0, ar1, arx, ary], axis = 4), np.int32)
				sample = IndexingRemap(inp, idxmap)
				sample *= bx.dimshuffle('x', 'x', 0, 'x') * by.dimshuffle('x', 'x', 'x', 0)
				nfmaps.append(sample)
		fmap = nfmaps[0]
		for i in range(1, 4):
			fmap += nfmaps[i]
		fmaps.append(fmap)
	fmap = Concat(fmaps, axis = 1)
	return fmap

def make_network(minibatch_size = 128):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 15, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	#lay = bn_relu_conv(inp, 3, 1, 1, 16, False, False)
	lay, conv = conv_bn(inp, 3, 1, 1, 16, True)
	out = [conv]
	for chl in [32, 64, 128]:
		for i in range(10):
			lay, conv = conv_bn(lay, 3, 1, 1, chl, True)
			out.append(conv)
		if chl != 128:
			lay = b_resize("pooling{}".format(chl), lay)
			lay = Pooling2D("pooling{}".format(chl), lay, window = 2, mode = "MAX")
	
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
