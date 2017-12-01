import numpy as np
from megskull.graph import Function
import megskull.opr.all as O
from megskull.network.visitor import NetworkVisitor

def init(net, batch):
	visitor = NetworkVisitor(net.loss_var)
	lisk = []
	lisb = []
	for i in visitor.all_oprs:
		if ":k" in i.name and "bnaff" in i.name:
			lisk.append(i)
		if ":b" in i.name and "bnaff" in i.name:
			lisb.append(i)
	for i, k, b in zip(range(len(lisk)), lisk, lisb):
		func = Function().compile(net.outputs)
		outputs = func(data = batch['data'])
		t = outputs[1 + i]
		mean = t.mean(axis = 3).mean(axis = 2).mean(axis = 0)
		std = ((t - mean[np.newaxis, :, np.newaxis, np.newaxis])**2).mean(axis = 3).mean(axis = 2).mean(axis = 0)**0.5
		nk = O.ParamProvider("new" + k.name, 1.0 / std)
		nb = O.ParamProvider("new" + b.name, -mean / std)
		visitor.replace_vars([(k, nk), (b, nb)], copy = False)
	
	visitor = NetworkVisitor(net.loss_var)
	for i in visitor.all_oprs:
		print(i)
	return net

