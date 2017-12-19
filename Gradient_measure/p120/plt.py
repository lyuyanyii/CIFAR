import sys
sys.path.append("/unsullied/sharefs/liuyanyi02/lyy/CIFAR/latest_tools")
from th import TensorboardHelper as TB
import pickle
import numpy as np

def Draw():
	for net_name in ["p120", "r120", "d120", "d40", "r40"]:
		tb = TB(net_name + "_tbdata/")
		grad_lis = pickle.load(open(net_name + "_grad.data", "rb"))
		for i in grad_lis:
			tb.tick()
			tb.add_scalar("loged_grad", np.log(i))
			tb.add_scalar("grad", i)
		tb.flush()

if __name__ == '__main__':
	Draw()
