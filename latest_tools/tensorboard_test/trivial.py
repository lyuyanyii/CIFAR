import sys
sys.path.append("/home/liuyanyi02/CIFAR/latest_tools")
from th import TensorboardHelper as TB
import os

tb = TB(os.getcwd())

for t in range(100):
	tb.add_scalar("tmp", t)
	tb.tick()

tb.flush()

