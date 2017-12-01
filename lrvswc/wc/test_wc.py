import megskull.opr.all as O
import numpy as np

a = O.ParamProvider('a', np.ones((10, )))
b = O.ParamProvider('b', np.ones((10, )))
loss = O.ZeroGrad((a * b).sum())

from My import MyWeightDecay

loss = MyWeightDecay(loss, {"*":0.001})

from megskull.optimizer import NaiveSGD, OptimizableFunc

func = OptimizableFunc.make_from_loss_var(loss)
NaiveSGD(1)(func)
func.compile(loss)

print(func())
print(np.array(a.eval(), dtype = np.float32))

loss.Mul_Wc(10)

print(func())
print(np.array(a.eval()))

print(func())
