import pickle
import numpy as np

W0 = pickle.load(open("p30W.data", "rb"))
W1 = pickle.load(open("p30lrW.data", "rb"))

W0 = W0 / ((W0**2).sum(axis = 0)**0.5)
W1 = W1 / ((W1**2).sum(axis = 0)**0.5)

A = np.dot(W0.T, W1)
I = np.identity(10)
print(A)
print(((A - I)**2).mean())
