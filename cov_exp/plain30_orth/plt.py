import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.signal as signal

with open("hisloss.data", "rb") as f:
	his = pickle.load(f)

his = np.array(his)
hisloss = his[:,1]
hisloss = signal.medfilt(hisloss, 9)
#print(np.max(hisloss[10000:]))
plt.plot(range(len(hisloss)), hisloss)
plt.show()
