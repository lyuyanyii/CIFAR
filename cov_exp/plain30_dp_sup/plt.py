import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.signal as signal

with open("histest.data", "rb") as f:
	acc = np.array(pickle.load(f))

plt.plot(acc[:,0], acc[:,1], 'b')

with open("hisloss.data", "rb") as f:
	his = pickle.load(f)

his = np.array(his)
hisloss = his[:,0]
#hisloss = signal.medfilt(hisloss, 9)
hisloss = signal.convolve(hisloss, np.ones((9)) / 9)

#with open("../resnet20_newpipe/hisloss.data", "rb") as f:
#	his_new = pickle.load(f)
#hisloss_new = np.array(his_new)[:, 0]
#hisloss_new = signal.medfilt(hisloss_new, 9)
#hisloss_new = signal.convolve(hisloss_new, np.ones((9)) / 9)
#plt.plot(range(len(hisloss)), hisloss, 'g')

#plt.plot(range(len(hisloss)), hisloss)

plt.show()
