import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.signal as signal

with open("histest.data", "rb") as f:
	acc = np.array(pickle.load(f))

with open("../plain30_orth_f/histest.data", "rb") as f:
	acc_dp = np.array(pickle.load(f))

plt.plot(acc[:,0], acc[:,1], 'b')
plt.plot(acc_dp[:,0], acc_dp[:,1], 'g')

with open("hisloss.data", "rb") as f:
	his = pickle.load(f)

his = np.array(his)
hisloss = his[:,1]
#hisloss = signal.medfilt(hisloss, 9)
hisloss = signal.convolve(hisloss, np.ones((9)) / 9)

with open("../plain30/hisloss.data", "rb") as f:
	his_new = pickle.load(f)
hisloss_new = np.array(his_new)[:, 1]
#hisloss_new = signal.medfilt(hisloss_new, 9)
hisloss_new = signal.convolve(hisloss_new, np.ones((9)) / 9)
plt.plot(range(len(hisloss)), hisloss, 'b')

plt.plot(range(len(hisloss_new)), hisloss_new, 'g')

plt.show()
