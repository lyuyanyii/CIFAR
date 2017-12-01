import pickle
import numpy as np

from ntools.visualizer.curve import Curve, WebCurveVisualizer

"""
with open("../plain30/pca_ener.data", "rb") as f:
	data1 = pickle.load(f)

with open("../plain30_dropout/pca_ener.data", "rb") as f:
	data2 = pickle.load(f)

with open("../plain30_orth/pca_ener.data", "rb") as f:
	data3 = pickle.load(f)
"""

paths = ["plain30", "plain30_dropout", "plain30_orth", "plain30_orth_f", "plain30_dp_sup", "plain30_xcep", "plain30_dfconv"]
name = ["NoReg", "Drop", "OrthW", "OrthF", "DpSup", "Xcep", "Dfpool"]
data = []
for i in paths:
	with open("../{}/pca_ener.data".format(i), "rb") as f:
		data.append(pickle.load(f))

curve = Curve(title="PCA energy", x_label="channel", y_label="yy")
idx = 10

"""
for i, j, k in zip(data1, data2, data3):
	idx += 1
	x = range(len(i))
	if idx > 20 and idx <= 30:
		i /= i[-1]
		j /= j[-1]
		k /= k[-1]
		curve.add(i, x, legend='NoReg{}'.format(idx), group_id=idx)
		curve.add(j, x, legend='Dropout{}'.format(idx), style = '.', group_id = idx)
		curve.add(k, x, legend='Orth{}'.format(idx), style = '--', group_id = idx)
"""
x = range(len(data[0][idx]))
for i in range(len(paths)):
	j = data[i][idx]
	j /= j[-1]
	j = j[:len(x)]
	curve.add(j, x, legend=name[i]+"{}".format(idx), group = i)

webcurve = WebCurveVisualizer(curve)
context = webcurve.plot()
WebCurveVisualizer.show(context)
