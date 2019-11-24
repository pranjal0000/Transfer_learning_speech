# import numpy as np
# import matplotlib
# from matplotlib import pyplot
# import scipy.integrate as it
# import librosa
# import math
# import os


# a = np.load('Noise/noise.npy')
# x = matplotlib.pyplot.psd(a)
# b = it.cumtrapz(x[0],x[1])

# # b[-1] is the Energy of signal
# l = []
# cdir = 'Russian/ru/clips_wav/'

# for file in os.listdir(str(cdir)):
# 	a1 = librosa.load(cdir+str(file))
# 	for i in a1:
# 		l.append(a1)

# x1 = matplotlib.pyplot.psd(l)
# b1 = it.cumtrapz(x1[0],x1[1])

# snr = float(b1/b)

# print(10*(math.log(snr,10)))

import numpy as np
import matplotlib
from matplotlib import pyplot
import scipy.integrate as it
import librosa
import math
import os


# a = np.load('Noise/noise.npy')
# # print(a.shape)
# x = matplotlib.pyplot.psd(a)
# b = it.cumtrapz(x[0],x[1])

# b[-1] is the Energy of signal
# ct = 0
sdir = 'noisy'
cdir = 'out'
l0 = []
for file in os.listdir(sdir):
	a1, sr = librosa.load(sdir+'/'+str(file))
	a2, sr = librosa.load(cdir+'/'+str(file))
	
	a1 = a1 - a2
	# ct = ct + 1

	# print(type(a1[0]), type(a1[1]))

	for i in a1:
		l0.append(i)






l = []
cdir = 'out'

# ct = 0

for file in os.listdir(cdir):
	a1, sr = librosa.load(cdir+'/'+str(file))
	# ct = ct + 1

	print(type(a1[0]), type(a1[1]))

	for i in a1:
		l.append(i)

	# print(l)
	# if ct == 100:
	# 	break

# l = np.array(l)
# print(l)
# delay(1)

x = matplotlib.pyplot.psd(l0)
b = it.cumtrapz(x[0],x[1])

x1 = matplotlib.pyplot.psd(l)
b1 = it.cumtrapz(x1[0],x1[1])

snr = float(b1[-1]/(b[-1]))

print(10*(math.log(snr,10)))









