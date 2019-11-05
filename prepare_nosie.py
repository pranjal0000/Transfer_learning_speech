from pydub import AudioSegment

import librosa
import soundfile as sf
import os
import numpy as np
import random

a=[]
a=np.array(a)
sample_rate = 0
for file in os.listdir('FSDKaggle2018.audio_test'):
	au, sr = librosa.load('FSDKaggle2018.audio_test/'+str(file))
	# if sample_rate == 0:
	sample_rate = sr 
	temp = librosa.effects.trim(au,top_db=5)
	a = np.append(temp[0],a)

# print(len(a))
new = []
c = 0
temp = []

for i in a:
	temp.append(i)
	c = (c + 1) % 3000
	if c == 0:
		new.append(temp)
		temp = []

new.append(temp)

random.shuffle(new)

fin = []

for i in new:
	for j in i:
		fin.append(j)

print(len(fin))
fin = np.array(fin)

# np.random.shuffle(a)
np.save('Noise/noise.npy',fin,allow_pickle = True)

sf.write('out.wav',fin,sample_rate,subtype='PCM_24')