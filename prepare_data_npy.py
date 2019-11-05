from pydub import AudioSegment

import librosa
import soundfile as sf
import os
import numpy as np
import random
from pysndfx import AudioEffectsChain

noise_dir = '/media/pranjal/Seagate Backup Plus Drive/DataASR/Noise/noise.npy'
clean_dir = '/media/pranjal/Seagate Backup Plus Drive/DataASR/Russian/ru/clips_wav/'
out_dir_nosiy = '/media/pranjal/Seagate Backup Plus Drive/DataASR/Russian/ru/clips_pickle_noise/'
out_dir_clean = '/media/pranjal/Seagate Backup Plus Drive/DataASR/Russian/ru/clips_pickle_clean/'
noise = np.load(noise_dir) 
fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .phaser()
    # .delay()
    .lowshelf()
)

for file in os.listdir(clean_dir):
	audio, sr = librosa.load(clean_dir+str(file))
	x = random.randrange(0,len(noise)-len(audio),1)
	noisy = audio + noise[x:x+len(audio)]*(np.random.uniform(0.01,0.012))
	noisy = noisy + fx(noisy)*(np.random.uniform(0.13,0.19))
	
	# spectrum = librosa.stft(audio,n_fft = 512) ##For clean speech
	# print(type(spectrum))
	# spectrum_imag = np.imag(spectrum)
	# spectrum_real = np.real(spectrum)
	# spectrum_new = np.dstack((spectrum_real,spectrum_imag))

	# spectrum1 = librosa.stft(noisy,n_fft = 512) ##For noisy audio
	# spectrum1_imag = np.imag(spectrum1)
	# spectrum1_real = np.real(spectrum1)
	# spectrum1_new = np.dstack((spectrum1_real,spectrum1_imag))

	np.save(out_dir_nosiy+str(file)[:-4]+'.npy',noisy,allow_pickle = True)
	np.save(out_dir_clean+str(file)[:-4]+'.npy',audio,allow_pickle = True)
### THE FINAL SAVED ARRAY IS A 3D array, 1st layer = c[:,:,0] (REAL), and 2nd layer = c[:,:,1] (IMAG)


# for file in os.listdir("ru/clips_wav"):
	# numrows = len(input)    # no of rows
	# numcols = len(input[0]) # no of columns

# audio, sample_rate = librosa.load('test.wav')
# print(audio)
# print(len(audio))

# spectrum = librosa.stft(audio,n_fft=512) 
# print(type(spectrum))
# np.save('save.npy',spectrum,allow_pickle=True)
# spec=np.load('save.npy',allow_pickle=True)

# spectrum_imag=np.imag(spectrum)
# spectrum_real=np.real(spectrum)
# spectrum_new=np.dstack((spectrum_real,spectrum_imag))

# print(spectrum_imag)
# print(spectrum_real)
# print(spectrum_new)
# # if(spec==spectrum):
# # 	print("yay")
# m=True
# for i,j in enumerate(spectrum):
# 	for k,l in enumerate(j):
# 		if(spec[i][k]!=l):
# 			m=False
# if(m):
# 	print("ok")
# else:
# 	print("not ok")
	#The spectrum is D[f,t], the rows are the frquencies, and the columns are time frames
	#Keeping the n_fft=512, gives frames of 23ms at 22050Hz

	# print(spectrum[:,0]) #This is the frist column, or the furst frame for discrete time 1
	# print(len(spectrum)) #prints the number of rows, ie. the frequencies
	# print(len(spectrum[0])) #prints the number of columns, ie. the time samples

	# reconstructed_audio = librosa.istft(spectrum)

	# sf.write('out.wav',reconstructed_audio,sample_rate,subtype='PCM_24')