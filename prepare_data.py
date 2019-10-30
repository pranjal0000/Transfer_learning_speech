from pydub import AudioSegment

import librosa
import soundfile as sf
import os
import numpy as np

path_dir = input("Enter the absolute path of the mp3 files: ")
out_dir = input("Enter the absolute path for the output files: ")
pickle_dir = input("ENter the absolute path for the pickle dump")

for file in os.listdir(str(path_dir)):
	# print(str(file))
	sound = AudioSegment.from_mp3(str(path_dir)+str(file))
	temp=str(file)
	if temp.endswith('.mp3'):
		temp = temp[:-4]

	# print(temp)
	sound.export(str(out_dir)+temp+".wav", format="wav")

	audio, sample_rate = librosa.load(str(out_dir)+temp+'.wav')


	spectrum = librosa.stft(audio,n_fft=512) 
	# print(type(spectrum))
	spectrum_imag=np.imag(spectrum)
	spectrum_real=np.real(spectrum)
	spectrum_new=np.dstack((spectrum_real,spectrum_imag))

	np.save(str(pickle_dir)+temp+'.npy',spectrum_new,allow_pickle=True)
	# spec=np.load('save.npy',allow_pickle=True)
	# if(spec==spectrum):
	# 	print("yay")
	# m=True
	# for i,j in enumerate(spectrum):
	# 	for k,l in enumerate(j):
	# 		if(spec[i][k]!=l):
	# 			m=False
	# if(m):
	# 	print("ok")
	# else:
	# 	print("not ok")

#THE ABOVE SECTION OF CODE CONVERTS MP3 TO WAV

# numrows = len(input)    # no of rows
# numcols = len(input[0]) # no of columns

# audio, sample_rate = librosa.load('test.wav')

# spectrum = librosa.stft(audio,n_fft=512) 

# #The spectrum is D[f,t], the rows are the frquencies, and the columns are time frames
# #Keeping the n_fft=512, gives frames of 23ms at 22050Hz

# # print(spectrum[:,0]) #This is the frist column, or the furst frame for discrete time 1
# # print(len(spectrum)) #prints the number of rows, ie. the frequencies
# # print(len(spectrum[0])) #prints the number of columns, ie. the time samples

# reconstructed_audio = librosa.istft(spectrum)

# sf.write('out.wav',reconstructed_audio,sample_rate,subtype='PCM_24')