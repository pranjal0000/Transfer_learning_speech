import os
# import pickle
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from pydub import AudioSegment

import librosa
import soundfile as sf
from pysndfx import AudioEffectsChain

class CommonVoice(Dataset):
	"""
		root_dir consists of the pickle dump of numpy arrays of im and re parts of the audio
	"""

	def __init__(self,root_dir, transform = transforms.ToTensor(), type = 'train'):
		super(CommonVoice).__init__()
		self.mapping = {}
		self.root_dir = ''
		self.transform = transform

		if type == 'train':
			self.dir = self.root_dir
		elif type == 'test':
			self.dir = self.root_dir

		files = os.listdir(self.dir+'/clips_pickle_noise')
		c = 0
		if type == 'train':
			files = files[0:int(len(files)*0.9)]
		elif type == 'test':
			files = files[int(len(files)*0.9):len(files)]

		for i in range(files):
			self.mapping[i] = files[i]	


	def __len__(self):
		return len(self.mapping)

	def __getitem__(self, idx):

		audio1 = np.load(self.dir + '/clips_pickle_noise/' + self.mapping[idx],allow_pickle = True)
		audio2 = np.load(self.dir + '/clips_pickle_clean/' + self.mapping[idx], allow_pickle = True)
		
		spectrum = librosa.stft(audio1,n_fft = 512) ##For clean speech
		# print(type(spectrum))
		spectrum_imag = np.imag(spectrum)
		spectrum_real = np.real(spectrum)
		spectrum_new = np.dstack((spectrum_real,spectrum_imag))

		spectrum1 = librosa.stft(audio2,n_fft = 512) ##For noisy audio
		spectrum1_imag = np.imag(spectrum1)
		spectrum1_real = np.real(spectrum1)
		spectrum1_new = np.dstack((spectrum1_real,spectrum1_imag))


		audio = np.dstack((spectrum_new,spectrum1_new))
		audio = self.transform(audio)

		return audio



