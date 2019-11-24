import sys

import torch.utils.data as data
import torch
from torch.autograd import Variable
import os
import numpy as np
import random

import shutil

from pydub import AudioSegment

import librosa
import soundfile as sf
import os
import numpy as np
import random
from pysndfx import AudioEffectsChain
from src.model.unet_model import UNet
from torchvision import transforms
from torch.utils.data import DataLoader as DL
from torch.utils.data import Dataset

from src.read_yaml import read_yaml

config = read_yaml()

class CommonVoice(Dataset):
	def __init__(self):
		super(CommonVoice).__init__()
		self.mapping = {}
		self.root_dir = clean_dir
		self.transform = transforms.ToTensor()

		files = os.listdir(self.root_dir)

		for i in range(len(files)):
			self.mapping[i] = files[i]	


	def __len__(self):
		return len(self.mapping)

	def __getitem__(self,idx):
		audio, sr = librosa.load(self.root_dir + self.mapping[idx])

		x = random.randrange(0,len(noise)-len(audio),1)
		# noisy = audio
		noisy = audio + noise[x:x+len(audio)]*(np.random.uniform(0.1,0.2))
		noisy = noisy + fx(noisy)*(np.random.uniform(0.33,0.43))

		spectrum = librosa.stft(audio,n_fft = 512)
		sf.write('noisy/'+self.mapping[idx],noisy,22050,subtype='PCM_24')

		audio = audio + noise[x:x+len(audio)]*(np.random.uniform(0.003,0.004))
		audio = audio + fx(noisy)*(np.random.uniform(0.1,0.15))		
		sf.write('out/'+self.mapping[idx],audio,22050,subtype='PCM_24')
		# print(spectrum.shape)

		spectrum_imag = np.imag(spectrum)
		spectrum_real = np.real(spectrum)
		spectrum_new = np.dstack((spectrum_real,spectrum_imag))
		audio = self.transform(spectrum_new).float()

		return audio


noise_dir = 'DataASR/Noise/noise.npy'
clean_dir = 'data_test/'

out_dir = 'data_out/'

tarnsform = transforms.ToTensor()

noise = np.load(noise_dir) 
fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
    .phaser()
    # .delay()
    .lowshelf()
)


model = UNet(config)
checkpoint = torch.load('Exp/19250_2_checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
# model.eval()



dataset = CommonVoice()
trainLoader = DL(dataset, batch_size = 1, shuffle = False, num_workers = 2)

# print(model)


for no, data in enumerate(trainLoader):


	print(data.size())

	data = data.to('cuda')
	
	model.cuda()
	output = model(data)

	output = output.detach().cpu().numpy()
	data = data.detach().cpu().numpy()

	datax = data[0,0,:,:] + 1j*output[0,1,:,:]
	outputx = output[0,0,:,:] + 1j*output[0,1,:,:]

	outputx.reshape([output.shape[2],output.shape[3]])
	datax.reshape([data.shape[2],data.shape[3]])

	reconstructed_audio = librosa.istft(outputx)

	# sf.write('out/out'+str(no) + '.wav',reconstructed_audio,22050,subtype='PCM_24')

	reconstructed_audio = librosa.istft(datax)

	# sf.write('noisydr/nout1'+str(no) + '.wav',reconstructed_audio,22050,subtype='PCM_24')
	# sf.write('out.wav',reconstructed_audio,sample_rate,subtype='PCM_24')



	# reconstructed_audio = librosa.istft(output)

	# sf.write('out.wav',reconstructed_audio,sample_rate,subtype='PCM_24')



# model = torch.load('src/Exp/19250_2_checkpoint.pth.tar')

	# reconstructed_audio = librosa.istft(spectrum)

	# sf.write('out.wav',reconstructed_audio,sample_rate,subtype='PCM_24')
