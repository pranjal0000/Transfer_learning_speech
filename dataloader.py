from dataset import *
import torch

import torch.utils.data.DataLoader as DL
import pickle
from torchvision import transforms
from torch.utils.data import Dataset
####
# TENSOR: Batchsize(N) x Channels(C) x Height(H) x Width(W)
#####
class dataL():
	def __init__(self,batch = 1):
		self.batch = batch

		trainDataset = CommonVoice(root_dir = '/media/pranjal/Seagate Backup Plus Drive/DataASR/Russian/ru', type = 'train')
		self.trainloader = DL(trainDataset, batch_size = self.batch, shuffle = True, num_workers = 2)

		testDataset = CommonVoice(root_dir = '/media/pranjal/Seagate Backup Plus Drive/DataASR/Russian/ru', type = 'test')
		self.testloader = DL(testDataset, batch_size = self.batch, shuffle = True, num_workers = 2)


