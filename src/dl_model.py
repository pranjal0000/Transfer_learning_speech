import sys
# from torchvision import transforms
import torch.utils.data as data
import torch
from torch.autograd import Variable
import os
import numpy as np
import random
import matplotlib.pyplot as plt
# import cv2
import shutil
# from scipy.misc import imresize
import time
import json

from .read_yaml import read_yaml
from .logger import Logger
# from .model import unet_model

from .dataloader import *
from .model.unet_model import UNet

log = Logger()
####
# TENSOR: Batchsize(N) x Channels(C) x Height(H) x Width(W)
#####

class dl_model():

	def __init__(self, model, mode):
		
		self.config = self.get_config()
		self.seed()
		self.model = self.get_model(model)
		self.mode = mode

		# if self.mode != 'test_one':
		# 	self.train_data_loader = own_DataLoader(self.config, Type='train')#, transform=self.train_transform, target_transform = self.target_transform)
		# 	self.test_data_loader = own_DataLoader(self.config, Type='test')#, transform=self.test_transform, target_transform = self.target_transform)
		# else:
		# 	self.test_data_loader = own_DataLoader(self.config, Type='test_one')#, transform=self.test_transform, target_transform = self.target_transform)

		self.dataL = dataL(self.config['batch_size'])

		if mode == 'train' or mode == 'test':

			self.cuda = self.config['cuda'] #and torch.cuda.is_available()

			self.plot_training = {'Loss' : []}	
			self.plot_testing = {'Loss' : []}

			self.training_info = {'Loss': []}
			# self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
			# self.testing_info = {'Acc': 0, 'Loss': 0, 'Count': 0, 'Keep_log': False}
			self.testing_info = {'Loss': 0}
			
			self.model_best = {'Loss': sys.float_info.max}

			if self.cuda:
				self.model.cuda()

			self.epoch_start = 0
			self.start_no = 0

			if self.config['PreTrained_model']['check'] == True :

				self.model_best = torch.load(self.config['PreTrained_model']['checkpoint_best'])['best']
				if mode == 'train':
					self.epoch_start, self.training_info = self.model.load(self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'])
				else:
					self.epoch_start, self.training_info = self.model.load(self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'])

				# self.start_no = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[0]) ########
				self.start_no = 0
				# self.epoch_start = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[1]) ########
				self.epoch_start = 0

				# if mode == 'train':

					# self.plot_training['Loss'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_training_loss.npy'))
					# self.plot_training['Acc'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_training_accuracy.npy'))
					# self.plot_testing['Acc'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_testing_accuracy.npy'))
					# self.plot_testing['Loss'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_testing_loss.npy'))
				
				log.info('Loaded the model')
	
	def get_model(self, model):

		if model == 'UNet':
			log.info("UNet")

			return UNet(self.config)
		else:
			log.info("Can't find model")

	def get_config(self):

		return read_yaml()

	def seed(self):

		np.random.seed(self.config['seed'])
		random.seed(self.config['seed'])
		torch.manual_seed(self.config['seed'])
		torch.cuda.manual_seed(self.config['seed'])
		torch.backends.cudnn.deterministic = True

	def __str__(self):

		return str(self.config)


	def start_training(self):

		self.model.requires_grad = True

		self.model.train()

		self.model.opt.zero_grad()

	def start_testing(self):

		self.model.requires_grad = False

		self.model.eval()

	def show_graph(self, epoch, no):

		plt.clf()

		plt.subplot(211)
		plt.plot(self.plot_training['Loss'], color='red')
		plt.plot(self.plot_testing['Loss'], color='blue')
		plt.title('Loss, Red:Train, Blue:Testing')
		# plt.subplot(212)
		# plt.plot(self.plot_training['Acc'], color='red')
		# plt.plot(self.plot_testing['Acc'], color='blue')
		# #plt.pause(0.1)
		plt.savefig(self.config['dir']['Plots']+'/'+str(epoch)+'_'+str(no)+'.png')
		plt.clf()

	def test_module(self, epoch_i, no):

		self.test_model()

		self.plot_training['Loss'].append(np.mean(self.training_info['Loss']))
		# self.plot_training['Acc'].append(np.mean(self.training_info['Acc']))

		self.show_graph(epoch_i, no)
		self.start_training()

	def train_model(self):

		try:

			self.start_training()
				
			for epoch_i in range(self.epoch_start, self.config['epoch']+1):

				log.info('Starting epoch : ', epoch_i)

				for no, data in enumerate(self.dataL.trainLoader):

					if self.cuda:

						data = data.to("cuda")

					data_t = data[:,0:2,:,:]
					data_exp = data[:,2:4,:,:]

					output = self.model(data_t)

					loss = self.model.lossf(output,data_exp)
					self.training_info['Loss'].append(loss.data.cpu())

					loss.backward()			


					self.model.opt.step()
					self.model.opt.zero_grad()

					# if (self.start_no + no)%self.config['print_log_steps'] == 0 and (self.start_no + no)!=0 and (self.start_no + no) > 20:
					# 	log.info()
					# 	# self.model.print_info(self.training_info)

					if (self.start_no + no)%self.config['log_interval_steps'] == 0 and (self.start_no + no)!=0:

						self.model.save(no=(self.start_no + no), epoch_i=epoch_i, info = self.training_info, best=self.model_best, is_best = False)


					if (self.start_no + no)%self.config['test_now']==0 and (self.start_no + no)!=0:

						torch.cuda.synchronize()
						torch.cuda.empty_cache()
						self.test_module(epoch_i, self.start_no + no)
						torch.cuda.synchronize()
						torch.cuda.empty_cache()

						# np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_testing_accuracy.npy', self.plot_testing['Acc'])
						np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_testing_loss.npy', self.plot_testing['Loss'])
						# np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_training_accuracy.npy', self.plot_training['Acc'])
						np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_training_loss.npy', self.plot_training['Loss'])

				log.info()
				self.start_no = 0
				# self.training_info = {'Loss': []}
				# self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
				# self.scheduler.step()

			return True

		except KeyboardInterrupt:

			return False

	def test_model(self):

		log.info('Testing Mode')

		try:

			self.start_testing()
			loss_cum = 0
			num_loss = 0
			with torch.no_grad():

				# start = time.time()
				
				for no, data in enumerate(self.dataL.testLoader):

					# fps['images'] += data.shape[0]

					if self.cuda:

						data = data.to("cuda")
					
					data_t = data[:,0:2,:,:]
					data_exp = data[:,2:4,:,:]

					
					output = self.model(data_t)
					loss = self.model.lossf(output,data_exp)
					loss_cum = loss_cum + float(loss)
					num_loss = num_loss + 1
					# fps['time_taken'] += time.time() - start

					# del data_t, data_exp, output, loss

					# start = time.time()

			log.info('Test Results\n\n', )
			self.testing_info['Loss'] = float(loss_cum/num_loss)

			if self.mode =='train':

				if self.testing_info['Loss'] < self.model_best['Loss']:

					log.info("New best model found")
					self.model_best['Loss'] = self.testing_info['Loss']
					
					self.model.save(no=0, epoch_i=0, info = self.testing_info, best=self.model_best, is_best=True)

				# self.plot_testing['Acc'].append(self.testing_info['Acc'])
				self.plot_testing['Loss'].append(self.testing_info['Loss'])

			log.info('\nTesting Completed successfully: Avg Loss = ', self.testing_info['Loss'])

			# self.testing_info = {'Acc': 0, 'Loss': 0, 'Count': 0, 'Keep_log': False}
			self.testing_info = {'Loss': 0}

			return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False

	
