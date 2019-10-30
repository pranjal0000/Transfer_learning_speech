import sys
from torchvision import transforms
import torch.utils.data as data
import torch
from torch.autograd import Variable
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import shutil
from scipy.misc import imresize
import time
import json

from read_yaml import read_yaml
from logger import Logger
from model import full_Conn

import threading
from .data_loader import own_DataLoader

log = Logger()

class dl_model():

	def __init__(self, model, mode = 'train', target_transform=None, train_transform=None, test_transform=None):
		
		self.config = self.get_config()
		self.seed()
		self.model = self.get_model(model)
		self.mode = mode
		self.get_transforms(target_transform, train_transform, test_transform)

		if mode != 'test_one':
			self.train_data_loader = own_DataLoader(self.config, Type='train', transform=self.train_transform, target_transform = self.target_transform)
			self.test_data_loader = own_DataLoader(self.config, Type='test', transform=self.test_transform, target_transform = self.target_transform)
		else:
			self.test_data_loader = own_DataLoader(self.config, Type='test_one', transform=self.test_transform, target_transform = self.target_transform)

		if mode == 'train' or mode == 'test':

			self.cuda = self.config['cuda'] and torch.cuda.is_available()

			self.plot_training = {'Loss' : [], 'Acc' : []}	
			self.plot_testing = {'Loss' : [], 'Acc' : []}

			self.training_info = {'Loss': [], 'Seg_loss':[], 'Link_Loss':[], 'Class_balanced_Loss':[], 'Reco_Loss':[], 'Acc': [], 'Keep_log': True, 'Count':0}
			# self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
			# self.testing_info = {'Acc': 0, 'Loss': 0, 'Count': 0, 'Keep_log': False}
			self.testing_info = {'Acc': 0, 'Loss': 0, 'Seg_loss':0, 'Link_Loss':0, 'Class_balanced_Loss':0, 'Reco_Loss':0, 'Count': 0, 'Keep_log': False}
			
			self.model_best = {'Loss': sys.float_info.max, 'Acc': 0.0, 'Acc_indi': np.zeros(self.config['n_classes']+1)}

			if self.cuda:
				self.model.cuda()

			self.epoch_start = 0
			self.start_no = 0

			if self.config['PreTrained_model']['check'] == True or mode=='testing':

				self.model_best = torch.load(self.config['PreTrained_model']['checkpoint_best'])['best']
				if mode == 'train':
					self.epoch_start, self.training_info = self.model.load(self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'], mode)
				else:
					self.epoch_start, self.training_info = self.model.load(self.config['PreTrained_model']['checkpoint'], self.config['PreTrained_model']['checkpoint_info'], mode)

				self.start_no = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[0])
				self.epoch_start = int(self.config['PreTrained_model']['checkpoint'].split('/')[-1].split('_')[1])

				if mode == 'train':

					self.plot_training['Loss'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_training_loss.npy'))
					self.plot_training['Acc'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_training_accuracy.npy'))
					self.plot_testing['Acc'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_testing_accuracy.npy'))
					self.plot_testing['Loss'] = list(np.load(self.config['dir']['Plots']+'/'+str(self.epoch_start)+'_'+str(self.start_no)+'_testing_loss.npy'))
				
				log.info('Loaded the model')

		elif mode == 'test_one':

			self.cuda = self.config['cuda'] and torch.cuda.is_available()
			self.get_transforms(target_transform, train_transform, test_transform)

			if self.cuda:
				self.model.cuda()

			self.model_best = torch.load(self.config['PreTrained_model']['checkpoint_best'])['best']
			self.epoch_start, self.training_info = self.model.load(self.config['PreTrained_model']['checkpoint_best'], self.config['PreTrained_model']['checkpoint_best_info'], mode)

			self.start_no = int(self.config['PreTrained_model']['checkpoint_best'].split('/')[-1].split('_')[0])
			self.epoch_start = int(self.config['PreTrained_model']['checkpoint_best'].split('/')[-1].split('_')[1])

			log.info('Loaded the model')
		else:

			log.info('Ensembling')
	
	def get_model(self, model):

		if model == 'ResNet_UNet':
			log.info("RESNET_UNET")
			channels, classes = self.config['n_channels'], self.config['n_classes'] # +1 for Background
			return UNetWithResnet50Encoder(config=self.config)
		else:
			log.info("Can't find model")

	def get_transforms(self, target_transform=None, train_transform=None, test_transform=None):

		if self.config['train']['transform'] == False or train_transform == None:
			self.train_transform = transforms.Compose([
											transforms.ColorJitter(brightness=self.config['augmentation']['brightness'], contrast=self.config['augmentation']['contrast'], saturation=self.config['augmentation']['saturation'], hue=self.config['augmentation']['hue']),
											transforms.ToTensor(),
											])
		else:
			self.train_transform = train_transform

		if self.config['test']['transform'] == False or test_transform == None:
			self.test_transform = transforms.Compose([
											 transforms.ToTensor(),
											 ])
		else:
			self.test_transform = test_transform
		
		if self.config['target_transform'] == False or target_transform == None:

			self.target_transform = transforms.Compose([
											 transforms.ToTensor(),
											 ])
		else:
			self.target_transform = target_transform

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

	def convert_argmax_to_channels(self, temp, masks):

		t_show = np.zeros([temp.shape[0], temp.shape[1], masks]).astype(np.uint8)

		for __i in range(t_show.shape[0]):
			for __j in range(t_show.shape[1]):
				t_show[__i, __j, temp[__i, __j]] = 255

		return t_show

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
		plt.title('Upper Plot: Loss, Red:train, Blue:Testing\nLower Plot: Accuracy, Red:train, Blue:Testing')
		plt.subplot(212)
		plt.plot(self.plot_training['Acc'], color='red')
		plt.plot(self.plot_testing['Acc'], color='blue')
		#plt.pause(0.1)
		plt.savefig(self.config['dir']['Plots']+'/'+str(epoch)+'_'+str(no)+'.png')
		plt.clf()

	def test_module(self, epoch_i, no):

		self.test_model()

		self.plot_training['Loss'].append(np.mean(self.training_info['Loss']))
		self.plot_training['Acc'].append(np.mean(self.training_info['Acc']))

		self.show_graph(epoch_i, no)
		self.start_training()

	def train_model(self):
		
		##############TODO: CONVERT RECURSION TO FOR LOOP########################

		try:

			self.start_training()
			sys.setrecursionlimit(1000000)
			threading.stack_size(0x20000000)
			t = None

			# self.scheduler = torch.optim.lr_scheduler.StepLR(self.model.opt, 1, gamma=(self.config['min_lr']/self.config['lr'])**(1/self.config['epoch']), last_epoch=-1)	
			for epoch_i in range(self.epoch_start, self.config['epoch']+1):

				log.info('Starting epoch : ', epoch_i)

				for no, (data, link, target, weight, contour, path) in enumerate(self.train_data_loader):

					# print(no)

					if t!=None:
						t.join()
						t = None
					# show_t = np.zeros([data.shape[2], data.shape[3], 3])
					# print(target[0].data.cpu().numpy().transpose(1, 2, 0).shape)
					# show_t[:, :, 0] = target[0].data.cpu().numpy().transpose(1, 2, 0)[:, :, 0]
					# plt.imshow(np.concatenate((data[0].data.cpu().numpy().transpose(1, 2, 0), show_t), axis=1))
					# plt.show()

					if self.cuda:

						data, target, link, weight = data.cuda(), target.cuda(), link.cuda(), weight.cuda()

					output = self.model(data)

					if (self.start_no + no)%self.config['show_some']==0 and (self.start_no + no)!=0:

						predicted_link = output[0, 2:, :, :].data.cpu().numpy().transpose(1, 2, 0)
						predicted_target = output[0, 0:2, :, :].data.cpu().numpy().transpose(1, 2, 0)
						
						t = threading.Thread(target=get_connected_components, args=(predicted_link, predicted_target, data[0].data.cpu().numpy().transpose(1, 2, 0), target[0, 0].data.cpu().numpy(), self.config, True, None, None))
						t.start()		
						# t.join()

						# images = np.concatenate((data[0:min(2, self.train_data_loader.batchsize)].data.cpu().numpy().transpose(0, 2, 3, 1)), axis=0)
						
						# soft_output = np.concatenate(np.argmax(output[0:min(2, self.train_data_loader.batchsize), 0:2, :, :].data.cpu().numpy(), axis=1), axis=0)
						# soft_output_final = np.zeros([soft_output.shape[0], soft_output.shape[1], 3])
						# soft_output_final[:, :, 0] = soft_output

						# show_target = np.concatenate((target[0:min(2, self.train_data_loader.batchsize), 0, :, :].data.cpu().numpy()), axis=0)
						# show_target_final = np.zeros([show_target.shape[0], show_target.shape[1], 3])
						# show_target_final[:, :, 0] = show_target
						
						# plt.clf()

						# lets_show = np.concatenate([images, soft_output_final, show_target_final], axis=1)

						# plt.imsave(self.config['dir']['Exp']+'/output_train.png', lets_show)
						# plt.imshow(lets_show)
						# plt.pause(0.1)

					loss = self.model.loss(data.data.cpu().numpy(), output, target, link, weight, contour, self.training_info)

					loss.backward()

					if (self.start_no + no)%self.config['update_config']==0 and (self.start_no + no)!=0:

						prev_config = self.config

						self.config = self.get_config()
						self.train_data_loader.config = self.config
						self.test_data_loader.config = self.config
						self.model.config = self.config

						if self.config['lr']!=prev_config['lr']:
							log.info('Learning Rate Changed from ', prev_config['lr'], ' to ', self.config['lr'])				

					if (self.start_no + no)%self.config['cummulative_batch_steps']==0:

						self.model.opt.step()
						self.model.opt.zero_grad()

					if (self.start_no + no) == len(self.train_data_loader) - 1:
						break

					if (self.start_no + no)%self.config['print_log_steps'] == 0 and (self.start_no + no)!=0 and (self.start_no + no) > 100:
						log.info()
						self.model.print_info(self.training_info)

					if (self.start_no + no)%self.config['log_interval_steps'] == 0 and (self.start_no + no)!=0:

						self.model.save(no=(self.start_no + no), epoch_i=epoch_i, info = self.training_info, best=self.model_best, is_best = False)

					del data, target, link, weight, output, loss

					if (self.start_no + no)%self.config['test_now']==0 and (self.start_no + no)!=0:

						torch.cuda.synchronize()
						torch.cuda.empty_cache()
						self.test_module(epoch_i, self.start_no + no)
						torch.cuda.synchronize()
						torch.cuda.empty_cache()

						np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_testing_accuracy.npy', self.plot_testing['Acc'])
						np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_testing_loss.npy', self.plot_testing['Loss'])
						np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_training_accuracy.npy', self.plot_training['Acc'])
						np.save(self.config['dir']['Plots']+'/'+str(epoch_i)+'_'+str((self.start_no + no))+'_training_loss.npy', self.plot_training['Loss'])

				log.info()
				self.start_no = 0
				self.training_info = {'Loss': [], 'Seg_loss':[], 'Link_Loss':[], 'Class_balanced_Loss':[], 'Reco_Loss':[], 'Acc': [], 'Keep_log': True, 'Count':0}
				# self.training_info = {'Loss': [], 'Acc': [], 'Keep_log': True, 'Count':0}
				# self.scheduler.step()

			return True

		except KeyboardInterrupt:

			return False

	def test_model(self):

		log.info('Testing Mode')

		try:

			self.start_testing()

			with torch.no_grad():

				fps = {'images': 0, 'time_taken': 0}

				indi_acc = {'spot': [], 'patch': [], 'wrinkle': []}

				start = time.time()

				for no, (data, link, target, weight, contour, path) in enumerate(self.test_data_loader):

					fps['images'] += data.shape[0]

					if self.cuda:

						data, target, link, weight = data.cuda(), target.cuda(), link.cuda(), weight.cuda()
					
					output = self.model(data)
					loss = self.model.loss(data.data.cpu().numpy(), output, target, link, weight, contour, self.testing_info)

					fps['time_taken'] += time.time() - start

					if no%10 == 0:

						predicted_link = output[0, 2:, :, :].data.cpu().numpy().transpose(1, 2, 0)
						predicted_target = output[0, 0:2, :, :].data.cpu().numpy().transpose(1, 2, 0)
						
						t = threading.Thread(target=get_connected_components, args=(predicted_link, predicted_target, data[0].data.cpu().numpy().transpose(1, 2, 0), target[0, 0].data.cpu().numpy(), self.config, True, None, None))
						t.start()
						t.join()

						# images = np.concatenate((data[0:3].data.cpu().numpy().transpose(0, 2, 3, 1)), axis=0)
						
						# soft_output = np.concatenate(np.argmax(output[0:3, 0:2, :, :].data.cpu().numpy(), axis=1), axis=0)
						# soft_output_final = np.zeros([soft_output.shape[0], soft_output.shape[1], 3])
						# soft_output_final[:, :, 0] = soft_output

						# show_target = np.concatenate((target[0:3, 0, :, :].data.cpu().numpy()), axis=0)
						# show_target_final = np.zeros([show_target.shape[0], show_target.shape[1], 3])
						# show_target_final[:, :, 0] = show_target

						# lets_show = np.concatenate([images, soft_output_final, show_target_final], axis=1)
						# plt.clf()
						# plt.imshow(lets_show)

						# plt.imsave(self.config['dir']['Output']+'/test'+str(no)+'.png', lets_show)
						# plt.pause(0.1)
					del data, target, link, weight, output, loss

					start = time.time()

			log.info('Test Results\n\n', )

			if self.mode =='train':

				if self.testing_info['Acc'] > self.model_best['Acc']:

					log.info("New best model found")
					self.model_best['Acc'] = self.testing_info['Acc']
					
					self.model.save(no=0, epoch_i=0, info = self.testing_info, best=self.model_best, is_best=True)

				self.plot_testing['Acc'].append(self.testing_info['Acc'])
				self.plot_testing['Loss'].append(self.testing_info['Loss'])

			log.info('\nTesting Completed successfully: Average accuracy = ', self.testing_info['Acc'], 'Average FPS = ', fps['images']/fps['time_taken'])

			# self.testing_info = {'Acc': 0, 'Loss': 0, 'Count': 0, 'Keep_log': False}
			self.testing_info = {'Acc': 0, 'Loss': 0, 'Seg_loss':0, 'Link_Loss':0, 'Class_balanced_Loss':0, 'Reco_Loss':0, 'Count': 0, 'Keep_log': False}

			return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False

	def test_one_image(self, path, out_path):

		try:

			self.start_testing()
			from scipy.misc import imresize

			sys.setrecursionlimit(1000000)
			threading.stack_size(0x20000000)

			low_bound = 1536+128*2
			up_bound = 1536+128*3
			step = 128

			size = np.arange(low_bound, up_bound, step)

			avg_output = torch.FloatTensor(np.zeros([1, 18, up_bound, up_bound]))

			with torch.no_grad():

				self.test_data_loader.image_size = up_bound
				image = self.test_data_loader.loader(path)
				image, image_shape, r_h = self.test_data_loader._aspect_resize(image)
				image = self.test_data_loader.transform(image).unsqueeze(0)

				for s in size:

					self.test_data_loader.image_size = s
					image_temp, image_shape, r_h = self.test_data_loader._aspect_resize(self.test_data_loader.loader(path))
					image_temp = self.test_data_loader.transform(image_temp).unsqueeze(0)

					if self.cuda:
						image_temp = image_temp.cuda()

					output = self.model(image_temp)

					for i in range(18):
						temp_image = output[0, i, :, :].data.cpu().numpy()
						min_, max_ = np.min(temp_image), np.max(temp_image)
						temp_image = (temp_image - min_)/(max_ - min_)
						avg_output[0, i, :, :] += torch.FloatTensor((imresize(temp_image, (up_bound, up_bound))/255)*(max_ - min_) + min_)

					del image_temp, output

				avg_output/=size.shape[0]

				cont_output = self.model.output(avg_output[:, 0:2, :, :]).data.cpu().numpy()
				cont_output = cont_output[0, 1, :, :][:, :, None]

				# plt.imshow(np.multiply(image[0].data.cpu().numpy().transpose(1, 2, 0), cont_output))
				# plt.show()

				# plt.imsave(out_path+'continuous_output.png', np.multiply(image[0].data.cpu().numpy().transpose(1, 2, 0), cont_output))

				predicted_link = avg_output[0, 2:, :, :].data.cpu().numpy().transpose(1, 2, 0)
				predicted_target = avg_output[0, 0:2, :, :].data.cpu().numpy().transpose(1, 2, 0)
				
				t = threading.Thread(target=get_connected_components, args=(predicted_link, predicted_target, image[0].data.cpu().numpy().transpose(1, 2, 0), None, self.config, True, None, '.'.join(out_path.split('.')[:-1])))
				t.start()	
				t.join()

				# for i in range(10):
				# 	thresh = i/10
				# 	new = cont_output.copy()
				# 	new[new<thresh] = 0

				# 	plt.imshow(np.multiply(image[0].data.cpu().numpy().transpose(1, 2, 0), new))
				# 	plt.show()

				# print(cont_output.shape)

				# exit(0)

				soft_output = np.argmax(avg_output[0, :2, :, :].data.cpu().numpy(), axis=0)
				soft_output_final = np.zeros([soft_output.shape[0], soft_output.shape[1], 3])
				soft_output_final[:, :, 0] = soft_output

				image = image[0].data.cpu().numpy().transpose(1, 2, 0)

				# plt.imsave(out_path+'argmax.png', np.concatenate((image, soft_output_final), axis=1))	

				return True

		except KeyboardInterrupt:

			log.info('Testing Interrupted')

			return False