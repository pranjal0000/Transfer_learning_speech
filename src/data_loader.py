import torch
import random
import pickle

import os
import numpy as np

import matplotlib.pyplot as plt
from .logger import Logger
import json

from pydub import AudioSegment

import librosa
import soundfile as sf

log = Logger()

class own_DataLoader():

	def __init__(self, config, Type, **kwargs):

		self.config = config
		self.dataset = config['dataset']

		self.Type = Type

		self.batchsize = config[Type]['batch_size']

		# if not config[Type]['loader']['flag']:
		# 	self.loader = self.pil_loader
		# else:
		# 	self.loader = kwargs['loader']

		if self.Type != 'test_one':
			self.get_all_names_refresh()

		# self.all_area = []

	def get_all_names_refresh(self):

		# all_characters_train = ['v', 'i', 'n', 'e', 's', 'y', 'a', 'r', 'd', 'Y', 'L', 'O', '5', '6', 'M', 'c', 'l', 'o', 'A', 'b', 'u', 'q', '2', '8', '0', '9', 'T', 'U', 'Z', 'N', 'E', 'S', 'R', 'B', 'C', 'K', 'F', 'D', 'V', 't', 'k', 'P', 'H', 'G', 'I', '7', 'W', '1', '4', 'h', ' ', '3', "'", 'J', 'Q', 'g', '.', 'f', 'm', '-', 'X', ':', 'j', 'p', '!', 'x', '&', '#', ';', '(', ')', 'w', '_', '?', '/', 'z', '%', '$', '\\', ',', '@', '"', '[', ']', '*', '|', '`', '°', '~', '{', '+', '=', '<', '>']
		# all_characters_test = ['P', 'E', 'R', 'M', 'A', 'N', 'T', '2', 'O', 'D', 'H', 'I', 'C', 'U', 'Y', '6', '5', '1', '4', 'o', 'r', 'v', 'e', 't', '0', '3', 'b', 'y', 'J', 'd', 'i', 's', 'm', 'a', 'n', 'h', 'p', 'G', 'L', 'B', 'W', 'S', '8', '.', 'u', '!', 'V', 'k', 'g', 'l', 'w', 'F', 'K', '7', 'c', '$', ' ', '-', "'", 'x', '9', 'Z', 'X', 'j', ':', 'Q', 'f', ',', 'z', 'q', '@', '&', ';', '/', '#', '(', '+', '?', '%', '*', '"', ')', '[', ']', '_']
		# all_characters = set(all_characters_test) | set(all_characters_train)

		if self.config['Text']:
			self.char_to_onehot_encoding = {}
			self.onehot_encoding_to_char = {}

			all_characters = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '~', '°']
			for no, i in enumerate(all_characters):
				temp_encoding = np.zeros(len(all_characters))
				temp_encoding[no] = 1
				self.char_to_onehot_encoding[i] = temp_encoding
				self.onehot_encoding_to_char[no] = i

			self.texts = []

		with open(self.meta+'/normalisation.pkl', 'rb') as f:

			self.normal = pickle.load(f)
			self.normal['average'][0] /= 255
			self.normal['average'][0] = torch.FloatTensor(self.normal['average'][0].reshape(1, 3, 1, 1))
			self.normal['std'][0] /= (255*255)
			self.normal['std'][0] = np.sqrt(self.normal['std'][0])
			self.normal['std'][0] = torch.FloatTensor(self.normal['std'][0].reshape(1, 3, 1, 1))

		with open(self.meta+'/'+self.Type+'_files.txt', 'r') as f:
			self.images = []
			for i in f:
				if os.path.exists(self.image_root+'/'+i[:-1]+'.png'):
					self.images.append(i[:-1]+'.png')
				elif os.path.exists(self.image_root+'/'+i[:-1]+'.jpg'):
					self.images.append(i[:-1]+'.jpg')
				else:
					print('Error: File not found', i[:-1], self.Type)

		self.annots = [[] for i in range(len(self.images))]

		check = False
		
		for no, i in enumerate(self.images):

			if check:
				print('Error')
			check = True
			
			f = open(self.label_root+'/'+i.split('.')[0]+'.pkl', 'rb')
				
			annot = pickle.load(f)[0]

			hull_annot = []

			if self.config['Text']:

				text_annot = []

				for j in annot:
					original = cv2.contourArea(j[0])
					if original == 0:
						continue
					hull_annot.append(j[0])
					text_annot.append(j[1])
			else:
				for j in annot:
					original = cv2.contourArea(j)
					#Can change threshold
					if original == 0:
						continue
					# hull_annot.append(cv2.convexHull(j, False))
					hull_annot.append(j)

			# if i == 'train1_img_171.jpg':
			# 	print(annot, np.array(hull_annot))
			# 	print(no)
			# 	exit(0)

			self.annots[no] = np.array(hull_annot)

			# print(i, np.array(hull_annot))
			check = False
			f.close()

		if self.Type != 'train':

			self.start = 0



	def get_link(self, contours, r_h, image_shape, prev_contour):

		link = np.zeros([self.image_size, self.image_size, 8])
		target = np.zeros([self.image_size, self.image_size]).astype(np.uint8)
		weight = np.zeros([self.image_size, self.image_size])

		"""
		Up-left, Up, Up-right, right, Down-right, Down, Down-left, left
		"""

		for no, contour in enumerate(contours):

			dummy_image = np.zeros([self.image_size, self.image_size]).astype(np.uint8)
			cv2.drawContours(dummy_image, [contour], -1, 1, cv2.FILLED)
			#intersection
			intersection = np.logical_and(dummy_image, target).astype(np.uint8)
			dummy_image -= intersection
			target = target + dummy_image - intersection

			# TODO Can be made faster - and with only small portion
			
			# sum_ = np.sum(dummy_image)
			sum_ = cv2.contourArea(contour)
			# if sum_ == 0:
			# 	print(r_h, image_shape)
			# 	print(prev_contour[no])
			# 	print(contour)
			# 	plt.imshow(dummy_image)
			# 	plt.show()
			weight = weight + dummy_image/sum_

			row, column = np.where(dummy_image == 1)

			for i, j in zip(row, column):

				if i>0 and j>0 and dummy_image[i-1, j-1] == 1:
					link[i, j, 0] = 1
				if i>0 and dummy_image[i-1, j] == 1:
					link[i, j, 1] = 1
				if i>0 and j<self.image_size-1 and dummy_image[i-1, j+1] == 1:
					link[i, j, 2] = 1
				if j<self.image_size-1 and dummy_image[i, j+1] == 1:
					link[i, j, 3] = 1
				if i<self.image_size-1 and j<self.image_size-1 and dummy_image[i+1, j+1] == 1:
					link[i, j, 4] = 1
				if i<self.image_size-1 and dummy_image[i+1, j] == 1:
					link[i, j, 5] = 1
				if i<self.image_size-1 and j>0 and dummy_image[i+1, j-1] == 1:
					link[i, j, 6] = 1
				if j>0 and dummy_image[i, j-1] == 1:
					link[i, j, 7] = 1

		# weight = target*weight
		if len(contours) !=0:
			weight = weight*np.where(weight!=0)[0].shape[0]/len(contours)

		# for i in range(8):

		# 	link[:, :, i] *= target

		return (link*255).astype(np.uint8), (target*255).astype(np.uint8), weight

	def __getitem__(self, index):

		if index>=self.__len__():
			raise IndexError

		img = torch.FloatTensor(np.zeros([self.batchsize, 3, self.image_size, self.image_size]))
		target = torch.FloatTensor(np.zeros([self.batchsize, 1, self.image_size, self.image_size]))
		link = torch.FloatTensor(np.zeros([self.batchsize, 8, self.image_size, self.image_size]))
		weight = torch.FloatTensor(np.zeros([self.batchsize, 1, self.image_size, self.image_size]))
		# big_target = torch.FloatTensor(np.zeros([self.batchsize, 1, self.image_size, self.image_size]))

		path = []
		contour = []

		random_images = np.random.choice(len(self.images), self.batchsize)
		# random_images = [138]

		for no, i in enumerate(random_images):
			image_new, link_new, target_new, weight_new = self.aspect_resize(self.loader(self.image_root+'/'+self.images[i]), self.annots[i].copy())#, big_target_new
			img[no] = (self.transform(image_new) - self.normal['average'][0])/self.normal['std'][0]
			target[no] = self.target_transform(target_new)
			link[no] = self.target_transform(link_new)
			weight[no] = torch.FloatTensor(weight_new).unsqueeze(0)
			# big_target[no] = self.target_transform(big_target_new)
			contour.append(self.annots[i])
			path.append(self.images[i])

		return img, link, target, weight, contour, path#, big_target

	def __len__(self):

		if self.Type == 'train':

			return len(self.images)

		else:

			return 100/self.batchsize


	def convert_float_tensor(self, x, y):

		x = torch.FloatTensor(x.transpose(0, 3, 1, 2))
		y = torch.FloatTensor(y.transpose(0, 3, 1, 2))

		return x, y

	def convert_float_tensor_one(self, x):

		x = torch.FloatTensor(x.transpose(0, 3, 1, 2))

		return x
