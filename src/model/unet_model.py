""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
	def __init__(self, n_channels=2, n_classes=2, bilinear=True):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		self.down4 = Down(512, 512)
		self.up1 = Up(1024, 256, bilinear)
		self.up2 = Up(512, 128, bilinear)
		self.up3 = Up(256, 64, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, n_classes)

		if self.config['optimizer'] == 'Adam':
			log.info('Using Adam optimizer')
			self.opt = optim.Adam(self.parameters(), lr=config['lr'])
		elif self.config['optimizer'] == 'SGD':
			log.info('Using SGD optimizer')
			self.opt = optim.SGD(self.parameters(), lr=config['lr'], momentum=0.9)
		elif self.config['optimizer'] == 'AdaDelta':
			log.info('Using AdaDelta optimizer')
			self.opt = optim.Adadelta(self.parameters(), lr=config['lr'])

		log.info('Using MSE')
		self.lossf = torch.nn.MSELoss()



	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		# if self.n_classes > 1:
		#     return F.softmax(x, dim=1)
		# else:
		#     return torch.sigmoid(x)

		return x

	def save(self, no, epoch_i, info, is_best=False, filename='checkpoint.pth.tar', best={}):


		if is_best:
			torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best},self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output_Best']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		else:
			torch.save({'epoch': epoch_i,
					'state_dict': self.state_dict(),
					'optimizer': self.opt.state_dict(),
					'seed': self.config['seed'],
					'best': best},self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename)
			torch.save(info, self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename)
		
		# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+filename, 'model_best.pth.tar')
		# shutil.copyfile(self.config['dir']['Model_Output']+'/'+str(no)+'_'+str(epoch_i)+'_'+'info_'+filename, 'info_model_best.pth.tar')

	def load(self, path, path_info):

		checkpoint = torch.load(path)

		self.load_state_dict(checkpoint['state_dict'])

		# if not self.config['optimizer_new']:
		self.opt.load_state_dict(checkpoint['optimizer'])
		
		return checkpoint['epoch'], torch.load(path_info)



	# def loss(self, pred, target, info):


	# 	# b, ch, h, w = pred.size()

	# 	# pred = pred.transpose(1, 3).contiguous().view(b, w*h, ch)
	# 	# target = target.transpose(1, 3).contiguous().view(b, h*w, ch//2).long()

	# 	# print("Here:",target.size())

	# 	loss_c = self.lossf(pred, target)

	# 	# pred1 = pred.view(b*w*h, ch)
	# 	# target1 = target.view(b*h*w, ch//2)

	# 	if info['Keep_log']:

	# 		# info['Acc'].append(self.accuracy(pred1, target1, True))
	# 		info['Loss'].append(loss_c.data.cpu().numpy())

	# 	else:

	# 		# acc = self.accuracy(pred, target, True)
	# 		# info['Acc'] = (acc + info['Count']*info['Acc'])/(info['Count']+1)
	# 		info['Loss'] = (loss_c.data.cpu().numpy() + info['Count']*info['Loss'])/(info['Count']+1)
	# 		# info['Acc_indi'] = (indi_acc + info['Count']*info['Acc_indi'])/(info['Count']+1)

	# 	info['Count'] += 1

	# 	return loss_c



