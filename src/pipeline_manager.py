import yaml
import os
from read_yaml import read_yaml
from logger import Logger

log = Logger()

class PipelineManager():

	def __init__(self):
		self.config_file = read_yaml()
		for i in self.config_file['dir']:
			if not os.path.exists(self.config_file['dir'][i]):
				os.mkdir(self.config_file['dir'][i])

	def train(self, model_name):
		train(model_name)

	def test(self, model_name):
		test(model_name)

	# def test_one(self, model_name, path, out_path):
	# 	test_one_image(model_name, path, out_path)

	# def test_entire_folder(self, model_name, path, out_path):
	# 	test_entire_folder(model_name, path, out_path)

def train(model_name):

	from dl_model import dl_model
	
	if model_name == 'UNet' or model_name == None:
		driver = dl_model('UNet', mode = 'train')
		driver.train_model()
	else:
		log.info("Not yet implemented")

def test(model_name):

	from dl_model import dl_model

	if model_name == 'UNet' or model_name == None:
		driver = dl_model('UNet', mode='test')
		driver.test_model()
	else:
		print("Not yet implemented")

# def test_one_image( model_name, path, out_path):

# 	from dl_model import dl_model

# 	if model_name == 'UNet' or model_name == None:
# 		driver = dl_model('UNet', mode='test_one')
# 		driver.test_one_image(path, out_path)
# 	else:
# 		print("Not yet implemented")

# def test_entire_folder(model_name, path, out_path):

# 	print(path, out_path)

# 	from dl_model import dl_model

# 	if model_name == 'UNet':

# 		if not os.path.exists(out_path):
# 			os.mkdir(out_path)
# 		driver = dl_model(model_name, mode='test_one')
# 		gen(driver, path, out_path)
