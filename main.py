import click
from src.pipeline_manager import PipelineManager

from src.logger import Logger

@click.group()
def main():
	pass

@main.command()
@click.option('-m', '--model', help='UNet' , required=False)
def train(model):
    pipeline_manager.train(model)

@main.command()
@click.option('-m', '--model', help='Unet' , required=False)
def test(model):
    pipeline_manager.test(model)

@main.command()
@click.option('-m', '--model', help='UNet' , required=False)
@click.option('-p', '--path', help='MP3 Path' , required=True)
@click.option('-o', '--out_path', help='Save Path' , required=True)
def test_one(model, path, out_path):
    pipeline_manager.test_one(model, path, out_path)

@main.command()
@click.option('-m', '--model', help='UNet' , required=False)
@click.option('-p', '--path', help='MP3 Path' , required=False)
@click.option('-o', '--out_path', help='Save Path' , required=False)
def test_entire_folder(model, path, out_path):

	if path == None and out_path == None:
		pipeline_manager.test_entire_folder(model)
	else:
		pipeline_manager.test_entire_folder(model, path, out_path)


if __name__ == "__main__":

	pipeline_manager = PipelineManager()
	log = Logger()
	log.first()
	
	main()
