import os
from argparse import ArgumentParser
from helper.manage_config import *
import numpy as np

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/config.yaml', help="testing configuration file")
args = parser.parse_args()
config = get_config(args.config)

project_root = os.getcwd()
dir_dataset = config['data_dir']
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]

'''---edsr method---'''
def action1(img, config):
    img_upscale = get_upscale_images(img, config['edsr_model'], "edsr", 4)
    return img_upscale


'''---espcn method---'''
def action2(img, config):
    img_upscale = get_upscale_images(img, config['espcn_model'], "espcn", 4)
    return img_upscale


'''---fsrcnn method---'''

def action3(img, config):
    img_upscale = get_upscale_images(img, config['fsrcnn_model'], "fsrcnn", 4)
    return img_upscale


'''---lapsrn method---'''


def action4(img, config):
    img_upscale = get_upscale_images(img, config['lapsrn_model'], "lapsrn", 4)
    return img_upscale


'''---increase brightness---'''


def action5(img, b):
    degenerate = np.zeros(img.shape)

    img = b * img + (1 - b) * degenerate
    img = np.clip(img, 0, 1)
    return img
