from helper.manage_config import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/config.yaml', help="testing configuration file")
args = parser.parse_args()
config = get_config(args.config)

project_root = os.getcwd()
dir_dataset = config['data_dir']
lr_imgs = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
img= lr_imgs[8]

# window_name = 'image sample'
# cv2.imshow(window_name,img)
# cv2.waitKey(0)

# download_pretrained(config)  #only first time
