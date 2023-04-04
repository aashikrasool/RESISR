import os
from argparse import ArgumentParser
from helper.manage_config import *
import numpy as np
import cv2

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/config.yaml', help="testing configuration file")
args = parser.parse_args()
config = get_config(args.config)

project_root = os.getcwd()
dir_dataset = config['data_dir']
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
img = files_img[8]
img = cv2.imread(img)

'''---edsr method---'''


def action1(img, config):
    model_path = config['edsr_model']
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 4)
    img_upscale = sr.upsample(img)
    return img_upscale


'''---espcn method---'''


def action2(img, config):
    model_path = config['espcn_model']
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", 4)
    img_upscale = sr.upsample(img)
    return img_upscale


'''---fsrcnn method---'''


def action3(img, config):
    model_path = config['fsrcnn_model']
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
    img_upscale = sr.upsample(img)
    return img_upscale


'''---lapsrn method---'''


def action4(img, config):
    model_path = config['lapsrn_model']
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("lapsrn", 4)
    img_upscale = sr.upsample(img)
    return img_upscale


'''---increase brightness---'''


def action5(img):
    # define the alpha and beta
    alpha = 0  # Contrast control
    beta = 5  # Brightness control
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


'''---decrease brightness---'''


def action6(img):
    # define the alpha and beta
    alpha = 0  # Contrast control
    beta = -5  # Brightness control
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


'''---increase Contrast---'''


def action7(img):
    # define the alpha and beta
    alpha = 5  # Contrast control
    beta = 0  # Brightness control
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


'''---decrease Contrast---'''


def action8(img):
    # define the alpha and beta
    alpha = -5  # Contrast control
    beta = 0  # Brightness control
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


b = action6(img)
