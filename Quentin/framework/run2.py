import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import OrderedDict
import shutil

import torch.onnx as onnx
import torchvision.models as models

from pathlib import Path
import random

import numpy as np
from tifffile import TiffFile
import matplotlib.pyplot as plt
import colorsys
import PIL
from PIL import Image 
from tqdm import tqdm
import matplotlib.patches as mpatches
import time
import warnings
warnings.filterwarnings('ignore')

from framework.dataset import LandCoverData as LCD

import wandb

from Quentin.framework.models import NeuralNetwork, NeuralNetworkSnow
from Quentin.framework.dataset import CustomDataset, LandCoverData, CustomTestDataset, CustomDatasetSnow
from Quentin.framework.train2 import train_model_Unet, model_unet

def Exemple(config, device, model):

    DATA_FOLDER_STR = 'dataset'
    DATA_FOLDER = Path(DATA_FOLDER_STR).expanduser()
    # path to the unzipped dataset: contains directories train/ and test/
    DATASET_FOLDER = DATA_FOLDER
    # get all train images and masks
    train_images_paths = sorted(list(DATASET_FOLDER.glob('train/images/*.tif')))
    train_masks_paths = sorted(list(DATASET_FOLDER.glob('train/masks/*.tif')))
    # get all test images
    test_images_paths = sorted(list(DATASET_FOLDER.glob('test/images/*.tif')))
    train_size = config['train_size']
    train_image_paths = train_images_paths[:int(config['train_size'])]
    test_image_paths = train_images_paths[int(config['train_size']):]

    train_mask_paths = train_masks_paths[:int(config['train_size'])]
    test_mask_paths = train_masks_paths[int(config['train_size']):]
    #ajout image de neige en plus dans la phase de test
    snow = [3449,3450,3837,3560,3671,3438,3484,3518,3519,3531,3533,3805,3836,4000,4001,4047,3496,3583,3581,3629,3582,3628,3882,3672,3627,3541,3804,3540,3763,3803,3808,3545,3544,4046,3542, 3589,3590,4091, 3656,3702,3452,3498]
    for i in range(np.shape(snow)[0]):
        test_image_paths += sorted(list(DATASET_FOLDER.glob(f'train/images/{snow[i]}.tif')))
        test_mask_paths += sorted(list(DATASET_FOLDER.glob(f'train/masks/{snow[i]}.tif')))
    #création du train
    train_dataset = CustomDataset(train_image_paths, train_mask_paths, train=True)
    trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    #création du test du train
    test_dataset = CustomDataset(test_image_paths, test_mask_paths, train=False)
    testloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    if config['model_load']:
        model.load_state_dict(torch.load(f"{config['model_name']}.pth"))
        print("model chargé")
    
    if config['test']:
        mask_cal = train_model_Unet(model, config, trainloader, testloader)
        return mask_cal
    else : 
        train_model_Unet(model, config, trainloader, testloader)
    
    
    


def ExempleSnow(config, device):

    snow = [3449,3450,3837,3560,3671,3438,3484,3518,3519,3531,3533,3805,3836,4000,4001,4047,3496,3583,3581,3629,3582,3628,3882,3672,3627,3541,3804,3540,3763,3803,3808,3545,3544,4046,3542, 3589,3590,4091, 3656,3702,3452,3498]
    random.shuffle(snow)
    DATA_FOLDER_STR = 'dataset'
    DATA_FOLDER = Path(DATA_FOLDER_STR).expanduser()
    # path to the unzipped dataset: contains directories train/ and test/
    DATASET_FOLDER = DATA_FOLDER
    train_masks_paths = []
    train_images_paths= []
    test_masks_paths = []
    test_images_paths= []

    for i in range (np.shape(snow)[0]-20):
        random_train = random.randint(1, 10086)
        train_images_paths += sorted(list(DATASET_FOLDER.glob(f'train/images/{snow[i]}.tif')))
        train_masks_paths += sorted(list(DATASET_FOLDER.glob(f'train/masks/{snow[i]}.tif')))
        #train_images_paths += sorted(list(DATASET_FOLDER.glob(f'train/images/{random_train}.tif')))
        #train_masks_paths += sorted(list(DATASET_FOLDER.glob(f'train/masks/{random_train}.tif')))
    train_images_paths += train_images_paths+train_images_paths
    train_masks_paths += train_masks_paths+train_masks_paths
    train_images_paths += train_images_paths+train_images_paths
    train_masks_paths += train_masks_paths+train_masks_paths

    for i in range(np.shape(snow)[0]-20, np.shape(snow)[0]):
        random_test = random.randint(1, 10086)
        test_images_paths += sorted(list(DATASET_FOLDER.glob(f'train/images/{snow[i]}.tif')))
        test_masks_paths += sorted(list(DATASET_FOLDER.glob(f'train/masks/{snow[i]}.tif')))
        #test_images_paths += sorted(list(DATASET_FOLDER.glob(f'train/images/{random_test}.tif')))
        #test_masks_paths += sorted(list(DATASET_FOLDER.glob(f'train/masks/{random_test}.tif')))
    test_images_paths += test_images_paths+test_images_paths
    test_masks_paths += test_masks_paths+test_masks_paths
    test_images_paths += test_images_paths+test_images_paths
    test_masks_paths += test_masks_paths+test_masks_paths

    
    modelsnow = nn.Sequential(
        NeuralNetworkSnow(),
        nn.Softmax(1)
    ).to(device)
    batch = config['batch_size']
    train_dataset = CustomDatasetSnow(train_images_paths, train_masks_paths, train=True)
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    test_dataset = CustomDatasetSnow(test_images_paths, test_masks_paths, train=False)
    testloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    test_snow=True
    if config['test']:

        mask_cal = train_model_Unet(modelsnow, config, trainloader, testloader)
        return mask_cal
    else : 
        train_model_Unet(modelsnow, config, trainloader, testloader)
    
    
    
    

