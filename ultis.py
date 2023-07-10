from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
from torch import nn
from torchvision.transforms._presets import SemanticSegmentation, ObjectDetection
from functools import partial

import numpy as np
import os, glob
import PIL.Image as Image

class base_dataset(Dataset):
    '''
    mode: 'train', 'val', 'test'
    data_path: path to data folder
    imgsize: size of image
    transform: transform function
    '''
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.img_list = None

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        '''
        return tran_image, target, original image
        '''
        img_path = self.img_list[index]
        return img_path
    
class LungCTscan(Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        img_list = sorted(glob.glob(data_path + '/2d_images/*.tif'))
        mask_list = sorted(glob.glob(data_path + '/2d_masks/*.tif'))
        
        n = len(img_list)
        if(mode == 'train'):
            self.img_list = img_list[:int(n*0.8)]
            self.mask_list = mask_list[:int(n*0.8)]
        elif(mode == 'val'):
            self.img_list = img_list[int(n*0.8):]
            self.mask_list = mask_list[int(n*0.8):]
        elif(mode == 'test'):
            self.img_list = img_list[int(n*0.8):]
            self.mask_list = mask_list[int(n*0.8):]

        self.transform = transform
        self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])
        if(self.transform is None):
            self.transformImg = partial(SemanticSegmentation, resize_size=imgsize)()
            
    def __len__(self):
        return len(self.img_list)
        
    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # load image
        image = Image.open(image_path).convert('RGB')
        # resize image with 1 channel

        # load image
        mask = Image.open(mask_path).convert('L')

        if self.transform is None:
            tran_image = self.transformImg(image)
            mask = self.transformAnn(mask)
        else:
            tran_image = self.transform(image)
            mask = self.transform(mask)

        return tran_image, mask.squeeze(0), self.transformAnn(image)
    

def rgb_to_2D_label(label):
 
    Land = np.array(tuple(int('#8429F6'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    Road = np.array(tuple(int('#6EC1E4'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    Vegetation = np.array(tuple(int('FEDD3A'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    Water = np.array(tuple(int('E2A929'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    Building = np.array(tuple(int('#3C1098'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    Unlabeled = np.array(tuple(int('#9B9B9B'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

    label_seg = np.zeros(label.shape, dtype=np.uint8)
    label_seg [np.all(label == Building, axis = -1)] = 2
    label_seg [np.all(label == Unlabeled, axis = -1)] = 0
    label_seg [np.all(label == Land, axis = -1)] = 0
    label_seg [np.all(label == Road, axis = -1)] = 1  
    label_seg [np.all(label == Vegetation, axis = -1)] = 0   
    label_seg [np.all(label == Water, axis = -1)] = 0
   
    label_seg = label_seg[:,:,0]
    
    return label_seg


class DubaiAerialread(Dataset):
    '''
    Dubai Aerial Imagery dataset:
    https://www.kaggle.com/code/gamze1aksu/semantic-segmentation-of-aerial-imagery
    The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated 
    with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images 
    grouped into 6 larger tiles. The classes are:
    Building: #3C1098
    Land (unpaved area): #8429F6
    Road: #6EC1E4
    Vegetation: #FEDD3A
    Water: #E2A929
    Unlabeled: #9B9B9B
    '''
    def __init__(self, mode, data_path, transform=None, imgsize=224):
        input_images = []
        input_labels = []
        self.transform = transform
        for path, _, _ in os.walk(data_path):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'images':
                input_images += [os.path.join(path, file) for file in os.listdir(path)]
            if dirname == 'masks':
                input_labels += [os.path.join(path, file) for file in os.listdir(path)]
        
        img_list = sorted(input_images)
        mask_list = sorted(input_labels)

        n = len(img_list)
        if(mode == 'train'):
            self.img_list = img_list[:int(n*0.8)]
            self.mask_list = mask_list[:int(n*0.8)]
        elif(mode == 'val'):
            self.img_list = img_list[int(n*0.8):]
            self.mask_list = mask_list[int(n*0.8):]

        if(self.transform is None):
            self.transformImg = partial(SemanticSegmentation, resize_size=(imgsize, imgsize))()
            self.transformAnn = transforms.Compose([transforms.Resize((imgsize, imgsize)),
                                                    transforms.ToTensor()])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        mask_path = self.mask_list[idx]

        # load image
        image = Image.open(image_path).convert('RGB')
        # load mask
        mask = Image.open(mask_path).convert('RGB')
        mask = rgb_to_2D_label(np.asarray(mask))

        if self.transform is None:
            image = self.transformImg(image)
            mask = Image.fromarray(np.uint8(mask))
            mask = self.transformAnn(mask)
        else:
            image = np.asarray(image)
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.squeeze(0), image