import torch                                                        
from torch.utils.data import DataLoader                             
from torchvision import transforms                                  
import torch.optim as optim                                         
import torch.nn as nn                                               
import torch.backends.cudnn as cudnn                                
import torchvision.datasets as datasets  

import os
import numpy as np
import random

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import linear_assignment_
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from itertools import chain

def load_imagenet(batch_size, image_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder(root='/home/libi/ImageNet2012/train/',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]))
    test_data =  datasets.ImageFolder(root='/home/libi/ImageNet2012/train/',
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
            ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, test_loader, 1000

def plot_samples_from_images(images, cam, batch_size, plot_path, filename):
    max_pix = torch.max(torch.abs(images))
    images = ((images/max_pix) + 1.0)/2.0
    cam = (cam-torch.min(cam))/torch.max(cam)
    images = images + cam*4.0
    images = images/torch.max(images)
    if(images.size()[1] == 1): # binary image
        images = torch.cat((images, images, images), 1)
    
    images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)

    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, filename))
    else:
        plt.show()
    plt.close()

