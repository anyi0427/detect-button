import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm
import cv2

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name):    
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    img = cv2.imread(dir_name)
    img = rgb2gray(img)
    imgs.append(img)
    return img

def visualize(img, format=None, gray=False):
    plt.figure()
    plt.axis('off')
    # for i, img in enumerate(imgs):
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    plt.imshow(img, format)
    # plt.savefig('output.png',dpi=100,transparent=True)
    plt.show()

    