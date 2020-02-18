from PIL import Image
import numpy as numpy
import cv2

def crop_image(image):
    '''
    Args: image as an numpy array
    returns: a crop image as a numpy array
    '''
    image = image[45:210,90:180,:]
    image=cv2.resize(image, (45,82))

    return image