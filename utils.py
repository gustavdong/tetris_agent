from PIL import Image
import numpy as np
import cv2


def crop_image(image):
    '''
    Args: image as an numpy array
    returns: a crop image as a standardised numpy array
    '''
    image = image[47:209,95:176,:]
    image = np.mean(image, axis=2)
    image[image > 0] = 1

    image = cv2.resize(image, (10,20))
    # image = cv2.resize(image, (82,82))
    # image = image.mean(2, keepdims=True)
    image = image.astype(np.float32)
    # image *= (1.0 / 255.0)
    # image = np.moveaxis(image, -1, 0)

    # print(image)
    return image
