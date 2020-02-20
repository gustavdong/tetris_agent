from PIL import Image
import numpy as np
import cv2


def crop_image(image):
    '''
    Args: image as an numpy array
    returns: a crop image as a standardised numpy array
    '''
    image = image[45:210, 90:180, :]
    image = cv2.resize(image, (45, 82))
    # image = cv2.resize(image, (82,82))
    # image = image.mean(2, keepdims=True)
    image = image.astype(np.float32)
    image *= (1.0 / 255.0)
    # image = np.moveaxis(image, -1, 0)

    # print(image)
    return image
