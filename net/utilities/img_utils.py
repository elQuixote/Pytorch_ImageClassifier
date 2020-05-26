import numpy as np
import torch as t
from PIL import Image

'''img_utils.py: Static class for general image processing utilities '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

class Image_Utilities(object):

    @staticmethod
    def process_image(image, width, height):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an torchNumpy array
        
        Args:
            image (nn_model.Neural_Network): The Neural_Network instance to use for the prediction.
            width (int): The path to the image we want to test
            height (int): The label map with the class names

        Raises:
            TODO: Add exceptions

        Returns:
            t_image (torch.Tensor): 
        '''

        # open and resize
        img = Image.open(image)
        img = img.resize((width,height))
        
        # crop
        current_width, current_height = img.size
        left = (current_width - width)/2
        top = (current_height - height)/2
        right = left + width
        bottom = top + height
        img = img.crop((left, top, right, bottom))
        
        # normalize the values
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = np.array(img) / 255
        np_img = (np_img - mean) / std
        
        # swap color channel position
        np_img = np.transpose(np_img, (2,0,1))

        # conver to tensor from numpy ndarray
        t_image = t.from_numpy(np_img)

        return t_image



