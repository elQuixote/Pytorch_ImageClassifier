import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

'''utils.py: General Utilities '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

class Utilities(object):
    '''
    Basic static utilities class. 

    '''    

    @staticmethod
    def map_idx_to_classes(class_to_idx, classes, predicted_idx, predicted_prob):
        '''
        Maps the predicted indexes to the keys which we need to retrieve the class names. Since our model gives us the 'value',
        we need to find the key for our class_to_idx dict, once we have the key we can use it to find the class mapping 
        (flower name in this case).

        Args:
            class_to_idx (dic of ints): This is where we store the dictionary mapping the name of the class to the index (label).
            classes (dict of strs): Dict containing the mapping of the class idx to the name.
            predicted_idx (list of ints): The topk list of predicted indexes.
            predicted_prob (list of floats): The probability list from topk.

        Raises:
            TODO: Update exceptions with error_handling class.

        Returns:
            idx_classes_dict (dict): Dictionary containing 'predicted_indexes': indexes predicted by network, 
                                                            'idx_to_class': mapped idx_to_class, 
                                                            'classes': class names,
                                                            'probabilites': the probabilities for classes.

        '''

        idx_classes_dict = {}
        predicted_class_names = []
        predicted_idx_to_class = []

        for x in predicted_idx:
            for k,v in class_to_idx.items():
                if x == v:
                    predicted_class_names.append(classes[k])
                    predicted_idx_to_class.append(k)

        idx_classes_dict['predicted_idx'] = predicted_idx
        idx_classes_dict['classes'] = predicted_class_names
        idx_classes_dict['idx_to_class'] = predicted_idx_to_class
        idx_classes_dict['probabilities'] = predicted_prob

        return idx_classes_dict

