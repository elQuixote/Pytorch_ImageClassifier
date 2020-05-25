import torch
from torch import nn
import torchvision.models as models

'''utils.py: Custom network object utilities '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

class net_utilities(object):
    '''
    Basic static utilities class for net operations. 

    '''    

    @staticmethod    
    def get_pretrained_model(name = 'vgg16', trained = True):
        '''Generates the nn.module container Sequential classfier as the default for this class.

        Args:
            name (str): The pretrained model name ('vgg16', 'resnet50', 'densenet121').
            trained (bool): If the model has been trained.

        Raises:
            TODO: Update exceptions with error_handling class.

        Returns:
            model (torchvision.models.vgg.VGG): The torch vision model specified
        '''        

        # get model from torchvision
        if name == 'vgg16':
            model = models.vgg16(pretrained = trained)
        elif name == 'resnet50':
            model = models.resnet50(pretrained = trained)
        elif name == 'densenet121':
            model = models.densenet121(pretrained = trained)
        else:
            raise ValueError('Please select from either vgg16, resnet50 or \
                            densenet121 pre-trained models')
            
        # freeze parameters
        for parameter in model.parameters():
            parameter.requires_grad = False

        return model  


