import torchvision.transforms as tf
import torchvision.datasets as ds
import torch

'''data_utils.py: Static class for general data manipulation '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

class Data_Utilities(object):

    @staticmethod
    def generate_datasets(params_dict, types_list, resize = 300, crop_size = 224):

        ''' Generators and data manipulation. Generates the required data transformations 
        for us to train properly. 
        
        Args:
            params_dict (dict): The nested dictionary containing the 'dir', 'batch' and 'shuffle' data.
            types_list (list of str): The list of param_dict keys, 'train', 'validate', 'test'.
            resize (int): The value to resize the image to.
            crop_size (int): The value we want to crop the image to

        Raises:
            TODO: Add exceptions

        Returns:
            datasets, dataloaders (tuple): The datasets and data loaders 
        '''

        # Define the transforms
        transforms = {}
        for t in types_list:

            transform_list = []
            transform_list.append(tf.Resize(resize))
            transform_list.append(tf.CenterCrop(crop_size))
            transform_list.append(tf.ToTensor())
            transform_list.append(tf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))

            if t == 'train':
                transform_list.pop(1)
                transform_list.insert(1, tf.RandomResizedCrop(crop_size))
                transform_list.insert(2, tf.RandomHorizontalFlip())

            transforms[t] = tf.Compose(transform_list)

        # Load the data sets, use dict comprehension to generate key vals for each type
        datasets = {t: ds.ImageFolder(params_dict[t]['dir'], 
                                    transforms[t]) for t in types_list}
        # Define the loaders using the datasets and the transforms
        dataloaders = {t: torch.utils.data.DataLoader(datasets[t], 
                                                    params_dict[t]['batch'], 
                                                    params_dict[t]['shuffle']) 
                                                    for t in types_list}

        return datasets, dataloaders

