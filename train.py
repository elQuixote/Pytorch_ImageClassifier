import argparse
import sys
from collections import defaultdict
import os
import datetime
import torch as t

from utilities.utils import Utilities as utils
from utilities.net_utils import Net_Utilities as net_utils
from utilities.data_utils import Data_Utilities as data_util

'''train.py: Train a new network on a dataset of images '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

def main():

    try:

        args_dict = {}

        names = ['data_dir', '--save_dir', '--arch', '--learning_rate', '--hidden_units', '--epochs', '--gpu']

        def_save_path = os.getcwd() + '/checkpoint_nn_train.pth'
        def_save_path = def_save_path.replace('\\','/')
        
        defaults = [None, def_save_path, 'vgg16', 0.001, '1024, 512', 2, False]
        types = [str, str, str, float, str, int, bool]
        helpers = ['the directory of the data, ex. flowers/',
                'the fullpath with name where we want to save our checkpoints, ex. saved/checkpoint_xx.pth',
                'the architecture to transfer learning, vgg16, resnet50, densenet121',
                'the learning rate value',
                'the values for the hidden layer sizes form ex. 1024,512',
                'the number of epochs to train',
                'Use the gpu for computing, if no use cpu']
                
        for i in range(len(names)):
            data = {}
            data['name'] = names[i]
            data['default'] = defaults[i]
            data['type'] = types[i]
            data['help'] = helpers[i]

            args_dict[i] = data

        # get the args
        args = utils.get_input_args(args_dict)        

        # variables
        directory = args.data_dir
        if not os.path.exists(directory):
            raise OSError('Directory does not exist, please specify a new one')

        checkpt_dir = args.save_dir
        arch = args.arch
        learn_rate = args.learning_rate
        hidden_layers = args.hidden_units
        epochs = args.epochs 
        enable_gpu = args.gpu 

        trim = hidden_layers.strip('[]').split(',')
        hidden_layers = [int(i) for i in trim] 

        # check for gpu
        if not t.cuda.is_available() and enable_gpu:
            print('Your device does not have a CUDA capable device, we will use the CPU instead')
            response = input('Your device does not have a CUDA capable device, would you like to run it on the CPU instead? Enter Yes or No -> ')
            while response not in ('yes', 'no'):  
                if response.lower() == 'yes':
                    break
                elif response.lower() == "no":
                    print('exiting the program')
                    exit()
                else:
                    print('Please respond yes or no ')
            enable_gpu = False

        # generate datasets
        params_dict = {'train': {'dir': directory + 'train', 'batch': 64, 'shuffle': True},
                    'validate':{'dir': directory + 'valid', 'batch': 64, 'shuffle': True}}

        datasets, dataloaders = data_util.generate_datasets(params_dict, list(params_dict.keys()))

        # network instance
        processor = 'cuda' if enable_gpu else 'cpu'
        neural_net = net_utils.net_from_torchvision(hidden_layers, 102, 'relu', processor, learn_rate = learn_rate, name = arch)

        # train for n epochs
        neural_net.train_network(dataloaders['train'], dataloaders['validate'], epochs, plot = True)

        # save model
        neural_net.save_model_checkpoint(checkpt_dir, datasets['train'].class_to_idx)

    except Exception as ex:
        raise ex

if __name__ == '__main__':
    main()
