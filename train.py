import argparse
import sys
from collections import defaultdict
import os
import datetime
import torch as t

from utilities.utils import Utilities as utils
from utilities.net_utils import Net_Utilities as net_utils
from utilities.data_utils import Data_Utilities as data_util

def main():

    try:

        args_dict = {}

        names = ['data_dir', '--save_dir', '--arch', '--learning_rate', '--hidden_units', '--epochs', '--gpu']

        def_save_path = os.getcwd() + '/checkpoint_nn_train.pth'
        def_save_path = def_save_path.replace('\\','/')
        
        defaults = [None, def_save_path, 'vgg16', 0.001, [1024, 512], 2, True]
        types = [str, str, str, float, list, int, bool]
        helps = ['the directory of the data, ex. flowers/',
                'the fullpath with name where we want to save our checkpoints, ex. saved/checkpoint_xx.pth',
                'the architecture to transfer learning, vgg16, resnet50, densenet121',
                'the learning rate value',
                'the list of hidden layer sizes',
                'the number of epochs to train',
                'Use the gpu for computing, if no use cpu']
                
        for i in range(len(names)):
            data = {}
            data['name'] = names[i]
            data['default'] = defaults[i]
            data['type'] = types[i]
            data['help'] = helps[i]

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
        neural_net.train_network(dataloaders['train'], dataloaders['validate'], epochs, plot = False)

        # save model
        neural_net.save_model_checkpoint(checkpt_dir, datasets['train'].class_to_idx)

    except Exception as ex:
        raise ex

if __name__ == '__main__':
    main()
