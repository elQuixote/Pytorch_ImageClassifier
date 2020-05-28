import argparse
import json
import torch as t
import os
import sys

from network.net_operations import Net_Operations as net_ops
from utilities.net_utils import Net_Utilities as net_utils

'''predict.py: Predict a flower name from an image along with the probability of that name '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

def get_input_args():
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    Command Line Args:
        image_path (path): The path to the image we want to predict.
        model_checkpoint (path): The pth checkpoint file we want to load
        top_k (int): The top number of classes we want to predict
        category_names (int): Json file with the mappings of categories to names
        gpu (bool): True if we want to use the gpu false if we want to use the cpu

    Raises:
        TODO: Add exceptions

    Returns:
        parse_args (): The data which stores the command line argument object   
    '''    

    description = 'This program uses a trained network to predict the class and probabliity for an input image.'

    parser = argparse.ArgumentParser(description = description)

    parser.add_argument('image_path', 
                        type = str, 
                        help = 'the path to the \
                        image we want to predict')

    parser.add_argument('model_checkpoint_path', 
                        type = str, 
                        help = 'the path to \
                        the model checkpoint to load')

    parser.add_argument('--top_k', 
                        default = 3, 
                        type = int, 
                        help = 'return the top K most likely cases')

    parser.add_argument('--category_names', 
                        default = 'flower_to_name.json', 
                        type = str, 
                        help = 'Json file with the mapping \
                                of categories to real names')

    parser.add_argument('--gpu',
                        default = False,
                        type = bool,
                        help = 'Use the gpu for computing, if no use cpu')

    return parser.parse_args()

def main():

    try:
        # get the args
        args = get_input_args()
        
        # variables
        img_path = args.image_path
        model_checkpoint = args.model_checkpoint_path
        top_k = args.top_k
        categories = args.category_names
        enable_gpu = args.gpu 
        
        # check if the img path exist
        while not os.path.isfile(img_path):
            img_path = input('Image file does not exist, please input a correct path \n')
            if img_path == 'quit':
                exit()

        # check if the checkpoint file exist
        while not os.path.isfile(model_checkpoint):
            model_checkpoint = input('Model checkpoint does not exist, please input a correct path \n')
            if model_checkpoint == 'quit':
                exit()

        while top_k < 1:
            val = input('Top_k value must be greater than 0, please enter a new value \n')
            top_k = int(val)

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

        # load from checkpoint and set device
        mfcp = net_utils.load_neural_net(model_checkpoint, 'eval')
        mfcp.device = 'cuda' if enable_gpu else 'cpu'

        # load json data
        with open(categories, 'r') as f:
            categories_to_name = json.load(f)

        # make the predictions
        results_dict = net_ops.predict(mfcp, img_path, categories_to_name, topk = top_k)
        names = [categories_to_name[x] for x in results_dict['idx_to_class']]
        
        # print the top n results
        print('THE TOP {} RESULTS ARE:'.format(top_k))
        for i, name in enumerate(names):
            print('Name = {} \nProbability = {} \n'.format(name, results_dict['probabilities'][i]))

    except Exception as ex:
        raise ex

if __name__ == "__main__":
    main()
