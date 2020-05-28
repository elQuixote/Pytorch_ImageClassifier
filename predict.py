import argparse
import json
import torch as t
import os
import sys

from network.net_operations import Net_Operations as net_ops
from utilities.net_utils import Net_Utilities as net_utils
from utilities.utils import Utilities as utils

'''predict.py: Predict a flower name from an image along with the probability of that name '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

def main():

    try:

        args_dict = {}

        names = ['image_path', 'model_checkpoint_path', '--top_k', '--category_names', '--gpu']
        defaults = [None, None, 3, 'flower_to_name.json', True]
        types = [str, str, int, str, bool]
        helpers = ['the path to the image we want to predict',
                'the path to the model checkpoint to load',
                'return the top k most likely cases',
                'Json file with the mapping of categories to real names',
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

        # get flower name from path
        flower_name = categories_to_name[img_path.split('/')[-2]]
        
        # print the top n results
        print('FLOWER NAME IS {} \n'.format(flower_name.upper()))
        print('THE TOP {} RESULTS ARE:'.format(top_k))
        for i, name in enumerate(names):
            print('Name = {} \nProbability = {} \n'.format(name, results_dict['probabilities'][i]))

    except Exception as ex:
        raise ex

if __name__ == "__main__":
    main()
