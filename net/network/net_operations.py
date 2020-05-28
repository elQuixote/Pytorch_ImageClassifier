import torch as t
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

from utilities.img_utils import Image_Utilities as im_util
from utilities.utils import Utilities as util

'''net_operations.py: Static class for Neural_Network operations '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

class Net_Operations(object):

    @staticmethod
    def predict(network, image_path, class_names, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.

        Args:
            network (nn_model.Neural_Network): The Neural_Network instance to use for the prediction.
            image_path (str): The path to the image we want to test
            class_names (dict of ints): The label map with the class names
            topk (int): The number of top probabilities and classes we want.

        Raises:
            TODO: Add exceptions

        Returns:
            data_dict (dict): Dictionary containing 'predicted_indexes': indexes predicted by network, 
                                                    'idx_to_class': mapped idx_to_class, 
                                                    'classes': class names,
                                                    'probabilites': the probabilities for classes.        
        '''
        
        # convert image 
        img = im_util.process_image(image_path, 224, 224)
        # need to pass the image tensor with first argument of n where n represents our batch size
        img.unsqueeze_(0)
        # move to device
        img.to(network.device)
        # generate the prediction
        top_prob, top_class = Net_Operations.generate_prediction(network.model, img, network.device, topk)
        # remove the tensor cuda by moving to cpu, squeeze to remove dimensions and send to list to index
        top_prob = top_prob.cpu().squeeze().tolist()
        top_class = top_class.cpu().squeeze().tolist()
        # generate the idx_to_class mapping dict
        data_dict = util.map_idx_to_classes(network.model.classifier.class_to_idx, class_names, top_class, top_prob)
        
        return data_dict

    @staticmethod
    def generate_prediction(model, data, device, topk = 5):
        '''Predicts probabilities and gets top k classes using topk for a single image test.

        Args:
            model (nn_model): The model to use for the prediction.
            data (torch.Tensor): The data (img) we want to test.
            device (str): The device to use for processing.
            topk (int): The number of top probabilities and classes we want.

        Raises:
            TODO: Add exceptions

        Returns:
            Tuple ([floats], [ints]): The tuple containing the list of k probs and k classes.
        '''
        
        # move model to device
        model.to(device)
        # enable eval mode, turn off dropout
        model.eval()
        # turn off the gradients since we are not updating params
        with t.no_grad():
            data = data.to(device, dtype=t.float)
            # get the log softmax
            output = model(data)
            # get the prob
            probabilities = t.exp(output)
            # get the top k values
            top_probabilities, top_classes = probabilities.topk(topk, dim=1)
    
        return top_probabilities, top_classes

    @staticmethod
    def plot_n_predictions(network, datasets, filter, count, label_map):
        # generate random n indexes to choose random testing images from dataset
        idx = np.random.randint(0,len(datasets[filter].imgs),size=(count,))
        print(idx)

        # get the image folder number idx from the randomly selected dataset image
        batch_idx = [datasets[filter].imgs[x][0].split('\\')[-2] for x in idx]
        print(batch_idx)

        # fix the full path for the batch idx's
        batch_paths = [datasets[filter].imgs[x][0].replace('\\','/') for x in idx]
        print(batch_paths)

        # get actual flower name from the mapping back to the label
        labeled_names = [label_map[x] for x in batch_idx]
        print(labeled_names)

        # zip the data
        data = dict(zip(labeled_names, batch_paths))

        # set the subplots
        rows = (len(data.items())) 
        cols = 2
        fig, axs = plt.subplots(nrows = rows, ncols= cols, figsize=(cols*4,rows*3), squeeze = False)
        axs = axs.flatten()
        plt.tight_layout()

        # iterate through the dict, plot the graphs on the even grid cell
        # plot matching imgs on the odd grid cells
        count, img_counter = 0, 1
        for name, path in data.items():
            # get the predictions
            results_dict = Net_Operations.predict(network, path, label_map)
            for k,v in results_dict.items():
                print('{}:{}'.format(k, v))
            print('flower is {}\n'.format(name))
            
            # barplots for the results
            bp = sb.barplot(x=results_dict['probabilities'], y=results_dict['classes'], ax=axs[count])
            bp.set_title(name)
            
            # plot the images
            img = im_util.process_image(path, 224, 224)
            im_util.show_image(img, axs[img_counter])
            
            # increment the counters
            count += 2
            img_counter += 2
                
        plt.show()

