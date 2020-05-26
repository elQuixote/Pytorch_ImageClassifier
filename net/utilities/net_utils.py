import torch as t
from torch import nn
import torchvision.models as models

from network.neural_net import Neural_Network

class Net_Utilities(object):

    @staticmethod
    def net_from_torchvision(hidden_sizes, outputs, hidden_activation, device, 
                            optimizer_name = 'adam', dropout = 0.3, learn_rate = 0.002, 
                            name = 'vgg16', trained = True):

        '''
        Generates a model from torchvision, and instatiates a new Neural_Network instance which sets new model 
        as the active model. A new optimizer and criterion are also generated and assigned to the class properties.

        Args:
            hidden_sizes (list of ints): The hidden layer sizes.
            outputs (int): The number of outputs.
            hidden_activation (str): The hidden layer activation functions (ex. relu, sigmoid, tahn).
            device (str): The gpu or the cpu.
            optimizer_name (str): The optimizer name ('sgd' or 'adam') to update the weights and gradients
            dropout (float): The dropout rate, value to randomly drop input units through training.
            learn_rate (float): The learning rate value, used along with the gradient to update the weights,
                small values ensure that the weight update steps are small enough.
            name (str): The pretrained model name ('vgg16', 'resnet50', 'densenet121').
            trained (bool): If the model has been trained.

        Raises:
            TODO: Update exceptions with error_handling class.

        Returns:
            net (nn_model.Neural_Network): An instance of the Neural_Network class with the trained model
                as its model and parameters.
        '''

        model = Net_Utilities.get_pretrained_model(name, trained)
        feature_count = model.classifier[0].in_features

        net = Neural_Network(feature_count, hidden_sizes, outputs, 
                            hidden_activation, device, dropout, learn_rate)

        model.classifier = net.model
        net.model = model

        if optimizer_name != 'adam' and optimizer_name != 'sgd':
            raise ValueError('Please use either SDG or Adam as optimizers')
        elif optimizer_name == 'adam':
            net.optimizer = t.optim.Adam(net.model.classifier.parameters(), learn_rate)
        else:
            net.optimizer = t.optim.SDG(net.model.classifier.parameters(), learn_rate)

        net.criterion = nn.NLLLoss()

        return net  

    @staticmethod
    def load_neural_net(filepath, mode = 'train'):
        '''
        Generates a model from torchvision, and instatiates a new Neural_Network instance which sets new model 
        as the active model. A new optimizer and criterion are also generated and assigned to the class properties.

        Args:
            file_path (str): The full path to the checkpoint
            mode (str): Mode to set the model to ('train', 'eval')

        Raises:
            TODO: Update exceptions with error_handling class.

        Returns:
            net (nn_model.Neural_Network): An instance of the Neural_Network class with the loeaded model
                as its model, parameters, criterion and optimizer.
        '''        

        print('loading_net')
        #TODO: Path validation
        checkpoint = t.load(filepath)
        # Set Params
        inputs = checkpoint['data']['input_count']
        hidden_layers = checkpoint['data']['hidden_sizes']
        outputs = checkpoint['data']['outputs']
        activation = checkpoint['data']['h_activation']
        dropout = checkpoint['data']['dropout']
        learn_rate = checkpoint['data']['learn_rate']
        device = checkpoint['device']
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        # Make Network
        net = Neural_Network(inputs, hidden_layers, outputs, activation, device, dropout, learn_rate)
        net.model = model
        net.epochs_completed = checkpoint['data']['epochs_completed']

        if mode == 'train':
            net.model.train()
        elif mode == 'eval':
            net.model.eval()
        else:
            raise ValueError('Error mode needs to be either train or eval')

        net.model.classifier.class_to_idx = checkpoint['class_to_idx']
        optimizer = t.optim.Adam(net.model.classifier.parameters(), learn_rate)
        optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
        criterion = nn.NLLLoss()
        net.optimizer = optimizer
        net.criterion = criterion
        # Move to processing device
        net.model.to(device)
        
        return net

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

