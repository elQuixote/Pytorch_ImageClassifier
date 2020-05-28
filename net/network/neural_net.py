import torch as t
from torch import nn
import math as m
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import copy

'''neural_net.py: Custom network object deriving from nn.Module to track the architecture '''
__author__ = "Luis Quinones"
__email__ = "luis@complicitmatter.com"
__status__ = "Prototype"

class Neural_Network(nn.Module):
    '''
    The neural network object sits a level above the classifier to
    store relevant properties and values. The classifier uses nn.LogSoftmax so use the 
    negative log likelihood loss criterion nn.NLLLoss

    Args:
        inputs (int): The number of inputs.
        hidden_sizes (list of ints): The hidden layer sizes.
        outputs (int): The number of outputs.
        hidden_activation (str): The hidden layer activation functions (ex. relu, sigmoid, tahn).
        device (str): The gpu or the cpu.
        optimizer_name (str): The optimizer name ('sgd' or 'adam') to update the weights and gradients
        dropout (float): The dropout rate, value to randomly drop input units through training.
        learn_rate (float): The learning rate value, used along with the gradient to update the weights,
            small values ensure that the weight update steps are small enough.

    Attributes:
        inputs (int): This is where we store the input count,
        hidden_sizes (list of int): This is where we store the hidden layer sizes,
        outputs (int): This is where we store the output size,
        hidden_activation (str): This is where we store the hidden activation type,
        dropout (float): This is where we store the random input unit dropout rate,
        learn_rate (float): This is where we store the learn rate value,
        processing_device (str): This is where we store the device to calculate the results,
        linear_layers (list): This is where we store the values to sequentially build the classifier,
        model (torch.nn.module or torchvision model): Where either the generated classifier or the loaded model is stored,
        optimizer (torch.optim): This is where we store the optimizer used,
        criterior (torch.nn.module.loss): This is where we store the loss function type,
        device (str): This is where we store the device,
        epochs_completed (int): This is where we store how many total epochs of training this model has.

    '''

    def __init__(self, inputs, hidden_sizes, 
                 outputs, hidden_activation, device,
                 dropout = 0.3, learn_rate = 0.002):

        super().__init__()
        # Props
        self.inputs = inputs
        self.hidden_sizes = hidden_sizes
        self.outputs = outputs
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        self.learn_rate = learn_rate
        self.processing_device = device
        # Layers
        self.linear_layers = []
        self.data = hidden_sizes
        self.data.insert(0,inputs)
        self.data.append(outputs)
        # Model Stuff
        self.model, self.optimizer =  None, None
        self.criterion = nn.NLLLoss()
        self.device = device

        self.epochs_completed = 0

        self.generate_classifier()

    def generate_classifier(self):
        '''Generates the nn.module container Sequential classfier as the default for this class.

        Args:
            None.

        Raises:
            TODO: Update exceptions with error_handling class.

        Returns:
            None.
        '''

        self.linear_layers = []
        n = len(self.data)
        for i in range(n-1):
            
            self.linear_layers.append(nn.Linear(self.data[i],self.data[(i + 1) % n]))
            
            if i != n-2:
                if self.hidden_activation == 'relu':
                    self.linear_layers.append(nn.ReLU())
                elif self.hidden_activation == 'sigmoid':
                    self.linear_layers.append(nn.Sigmoid())
                elif self.hidden_activation == 'tanh':
                    self.linear_layers.append(nn.Tanh())
                self.linear_layers.append(nn.Dropout(self.dropout))

        self.linear_layers.append(nn.LogSoftmax(dim = 1))
        # expand the list into sequential args
        self.model = nn.Sequential(*self.linear_layers)

    def train_network(self, train_data, validation_data, epochs = 1, load_best_params = False, plot = False):
        '''Trains the model, requires the criterion and optimizer to be passed into the class args before hand.

        TODO: add exception handling for optimizer and criterion as None values.

        Args:
            train_data (torch.utils.data.dataloader.DataLoader): The training torch data loader.
            validation_data (torch.utils.data.dataloader.DataLoader): The validation torch data loader.
            epochs (int): The number of epochs for training.
            load_best_params (bool): If true then we will load the model_state_dict from the highest accuracy iteration
            plot (bool): If true we plot both losses.

        Raises:
            TODO: Add exceptions.

        Returns:
            None.
        '''        
        
        # move the model to whatever device we have
        self.model.to(self.device)

        # if we loaded the model in eval mode and want to train switch it
        if not self.model.training:
            self.model.train()

        iteration, running_loss = 0, 0
        highest_accuracy, high_acc_iter, high_acc_epoch = 0, 0, 0
        training_loss_set, validation_loss_set = [], []
        best_params = None

        for epoch in range(epochs):
            batch_iteration = 0
            for x, y_labels in train_data:
                # move to whatever device we have
                x, y_labels = x.to(self.device), y_labels.to(self.device)
                # zero out the gradients
                self.optimizer.zero_grad()
                # forward pass - get the log probabilities (logits / scores)
                output = self.model(x)
                # calculate the loss
                loss = self.criterion(output, y_labels)
                # backprop - calculate the gradients for the parameters
                loss.backward()
                # parameter update based on gradient
                self.optimizer.step()
                # update stats
                running_loss += loss.item()
                iteration += 1
                batch_iteration += 1

            else:
                # Validation Process
                validation_loss, accuracy = self.validate_network(validation_data)

                training_loss = running_loss/len(train_data)
                print('Model has a total of {} training epochs completed.'.format(self.epochs_completed))
                print('Active session Epoch {} out of {}'.format(epoch + 1, epochs))
                print('Currently model has Accuracy of {}% \nCurrent training loss is {} \
                    \nCurrent validation loss is {}'.format(accuracy, 
                    training_loss, validation_loss))

                training_loss_set.append(training_loss)
                validation_loss_set.append(validation_loss)
                
                print('-------------')
                running_loss = 0

                # Track best run
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    high_acc_iter = batch_iteration
                    high_acc_epoch = epoch + 1
                    if load_best_params:
                        best_params = copy.deepcopy(self.model.state_dict())

                # Set the model back to train mode, enable dropout again
                self.model.train()
            self.epochs_completed += 1

        t_slope, v_slope = self.check_overfitting(training_loss_set, validation_loss_set, plot)
        print('Slope of linear reg training curve fit is {} \nSlope of linear reg Validation curve fit is {}'.format(t_slope, 
                                                                                                                    v_slope))
        print('Training session highest accuracy was {} on epoch {} batch iteration {}'.format(highest_accuracy, 
                                                                                             high_acc_epoch, 
                                                                                             high_acc_iter))
        if load_best_params:
            self.model.load_state_dict(best_params)
            print('Params from {} epoch, {} batch iteration were loaded'.format(high_acc_epoch, high_acc_iter))

    def validate_network(self, data):
        '''Validate our model to check the loss and accuracy.

        Args:
            data (torch.utils.data.dataloader.DataLoader): The data we want to validate as torch data loader.

        Raises:
            TODO: Add exceptions.

        Returns:
            loss,accuracy (tuple): The loss and accuracy of the validation.
        '''
        
        # enable eval mode, turn off dropout
        self.model.eval()
        # turn off the gradients since we are not updating params
        with t.no_grad():
            batch_loss = 0
            batch_accuracy = 0
            # validation pass
            for x, y_labels in data:
                # move to device
                x, y_labels = x.to(self.device), y_labels.to(self.device)
                output = self.model(x)
                # update loss and extract tensor as python float
                batch_loss += self.criterion(output, y_labels).item()
                # calculate the probability
                probability = t.exp(output)
                # get the top n indexes and values
                _, top_class = probability.topk(1, dim=1)
                # reshape top class to match label and get binary value from equals, 
                # check if the prediction matches label
                equals = top_class == y_labels.view(*top_class.shape)
                # have to convert byte tensor to float tensor and get accuracy
                batch_accuracy += t.mean(equals.type(t.FloatTensor)).item()

            test_accuracy = (batch_accuracy / len(data))*100
            test_loss = batch_loss / len(data)

            return test_loss, test_accuracy
    
    def check_overfitting(self, train_losses, validation_losses, plot = False):
        '''Validate our model to check the loss and accuracy

        Args:
            train_losses (list of floats): The list of training losses per epoch.
            validation_losses (list of floats): The list of validation losses per epoch.
            plot (bool): If true we plot both losses.

        Raises:
            TODO: Add exceptions.

        Returns:
            slopes (tuple): The slopes of the linear reg curve fits for both validation/training.
        '''
        # Data 
        tl_x_val = np.arange(0, len(train_losses))
        vl_x_val = np.arange(0, len(validation_losses))   
        # To numpy
        train_data = np.array([tl_x_val, train_losses])
        validate_data = np.array([vl_x_val, validation_losses])
        # Least squares polynomial fit.
        train_slope, train_intercept = np.polyfit(train_data[0], train_data[1], 1)
        validation_slope, validation_intercept = np.polyfit(validate_data[0], validate_data[1], 1)

        if plot:
            plt.plot(train_data[0], train_data[1], 'o', label='training loss')
            plt.plot(validate_data[0], validate_data[1], 'o', label='validation loss')
            plt.plot(train_data[0], train_intercept + train_slope*train_data[0], 'r', label='train_regg')
            plt.plot(validate_data[0], validation_intercept + validation_slope*validate_data[0], 'r', label='val_regg')
            plt.legend()
            plt.show()
        
        return train_slope, validation_slope

    def save_model_checkpoint(self, full_path, training_class_to_idx):
        '''Save the model checkpoint.

        Args:
            full_path (str): The full path to save the checkpoint to
            training_class_to_idx (dic of ints): This is where we store the dictionary mapping the name of the class to the index (label)

        Raises:
            TODO: Add exceptions

        Returns:
            None
        '''
                
        net_data_dic = {'input_count': self.inputs,
                        'hidden_sizes': self.hidden_sizes,
                        'outputs': self.outputs,
                        'h_activation': self.hidden_activation,
                        'dropout': self.dropout,
                        'learn_rate': self.learn_rate,
                        'epochs_completed' : self.epochs_completed}
        
        checkpoint = {'data' : net_data_dic,
                      'model' : self.model, 
                      'classifier' : self.model.classifier,
                      'optimizer.state_dict' : self.optimizer.state_dict(),
                      'state_dict' : self.model.state_dict(),
                      'device' : self.device,
                      'class_to_idx': training_class_to_idx}

        t.save (checkpoint, full_path)


