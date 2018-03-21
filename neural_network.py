# -*- coding: utf-8 -*-

import numpy as np

class NeuralNetwork(object):
    
    def __init__(self,*args,**kwargs):
        
        self.nr_of_neurons   = None
        self.nr_of_inputs    = None
        self.nr_of_outputs   = None
        
        self.weight_matrix   = None
        self.bias_vector     = None
        
        self.chromosome      = None
        
        self.fire_function   = self._sigmoid
        
        self.signal_vector   = []
        
        self.configure(*args, **kwargs)
        
        
    def configure(self, *args, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'activation':
                    if value == 'sigmoid':
                        self.fire_function = self._sigmoid
                    if value == 'tanhyp':
                        self.fire_function = self._tanhyp
                    if value == 'sine':
                        self.fire_function = self._sine
                    if value == 'feedforward_config':
                        self.create_feedforward_network(value[0], value[1], value[2])
                    if value == 'genetic_config':
                        self.create_genetic_network(value[0], value[1], value[2])
        
        
        
    def create_feedforward_network(self, n_inputs, hidden_array, n_outputs):
        
        pass
        
    def create_genetic_network(self, n_inputs, chromosome, n_outputs):
        
        pass
        
    def run(self, input_vector):
        
        pass
        
    def _sigmoid(self, input_vector, weight_vector, bias):
        z = weight_vector.dot(input_vector) + bias
        return 1.0/(1.0 + np.exp(-z))
        
    def _tanhyp(self, input_vector, weight_vector, bias):
        z = weight_vector.dot(input_vector) + bias
        return np.tanh(z)
        
    def _sine(self, input_vector, weight_vector, bias):
        z = weight_vector.dot(input_vector) + bias
        return np.sine(z)
        
        
    
