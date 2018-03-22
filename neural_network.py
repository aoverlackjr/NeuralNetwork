# -*- coding: utf-8 -*-

import numpy as np

class NeuralNetwork(object):
    
    def __init__(self, *args, **kwargs):
        
        self.nr_of_neurons    = None
        self.nr_of_hidden     = None
        self.nr_of_inputs     = None
        self.nr_of_outputs    = None
        
        self.weight_matrix    = None
        self.bias_vector      = None
        
        self.chromosome       = None
        
        self.fire_function    = self._sigmoid
        
        self.signal_vector    = None
        self.input_addresses  = None
        self.output_addresses = None
        self.weight_addresses = None
        
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
                if key == 'feedforward_config':
                    self.create_feedforward_network(value[0], value[1], value[2], **kwargs)
                if key == 'genetic_config':
                    self.create_genetic_network(value[0], value[1], value[2], **kwargs)
        
        
        
    def create_feedforward_network(self, n_inputs, hidden_array, n_outputs, **kwargs):
        # set characteristics
        self.nr_of_inputs  = n_inputs
        self.nr_of_outputs = n_outputs
        self.nr_of_hidden  = np.sum(hidden_array)
        self.nr_of_neurons = n_inputs + self.nr_of_hidden + n_outputs
        
        # pre-allocate the network data
        self.bias_vector   = np.zeros(self.nr_of_neurons)
        self.weight_matrix = np.zeros((self.nr_of_neurons, self.nr_of_neurons))
        self.signal_vector = np.zeros(self.nr_of_neurons)
        
        # set the signal addressing
        self.input_addresses  = range(0,self.nr_of_inputs)
        self.output_addresses = range(self.nr_of_neurons-self.nr_of_outputs, self.nr_of_neurons)
        
        # pre set some parameters.
        sigma_weight = 1.0
        sigma_bias   = 1.0
        
        # create weight addresses
        self.create_weight_addresses(**kwargs)
        
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'sigma_weight':
                    max_weight = value
                if key == 'sigma_bias':
                    max_bias = value
        
        # populate weight matrix with random values
        for wa in self.weight_addresses:
            self.weight_matrix[wa[0]][wa[1]] = np.random.randn()*sigma_weight
            
        # populate bias vector with random values
        for i in range(len(self.bias_vector)):
            self.bias_vector[i] = np.random.randn()*sigma_weight
            
        
        
        
                    
        
                    
        
    def create_genetic_network(self, n_inputs, chromosome, n_outputs, **kwargs):
        
        pass
        
    def create_weight_addresses(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'feedforward_config':
                    n_inputs = value[0]
                    hidden_array = value[1]
                    nrs = []
                    nrs.append(n_inputs)
                    for layer in hidden_array:
                        nrs.append(layer)
                    
                    self.weight_addresses = []
                    ranges = []
                    prevsze = 0
                    for r in range(len(nrs)):
                        ranges.append(range(prevsze, prevsze + nrs[r]))
                        prevsze += nrs[r]
                        
                    for i in range(len(ranges)-1):
                        for f in ranges[i]:
                            for t in ranges[i+1]:
                                self.weight_addresses.append([f,t])
            
        # if no input was given, try to use the existing weight matrix to extract
        # the non-zero weights' addresses
        else:
            pass
            
            
    def create_bias_vector(self, **kwargs):
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
        
        
    
