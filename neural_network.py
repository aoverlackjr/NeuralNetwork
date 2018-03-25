# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt

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
                    self.create_genetic_network(value[0], value[1], value[2])
                if key == 'hybrid_config':
                    self.create_hybrid_network(value[0], value[1], value[2], value[3], **kwargs)

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
                    sigma_weight = value
                if key == 'sigma_bias':
                    sigma_bias = value

        # populate weight matrix with random values
        for wa in self.weight_addresses:
            self.weight_matrix[wa[0]][wa[1]] = np.random.randn()*sigma_weight

        # populate bias vector with random values
        self.bias_vector = np.random.randn(self.nr_of_neurons)*sigma_bias

    def create_genetic_network(self, n_inputs, n_outputs, chromosome_in, **kwargs):

        # First check if the chromosome lenght is compatible with the given
        # network parameters.
        # The first stretch of the cromo is for the network connectivity,
        # the second (of equal lenght) for the weights
        # The chromo length should be 2*(N^2 - N)/2 + N
        # which simplifies to N^2, where N = n_hidden + n_in + n_out

        n_hidden = int(sqrt(len(chromosome_in)) - n_inputs - n_outputs)

        if n_hidden < 1:
            raise Exception('Chromosome length not of compatible length')
        else:
            self.nr_of_neurons    = n_inputs + n_hidden + n_outputs
            self.nr_of_hidden     = n_hidden
            self.nr_of_inputs     = n_inputs
            self.nr_of_outputs    = n_outputs

            # pre-allocate the network data
            self.bias_vector   = np.zeros(self.nr_of_neurons)
            self.weight_matrix = np.zeros((self.nr_of_neurons, self.nr_of_neurons))
            self.signal_vector = np.zeros(self.nr_of_neurons)

            # set the signal addressing
            self.input_addresses  = range(0,self.nr_of_inputs)
            self.output_addresses = range(self.nr_of_neurons-self.nr_of_outputs, self.nr_of_neurons)

            n_conns  = int(self.nr_of_neurons*(self.nr_of_neurons - 1)/2)

            # extract the stretch determining the connections from the chromo:
            chr_connections  = chromosome_in[0:n_conns]
            chr_weights      = chromosome_in[n_conns:2*n_conns]
            self.bias_vector = np.array(chromosome_in[2*n_conns:])

            index = 0
            for i in range(0, self.nr_of_neurons):
                for j in range(i+1, self.nr_of_neurons):
                    if chr_connections[index] > 0.0:
                        self.weight_matrix[i][j] = chr_weights[index]
                    index += 1

            # create weight addresses from the genetically created weight
            # matrix
            self.create_weight_addresses()

    def create_hybrid_network(self, n_inputs, hidden_array, n_outputs, chromo, **kwargs):
        # Create normal feedforward config
        self.create_feedforward_network(n_inputs, hidden_array, n_outputs, **kwargs)
        # load with the chromo
        self.implement_state(chromo)


    def create_weight_addresses(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                if key == 'feedforward_config' or 'hybrid_config':
                    n_inputs = value[0]
                    hidden_array = value[1]
                    n_outputs    = value[2]
                    nrs = []
                    nrs.append(n_inputs)
                    for layer in hidden_array:
                        nrs.append(layer)
                    nrs.append(n_outputs)

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
            self.weight_addresses = []
            for i in range(self.nr_of_neurons):
                for j in range(self.nr_of_neurons):
                    if self.weight_matrix[i][j] != 0:
                        self.weight_addresses.append([i,j])

    def run(self, input_vector):
        # input the input vector into the signal field
        for i in self.input_addresses:
            self.signal_vector[i] = input_vector[i]
        # run the neurons in the network
        for n in range(self.nr_of_inputs, self.nr_of_neurons):
            weights = self.weight_matrix[:,[n]].reshape(self.nr_of_neurons)
            self.signal_vector[n] = self.fire_function(self.signal_vector, weights, self.bias_vector[n])
        # get the output
        return self.signal_vector[self.output_addresses]

    def extract_state(self):
        # return the weights and biases as one single vector, based on
        # the address list
        state = []
        for address in self.weight_addresses:
            state.append(self.weight_matrix[address[0]][address[1]])
        for bias in self.bias_vector:
            state.append(bias)
        return state

    def implement_state(self, state_vector):
        # put all the elements of the state vector into the weight
        # matrix and bias_vector

        # do some checking:
        if len(state_vector) == len(self.weight_addresses) + len(self.bias_vector):
            index = 0
            for address in self.weight_addresses:
                self.weight_matrix[address[0]][address[1]] = state_vector[index]
                index += 1
            # put the trailing remainder in the bias vector
            self.bias_vector = state_vector[index:]
        else:
            raise Exception('Weight and bias state to be implemented not of correct length')

    def _sigmoid(self, input_vector, weight_vector, bias):
        z = weight_vector.dot(input_vector) + bias
        return 1.0/(1.0 + np.exp(-z))

    def _tanhyp(self, input_vector, weight_vector, bias):
        z = weight_vector.dot(input_vector) + bias
        return np.tanh(z)

    def _sine(self, input_vector, weight_vector, bias):
        z = weight_vector.dot(input_vector) + bias
        return np.sine(z)

    @staticmethod
    def count_feedforward_connections(n_inputs, hidden_array, n_outputs):
        nrs = []
        nrs.append(n_inputs)
        for n in hidden_array:
            nrs.append(n)
        nrs.append(n_outputs)

        nr_of_connections = 0
        for i in range(1,len(nrs)):
            nr_of_connections += nrs[i-1]*nrs[i]

        # add biases:
        nr_of_connections += sum(hidden_array) + n_outputs + n_inputs
        return nr_of_connections
