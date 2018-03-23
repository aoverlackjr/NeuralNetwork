from neural_network import NeuralNetwork
import numpy as np

n_inputs = 2
n_outputs = 2
n_hidden  = 2

N = n_inputs + n_outputs + n_hidden
chromo = np.random.randn(N**2)

nn = NeuralNetwork( activation = 'tanhyp', 
                    genetic_config = (n_inputs,n_outputs,chromo) )

nn.implement_state(nn.extract_state())

print(nn.run([0,0]))
nn.implement_state(nn.extract_state())
print(nn.run([0,0]))
