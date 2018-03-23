from neural_network import NeuralNetwork

nn = NeuralNetwork( activation = 'tanhyp', 
                    feedforward_config = (2,(3,2),2) )


print(nn.run([1,2]))
