import torch
import torch.nn as nn

#nn.Module is superclass NeuralNetwork() subclass
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, hidden_size) 
        self.l4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    #l1,l2,l3,l4 are the 4 layers of the network

    #forward() defines structure of nueralnetwork, maps x to y
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        # no activation and no softmax at the end
        return out
