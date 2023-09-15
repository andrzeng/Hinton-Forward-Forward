import torch
import torch.nn as nn
from torchvision import datasets
from typing import List

class FFmodel(nn.Module):
    def __init__(self, 
                 input_size: int,
                 layer_sizes: List[int]):
        super().__init__()

        if(len(layer_sizes) == 0):
            raise ValueError("You must include at least one layer size in layer_sizes")
        if(input_size <= 0):
            raise ValueError("You msut provide a positive integer as the input size")
        
        self.linear_layers = nn.ModuleList([])
        self.layernorms = nn.ModuleList([])

        for index, layer_size in enumerate(layer_sizes):
            if(index == 0):
                self.linear_layers.append(nn.Linear(input_size, layer_size))
            else:
                self.linear_layers.append(nn.Linear(layer_sizes[index-1], layer_size))
            self.layernorms.append(nn.LayerNorm(layer_size, elementwise_affine=False))

    def forward(self, X):
        X = X.flatten(start_dim=1)
        
        sum_squared_activities = []
        for index, linear_layer in enumerate(self.linear_layers):
            X = linear_layer(X)
            X = self.layernorms[index](X).relu()
            sum_squared_activities.append(torch.sum(X**2))
            X = X.detach()

        return X, sum_squared_activities
   
def evaluate(model: FFmodel, 
         dataset: datasets.MNIST):
    correct = 0
    with torch.no_grad():
        for _, (image, label) in enumerate(dataset):
            
            activities_list = []
            for c in range(10):
                image[:,0,:10] = torch.zeros_like(image[:,0,:10])
                image[:,0,c] = 1

                _, activities = model(image)
                activities_list.append((c, activities[1] + activities[2] + activities[3]))
            predicted_label = max(activities_list, key=lambda tup: tup[1])[0]
            
            if(predicted_label == label):
                correct += 1

    return correct/len(dataset)