from model import FFmodel, evaluate
import torch
import numpy as np
import random
from torchvision import datasets, transforms

if __name__ == '__main__':
    train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transforms.Compose([
                                                                                     transforms.ToTensor(),
                                                                                     transforms.Normalize((0.1307,), (0.3081,)),
                                                                                     transforms.Resize((14,14))
                                                                                ]))

    test_dataset = datasets.MNIST(root="data", download=True,train=False, transform=transforms.Compose([
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize((0.1307,), (0.3081,)),
                                                                                    transforms.Resize((14,14))
                                                                                ]))
    model = FFmodel(196, [40, 40, 40, 40])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    arr = np.zeros((10,9))
    for row_index in range(10):
        arr[row_index] = [i for i in range(10) if i != row_index]

    print("Beginning training")
    accuracy_history = []
    for epoch in range(40):
        for index, (image, label) in enumerate(train_dataset):
            
            image[:,0,:10] = torch.zeros_like(image[:,0,:10])
            if(random.uniform(0,1) < 0.5): # Positive example
                valency = -1
                image[:,0,label] = 1

            else: # Negative example
                valency = 1
                image[:,0, random.choice(arr[label]).astype(np.int16)] = 1
            
            optimizer.zero_grad()
            out, activities = model(image)
            for act in activities:
                (act * valency).backward()
                
            optimizer.step()
        
        accuracy = evaluate(model, test_dataset)
        print(f'Done epoch {epoch}, the accuracy is {accuracy}')
        accuracy_history.append(accuracy)