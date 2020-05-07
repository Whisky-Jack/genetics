import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#######################################
# CONSTRUCTION
#######################################
# DEFAULT CONSTRUCTOR
class DefaultNet(nn.Module):
    def __init__(self):
        super(DefaultNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 30, 5)
        self.fc1 = nn.Linear(30 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CUSTOM CONSTRUCTOR
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 30, 5)

        # initialize layers from input
        layers.insert(0, 30 * 5 * 5)
        layers.insert(len(layers), 10)

        fcs = []
        for i in range(len(layers) - 1):
            fcs.append(nn.Linear(layers[i], layers[i + 1]))
            print(self.fcs[i].weight.data)
        
        self.fcs = fcs

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 30 * 5 * 5)
        for i in range(len(self.fcs) - 1):
            fc = self.fcs[i]
            x = F.relu(fc(x))
        final_layer = self.fcs[-1]
        x = final_layer(x)
        return x
    
    def adopt_new_layout(self, new_layers):
        switch = random.randint(0, 1)
        if (switch == 0):
            switch = random.randint(0, 1)
            if (switch == 0):
                add_layer()
            else:
                if (len(new_layers) > 1):
                    remove_layer()
        else:
            mutate_layer_size()
        
        # This fucks up two layers
        # The layers between the layer before and after the index

        # surgically update fcs
    
    def add_layer(self):
        new_layers = self.layers

        layer_index = random.randrange(len(new_layers))
        layer_size = random.randint(1, 11)*10

        new_layers.insert(layer_index, layer_size)
        fcs = self.fcs
        
        old_layer = self.fcs[layer_index]
        weight_value = old_layer.weight.data

        if (layer_index > 0 ):
            init_weights = weight_value
            new_layer = nn.Linear(layers[layer_index - 1], layers[layer_index])

            print(type(new_layer))
            
            new_layer_data = new_layer.weight.data

            new_layer_data[:2, :4] = 5

            self.fcs[layer_index] = new_layer
        
        if (layer_index < len(new_layers) - 1):
            nn.Linear(layers[layer_index], layers[layer_index + 1])

        


    def remove_layer(self):
        new_layers = self.layers
        layer_index = random.randrange(len(new_layers))

        del(new_layers[layer_index])
    
    def mutate_layer_size(self):
        new_layers = self.layers
        layer_index = random.randrange(len(new_layers))
    
        layer_dimension_change = random.randint( 1 - new_layers[layer_index], new_layers[layer_index] - 1)
        new_layers[layer_index] += layer_dimension_change

        old_layer = self.fcs[layer_index]
        
        weight_value = old_layer.weight.data
        init_weights = weight_value

        new_layer = thing
        self.fcs[layer_index] = new_layer

        if (layer_index > 0 ):
            nn.Linear(layers[layer_index - 1], layers[layer_index])
        if (layer_index < len(new_layers) - 1):
            nn.Linear(layers[layer_index], layers[layer_index + 1])
        
        print("fat")
    
    def reshape_layer(self, )
#######################################
# TRAINING
#######################################

# train the neural net
def train(net, trainloader, num_epochs, save = False):
    net = net.to(device)

    # define the loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # loop over the training set
    for epoch in range(num_epochs):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    if (save):
        #save model
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)

#######################################
# PREDICTION
#######################################

#evaluate the network on the test data
def computePerformance(net, dataloader):
    #evaluate on entire test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = ( correct / total)
    print('Accuracy of the network on the 10000 test images: ', accuracy)
    return accuracy
