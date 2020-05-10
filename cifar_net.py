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
        full_layers = layers.copy()
        full_layers.insert(0, 30 * 5 * 5)
        full_layers.insert(len(layers) + 1, 10)

        fcs = []
        for i in range(len(full_layers) - 1):
            fcs.append(nn.Linear(full_layers[i], full_layers[i + 1]))
        
        self.fcs = fcs
    
    def __init__(self, layers, cifar_model):
        super(Net, self).__init__()
        self.layers = layers
        self.conv1 = cifar_model.conv1
        self.pool = cifar_model.pool
        self.conv2 = cifar_model.conv2

        # initialize layers from input
        full_layers = layers.copy()
        full_layers.insert(0, 30 * 5 * 5)
        full_layers.insert(len(layers) + 1, 10)

        fcs = []
        for i in range(len(full_layers) - 1):
            fcs.append(nn.Linear(full_layers[i], full_layers[i + 1]))
        
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
    
    def mutate_layout(self):
        switch = random.randint(0, 1)
        if (switch == 0):
            switch = random.randint(0, 1)
            if (switch == 0):
                self.add_layer()
            else:
                if (len(self.layers) > 1):
                    self.remove_layer()
        else:
            self.mutate_layer_size()
    
    def add_layer(self):
        # LAYERS [10, 30]
        # OLD_FULL_LAYERS [750, 10, 30, 10]
        # OLD_FCS [(750, 10), *(10, 30)*, (30, 10)]

        # INDEX = 1
        # NEW_LAYERS [10, 20, 30]
        # NEW_FULL_LAYERS [750, 10, 20, 30, 10]
        # NEW_FCS [(750, 10), *(10, 20)*, (20, 30), (30, 10)]
        # layer index i corresponds to fcs index i and i + 1

        new_layers = self.layers.copy()

        layer_index = random.randrange(len(new_layers))
        layer_size = random.randint(1, 11)*10

        new_layers.insert(0, 30 * 5 * 5)
        new_layers.insert(len(new_layers) + 1, 10)

        fcs = self.fcs
        
        # INITIALIZE BEFORE LAYER
        # Don't forget about biases
        old_layer = self.fcs[layer_index]
        old_layer_data = old_layer.weight.data
        new_layer = nn.Linear(new_layers[layer_index], layer_size) #new_layers[layer_index + 1])
        
        new_layer_data = new_layer.weight.data

        sizes = [old_layer_data.shape[0], new_layer_data.shape[0]]

        # Finds the smaller layer
        min_size, min_arg = min((val, idx) for (idx, val) in enumerate(sizes))
        opposite_arg = not min_arg

        # Obtains random sample of indices to include
        indices_to_copy = random.sample(range(sizes[opposite_arg]), min_size)

    
        if (opposite_arg):  # If old layer is smaller
            # Populate random indices of new layer with old layer indices
            new_layer.weight.data[indices_to_copy, :] = old_layer.weight.data[:, :]
        else:          # New layer is smaller
            # Populate new layer with random sample of indices of old layer
            new_layer.weight.data[:, :] = old_layer.weight.data[indices_to_copy, :]

        self.fcs[layer_index]= new_layer
        print(self.fcs)
        
        # INITIALIZE OLD LAYER

        new_layer = nn.Linear(layer_size, new_layers[layer_index + 1])
        new_layer_data = new_layer.weight.data

        sizes = [old_layer_data.shape[1], new_layer_data.shape[1]]

        # Finds the smaller layer
        min_size, min_arg = min((val, idx) for (idx, val) in enumerate(sizes))
        opposite_arg = not min_arg

        # Obtains random sample of indices to include
        indices_to_copy = random.sample(range(sizes[opposite_arg]), min_size)


        """
        print("new layers is ", new_layers)
        print("Connecting from ", new_layers[layer_index], "to ", layer_size)
        print("New layer size is ", layer_size)
        print("New layer index is ", layer_index)
        fuck = self.layers.copy()
        fuck.insert(layer_index, layer_size)
        fuck.insert(0, 30 * 5 * 5)
        fuck.insert(len(new_layers) + 1, 10)
        print("New layers will be", fuck)
        print("fcs: ", self.fcs)
        print("Shapes: ", old_layer_data.shape, new_layer_data.shape)
        print("Min size is: ", min_size)
        print("Indices to copy: ", len(indices_to_copy))
        """


        if (opposite_arg):  # If old layer is smaller
            # Populate random indices of new layer with old layer indices
            new_layer.weight.data[:, indices_to_copy] = old_layer.weight.data[:, :]
        else:          # New layer is smaller
            # Populate new layer with random sample of indices of old layer
            new_layer.weight.data[:, :] = old_layer.weight.data[:, indices_to_copy]

        self.fcs.insert(layer_index + 1, new_layer)
        new_layers.insert(layer_index, layer_size)



    def remove_layer(self):
        # NEW_LAYERS [10, 20, 30]
        # NEW_FULL_LAYERS [750, 10, 20, 30, 10]
        # NEW_FCS [(750, 10), *(10, 20)*, *(20, 30)*, (30, 10)]
        # layer index i corresponds to fcs index i and i + 1

        # INDEX = 1
        # LAYERS [10, 30]
        # OLD_FULL_LAYERS [750, 10, 30, 10]
        # OLD_FCS [(750, 10), *(10, 30)*, (30, 10)]


        new_layers = self.layers.copy()
        layer_index = random.randrange(len(new_layers))

        new_layers.insert(0, 30 * 5 * 5)
        new_layers.insert(len(new_layers) + 1, 10)

        new_layer = nn.Linear(new_layers[layer_index], new_layers[layer_index + 2])
        new_layer_data = new_layer.weight.data

        # Populate new layer with random weights from old layers
        before_layer_data = self.fcs[layer_index].weight.data
        after_layer_data = self.fcs[layer_index + 1].weight.data

        input_sizes = [before_layer_data.shape[0], new_layer_data.shape[0]]

        min_size, min_arg = min((val, idx) for (idx, val) in enumerate(input_sizes))
        opposite_arg = not min_arg

        # Obtains random sample of indices to include
        indices_to_copy = random.sample(range(input_sizes[opposite_arg]), min_size)

        """
        print("new layers is ", new_layers)
        print("New layer index is ", layer_index)
        fuck = self.layers.copy()
        fuck.insert(0, 30 * 5 * 5)
        fuck.insert(len(new_layers) + 1, 10)
        print("New layers will be", fuck)
        print("fcs: ", self.fcs)
        print("Shapes: ", before_layer_data.shape, new_layer_data.shape)
        print("Min size is: ", min_size)
        print("Indices to copy: ", len(indices_to_copy))
        """

        if (opposite_arg):  # If old layer is smaller
            # Populate random indices of new layer with old layer indices
            new_layer.weight.data[indices_to_copy, :] = before_layer_data[:, :]
        else:          # New layer is smaller
            # Populate new layer with random sample of indices of old layer
            new_layer.weight.data[:, :] = before_layer_data[indices_to_copy, :]
        

        #definitely wrong
        print("fcs: ", self.fcs)
        del(self.fcs[layer_index + 1])
        print("fcs: ", self.fcs)
        self.fcs[layer_index] = new_layer
        print("fcs: ", self.fcs)
    
    def mutate_layer_size(self):
        # NEW_LAYERS [10, 20, 30]
        # NEW_FULL_LAYERS [750, 10, 20, 30, 10]
        # NEW_FCS [(750, 10), *(10, 20)*, *(20, 30)*, (30, 10)]
        # layer index i corresponds to fcs index i and i + 1

        # INDEX = 1
        # NEW_LAYERS [10, 40, 30]
        # NEW_FULL_LAYERS [750, 10, 40, 30, 10]
        # NEW_FCS [(750, 10), *(10, 40)*, *(40, 30)*, (30, 10)]

        new_layers = self.layers.copy()
        layer_index = random.randrange(len(new_layers))
    
        layer_dimension_change = random.randint( 1 - new_layers[layer_index], new_layers[layer_index] - 1)
        new_layer_size = new_layers[layer_index] + layer_dimension_change

        new_layers.insert(0, 30 * 5 * 5)
        new_layers.insert(len(new_layers) + 1, 10)

        # Get relevant layers
        before_layer_data = self.fcs[layer_index].weight.data
        after_layer_data = self.fcs[layer_index + 1].weight.data

        # INITIALIZE LAYERS
        new_before_layer = nn.Linear(new_layers[layer_index], new_layer_size)
        new_after_layer = nn.Linear(new_layer_size, new_layers[layer_index + 2])

        input_sizes = [before_layer_data.shape[0], new_before_layer.weight.data.shape[0]]

        min_size, min_arg = min((val, idx) for (idx, val) in enumerate(input_sizes))
        opposite_arg = not min_arg

        # Obtains random sample of indices to include
        indices_to_copy = random.sample(range(input_sizes[opposite_arg]), min_size)

        """
        print("new layers is ", new_layers)
        print("New layer index is ", layer_index)
        print("New layer size is ", new_layer_size)
        fuck = self.layers.copy()
        fuck[layer_index] = new_layer_size
        fuck.insert(0, 30 * 5 * 5)
        fuck.insert(len(new_layers) + 1, 10)
        print("New layers will be", fuck)
        print("fcs: ", self.fcs)
        print("Shapes: ", before_layer_data.shape, new_before_layer.weight.data.shape)
        print("Min size is: ", min_size)
        print("Indices to copy: ", len(indices_to_copy))
        """

        if (opposite_arg):  # If old layer is smaller
            # Populate random indices of new layer with old layer indices
            new_before_layer.weight.data[indices_to_copy, :] = before_layer_data[:, :]
            new_after_layer.weight.data[:, indices_to_copy] = after_layer_data[:, :]
        else:          # New layer is smaller
            # Populate new layer with random sample of indices of old layer
            new_before_layer.weight.data[:, :] = before_layer_data[indices_to_copy, :]
            new_after_layer.weight.data[:, :] = after_layer_data[:, indices_to_copy]
        
        self.fcs[layer_index] = new_before_layer
        self.fcs[layer_index + 1] = new_after_layer
    
    def reshape_layer(self):
        print("Not implemented")
#######################################
# TRAINING
#######################################

# train the neural net
def train(net, trainloader, num_epochs, save = False):
    net.to(device)

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
    net = net.to(device)
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
