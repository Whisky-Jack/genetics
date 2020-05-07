from pyeasyga import pyeasyga
import numpy as np
import random
from cifar_net import DefaultNet, Net, train, computePerformance

class GeneticFit():
    def __init__(self, trainloader, validationloader, num_epochs, max_iter, pop_size, num_generations):
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.num_epochs = num_epochs
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.seen = 0
        self.gens = 0
    
    def fitness(self, individual, data):
        train(individual, self.trainloader, self.num_epochs)
        
        fitness = computePerformance(individual, self.validationloader)

        self.seen += 1
        
        if (self.seen == self.pop_size):
            self.gens += 1
            print("Generation ", self.gens, " complete")
            print("###############################################################")
            self.seen = 0

        return fitness
    
    def create_individual(self, data):
        # Generate random number of layers and dimensions
        num_layers = random.randint(1, 2)
        layers = [random.randint(1, 11)*10 for i in range(num_layers)]
        print("layer dimensions are", layers)

        individual = Net(layers)
        return individual
    
    def mutate(self, individual):
        # randomly mutate the number of layers or the size of a layer
        individual.mutate_layout()
    
    def crossover(self, parent_1, parent_2):
        # get index of shorter parent to ensure crossover can occur
        min_parent_size = np.min([len(parent_1), len(parent_2)])
        crossover_index = random.randint(1, min_parent_size)

        child_1_layers = parent_1[:crossover_index] + parent_2[crossover_index:]
        child_2_layers = parent_2[:crossover_index] + parent_1[crossover_index:]
        return child_1, child_2
    
    def geneticFit(self, pretrained = False):
        data = self.trainloader


        print("Training default network")
        # ifnotdef, train model, else use pretrained model
        net = DefaultNet()
        train(net, trainloader, num_epochs=100)
        computePerformance(net, testloader)
        print("Base training complete")

        ga = pyeasyga.GeneticAlgorithm(data,
                               population_size=self.pop_size,
                               generations=self.num_generations,
                               crossover_probability=0.8,
                               mutation_probability=0.05,
                               elitism=True,
                               maximise_fitness=True)

        ga.create_individual = self.create_individual
        ga.fitness_function = self.fitness
        ga.mutate_function = self.mutate
        ga.crossover_function = self.crossover

        ga.run()

        print("Best individual is: ")
        print(ga.best_individual()[1])
        #print(self.fitness(ga.best_individual()[1], self.X))
        return ga.best_individual()

    
