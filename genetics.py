from pyeasyga import pyeasyga
import numpy as np
import random
from cifar_net import Net, train, computePerformance

class GeneticFit():
    def __init__(self, trainloader, validationloader, max_iter, pop_size, num_generations):
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.num_generations = num_generations
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.seen = 0
        self.gens = 0
    
    def fitness(self, individual, data):
        model = Net(individual)
        train(model, self.trainloader)
        
        fitness = computePerformance(model, self.validationloader)

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
        print("number of layers is", num_layers)
        dimensions = [random.randint(1, 11)*10 for i in range(num_layers)]
        return dimensions
    
    def mutate(self, individual):
        # randomly mutate the number of layers or the size of a layer
        switch = random.randint(0, 1)
        layer_index = random.randrange(len(individual))

        if (switch == 0):
            switch = random.randint(0, 1)
            if (switch == 0):
                individual.insert(layer_index, random.randint(1, 11)*10)
            else:
                if (len(individual) > 1):
                    del(individual[layer_index])
        else:
            layer_dimension_change = random.randint( 1 - individual[layer_index], individual[layer_index] - 1)
            individual[layer_index] += layer_dimension_change
    
    def crossover(self, parent_1, parent_2):
        # get index of shorter parent to ensure crossover can occur
        min_parent_size = np.min([len(parent_1), len(parent_2)])
        crossover_index = random.randint(1, min_parent_size)

        child_1 = parent_1[:crossover_index] + parent_2[crossover_index:]
        child_2 = parent_2[:crossover_index] + parent_1[crossover_index:]
        return child_1, child_2
    
    def geneticFit(self):
        data = self.trainloader
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

    
