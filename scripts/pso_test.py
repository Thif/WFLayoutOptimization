
# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps

#Import PyGad
import pygad

def func(x):
    return (x-2)**-2

def fx(x):
    #this will return 1/AEP
    return 1/np.sum(func(x))

def fx_gan(x,idx):
    #this will return AEP
    return np.sum(func(x))

# Create bounds
max_bound = 1.5*np.ones(1)
min_bound = 0.5*np.ones(1)
bounds = (min_bound, max_bound)

options = {'c1': 0.5, 'c2': 0.5, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=1, options=options)

# Perform optimization
cost, pos = optimizer.optimize(fx, iters=100)

############## Using GA ################
fitness_function = fx_gan

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = 1

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))