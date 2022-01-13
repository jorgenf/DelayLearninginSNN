import Population
import numpy as np
n = 81
pop = Population.Population((n, Population.RS))
d = [round(np.random.random() * 10,1) for x in range(n)]
w = 10
#pop.create_grid()
#pop.create_random_connections(0.1,d=d, w=w, trainable=False)
seed = np.random.seed(3)
pop.create_distance_probability_connections(p=0.1,w=10, seed=seed)
pop.show_network() 