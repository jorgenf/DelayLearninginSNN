import functions
import Population
import numpy as np


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    seeds = rng.integers(low=0, high=10000000, size=3)
    pop = Population.Population((7,Population.RS))

    pop.create_barabasi_albert_connections(d=list(range(1,10)), w=16, trainable=True, seed=seeds[0])

    inp1 = pop.create_input(p=0.5, seed=seeds[1])
    inp2 = pop.create_input(p=0.5, seed=seeds[2])

    pop.create_synapse(inp1.ID, 3, w=16)
    pop.create_synapse(inp1.ID, 5, w=16)
    pop.create_synapse(inp2.ID, 1, w=16)
    pop.create_synapse(inp2.ID, 3, w=16)

    pop.run(30, 0.1, plot_network=True)
    functions.create_video(1)
    pop.plot_delays()
    pop.plot_raster()





