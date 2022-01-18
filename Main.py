import functions
import Population
import numpy as np


'''
rng = np.random.default_rng(7)
seeds = rng.integers(low=0, high=10000000, size=3)
pop = Population.Population((7,Population.RS))

pop.create_ring_lattice_connections(1, 3,50,True)
inp1 = pop.create_input(p=0.1, seed=seeds[1])

pop.create_synapse(inp1.ID, 0, w=50)

pop.run(100, 0.1, plot_network=True)
pop.create_video(10)
pop.plot_delays()
pop.plot_raster()

'''

rng = np.random.default_rng(3)
pop = Population.Population((9, Population.RS))
layer = 0
for i in pop.neurons:
    if int(i) == 6:
        break
    if int(i) % 3 == 0:
        layer += 1
    for j in range(3*layer, 3*layer + 3):
        pop.create_synapse(i, j, w=16, d=rng.integers(10, 60)/10)

pop.structure= "grid"


time = 50000
s0 = []
s1 = []
s2 = []
for t in range(time):
    if t % 5 == 0 or t % 5 == 1 or t % 5 == 2:
        s0.append(t)
    if t % 8 == 0 or t % 8 == 1 or t % 8 == 2:
        s1.append(t)
    if t % 12 == 0 or t % 12 == 1 or t % 12 == 2:
        s2.append(t)


in0 = pop.create_input(s0)
in1 = pop.create_input(s1)
in2 = pop.create_input(s2)

pop.create_synapse(in0.ID,0, w=60, trainable=False)
pop.create_synapse(in1.ID,1, w=60, trainable=False)
pop.create_synapse(in2.ID,2, w=60, trainable=False)

pop.run(time, dt=0.1, plot_network=False)
pop.create_video(10)
pop.plot_delays()
pop.plot_raster()