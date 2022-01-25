import functions as fn
#import Population
import numpy as np
import multiprocessing as mp

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


'''
def run_9_neuron_sim(delay_seed, input_seed):
    rng = np.random.default_rng(delay_seed)
    rng2 = np.random.default_rng(input_seed)
    pop = Population.Population((9, Population.RS))
    layer = 0
    for i in pop.neurons:
        if int(i) == 6:
            break
        if int(i) % 3 == 0:
            layer += 1
        for j in range(3*layer, 3*layer + 3):
            pop.create_synapse(i, j, w=16, d=rng.integers(1, 60)/10)

    pop.structure = "grid"


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
    
    t0 = rng2.integers(0, 6)
    t1 = rng2.integers(0, 6)
    t2 = rng2.integers(0, 6)
    period = rng2.integers(5,11)
    while True:
        s0.append(t0)
        s1.append(t1)
        s2.append(t2)
        t0 += period
        t1 += period
        t2 += period
        if max([max(s0), max(s1), max(s2)]) >= time:
            break
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
    pop.show_network(save=True)

'''



fn.run_xnxi_async(100, 9,3, 1, 1)

'''

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    seeds = [(2000, 100,10, rng.integers(0,100), rng.integers(0,100)) for x in range(9)]
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(fn.run_xnxi_async, seeds)
    

'''

