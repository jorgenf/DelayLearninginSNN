import Population
import Simulations as sim
#import Population
import numpy as np
import multiprocessing as mp



if __name__ == "__main__":
    rng = np.random.default_rng(112 )
    alt_seeds = [(5000, 1, 3, 200, 200, rng.integers(0,100), rng.integers(0,100), f"altn1i3_{x}") for x in range(100)]
    async_seeds = [(5000, 1, 3, rng.integers(0, 100), rng.integers(0, 100), f"asyncn1i3_{x}") for x in range(100)]
    rep_seeds = [(5000, 1, 3, rng.integers(0, 100), rng.integers(0, 100), f"repn1i3_{x}") for x in range(100)]
    n4i2_alt_seeds = [(5000, 4, 2, 200, 200, rng.integers(0,100), rng.integers(0,100), f"altn4i2_{x}") for x in range(100)]
    n4i2_async_seeds = [(5000, 4, 2, rng.integers(0, 100), rng.integers(0, 100), f"asyncn4i2_{x}") for x in range(100)]
    n4i2_rep_seeds = [(5000, 4, 2, rng.integers(0, 100), rng.integers(0, 100), f"repn4i2_{x}") for x in range(100)]
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_rep, n4i2_rep_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_alt, n4i2_alt_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_async, n4i2_async_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_async, async_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_rep, rep_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_alt, alt_seeds)
'''
input_rng = np.random.default_rng(4)
delay_rng = np.random.default_rng(3)
i = 1
t = 1000
n = 4
pop = Population.Population((n, Population.RS))
pop.create_ring_lattice_connections(1, d=list(range(1,5)) , w=32, trainable=True)
period = input_rng.integers(10, 20)
for x in range(i):
    offset = input_rng.integers(0, 6)
    pattern = [offset + (period * x) for x in range(int(np.ceil((t-offset)/period))) if offset + (period * x) < t]
    inp = pop.create_input(pattern)
    pop.create_synapse(inp.ID, 0, w=32, d=delay_rng.integers(1, 60) / 10)
pop.run(t, dt=0.1, plot_network=False)
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()
pop.show_network(save=True)
'''