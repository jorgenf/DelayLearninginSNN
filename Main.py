import Population
import Simulations as sim
#import Population
import numpy as np
import multiprocessing as mp



if __name__ == "__main__":

    t= 10000
    sim.run_xnxi_async(t, 4, 2, 1, 1, [20, 20.1], [1, 1.1], f"low_delay")
    sim.run_xnxi_async(t, 4, 2, 1, 1, [20, 20.1], [20, 20.1], f"mid_delay")
    sim.run_xnxi_async(t, 4, 2, 1, 1, [20, 20.1], [40, 40.1], f"high_delay")

    sim.run_xnxi_async(t, 4, 2, 1, 1, [40, 40.1], [1, 1.1], f"low_delay")
    sim.run_xnxi_async(t, 4, 2, 1, 1, [40, 40.1], [20, 20.1], f"mid_delay")
    sim.run_xnxi_async(t, 4, 2, 1, 1, [40, 40.1], [40, 40.1], f"high_delay")

    sim.run_xnxi_async(t, 4, 2, 1, 1, [60, 60.1], [1, 1.1], f"low_delay")
    sim.run_xnxi_async(t, 4, 2, 1, 1, [60, 60.1], [20, 20.1], f"mid_delay")
    sim.run_xnxi_async(t, 4, 2, 1, 1, [60, 60.1], [40, 40.1], f"high_delay")

    '''
    t = 10000
    rng = np.random.default_rng(115)
    low_d = [(t, 4, 2, rng.integers(0, 100000), rng.integers(0, 100000), [30, 61], [1, 1.1], f"low_delay_{x}") for x in range(100)]
    mid_d = [(t, 4, 2, rng.integers(0, 100000), rng.integers(0, 100000), [30, 61], [20, 20.1], f"mid_delay_{x}") for x in range(100)]
    high_d = [(t, 4, 2, rng.integers(0, 100000), rng.integers(0, 100000), [30, 61], [40, 40.1], f"high_delay_{x}") for x in range(100)]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, low_d)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, mid_d)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, high_d)
'''
# plot_spike_rate_data(f"C:/Users/{os.environ.get('USERNAME')}/Documents/MASTER THESIS - SIMULATION RESULTS/feed forward", "t10000", "4n2i")
# plot_delay_categories("C:/Users/jorge/Documents/MASTER THESIS - SIMULATION RESULTS/feed forward", "t10000", "4n2i")
# delete_compile_sum_data("C:/Users/jorge/Documents/MASTER THESIS - SIMULATION RESULTS/feed forward", t_folder="t10000")

    '''
    rng = np.random.default_rng(114)
    t = 10000
    delay_seeds = rng.integers(0,1000, size=6)
    in1i2_alt_seeds = [(t, 1, 2, 200, 200, delay_seeds[0], rng.integers(0, 100), f"1n2i_alt_i_{x}") for x in range(100)]
    in1i2_async_seeds = [(t, 1, 2, delay_seeds[1], rng.integers(0, 100), f"1n2i_async_i_{x}") for x in range(100)]
    in1i2_rep_seeds = [(t, 1, 2, delay_seeds[2], rng.integers(0, 100), f"1n2i_rep_i_{x}") for x in range(100)]
    in1i3_alt_seeds = [(t, 1, 3, 200, 200, delay_seeds[0], rng.integers(0,100), f"1n3i_alt_i_{x}") for x in range(100)]
    in1i3_async_seeds = [(t, 1, 3, delay_seeds[1], rng.integers(0, 100), f"1n3i_async_i_{x}") for x in range(100)]
    in1i3_rep_seeds = [(t, 1, 3, delay_seeds[2], rng.integers(0, 100), f"1n3i_rep_i_{x}") for x in range(100)]
    in4i2_alt_seeds = [(t, 4, 2, 200, 200, delay_seeds[3], rng.integers(0,100), f"4n2i_alt_i_{x}") for x in range(100)]
    in4i2_async_seeds = [(t, 4, 2, delay_seeds[4], rng.integers(0, 100), f"4n2i_async_i_{x}") for x in range(100)]
    in4i2_rep_seeds = [(t, 4, 2, delay_seeds[5], rng.integers(0, 100), f"4n2i_rep_i_{x}") for x in range(100)]

    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_rep, in1i2_rep_seeds)

    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_alt, in1i2_alt_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_async, in1i2_async_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_async, in1i3_async_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_rep, in1i3_rep_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_alt, in1i3_alt_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_rep, in4i2_rep_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_alt, in4i2_alt_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_async, in4i2_async_seeds)

    input_seeds = rng.integers(0, 1000, size=6)
    dn1i2_alt_seeds = [(t, 1, 2, 200, 200, rng.integers(0, 100), input_seeds[3], f"1n2i_alt_d_{x}") for x in range(100)]
    dn1i2_async_seeds = [(t, 1, 2, rng.integers(0, 100), input_seeds[4], f"1n2i_async_d_{x}") for x in range(100)]
    dn1i2_rep_seeds = [(t, 1, 2, rng.integers(0, 100), input_seeds[5], f"1n2i_rep_d_{x}") for x in range(100)]
    dn1i3_alt_seeds = [(t, 1, 3, 200, 200, rng.integers(0,100), input_seeds[0], f"1n3i_alt_d_{x}") for x in range(100)]
    dn1i3_async_seeds = [(t, 1, 3, rng.integers(0, 100), input_seeds[1], f"1n3i_async_d_{x}") for x in range(100)]
    dn1i3_rep_seeds = [(t, 1, 3, rng.integers(0, 100), input_seeds[2], f"1n3i_rep_d_{x}") for x in range(100)]
    dn4i2_alt_seeds = [(t, 4, 2, 200, 200, rng.integers(0,100), input_seeds[3], f"4n2i_alt_d_{x}") for x in range(100)]
    dn4i2_async_seeds = [(t, 4, 2, rng.integers(0, 100), input_seeds[4], f"4n2i_async_d_{x}") for x in range(100)]
    dn4i2_rep_seeds = [(t, 4, 2, rng.integers(0, 100), input_seeds[5], f"4n2i_rep_d_{x}") for x in range(100)]

    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_rep, dn1i2_rep_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, dn1i2_alt_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, dn1i2_async_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, dn1i3_async_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_rep, dn1i3_rep_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, dn1i3_alt_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_rep, dn4i2_rep_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, dn4i2_alt_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, dn4i2_async_seeds)

    din1i2_alt_seeds = [(t, 1, 2, 200, 200, rng.integers(0, 100), rng.integers(0, 100), f"1n2i_alt_di_{x}") for x in range(100)]
    din1i2_async_seeds = [(t, 1, 2, rng.integers(0, 100), rng.integers(0, 100), f"1n2i_async_di_{x}") for x in range(100)]
    din1i2_rep_seeds = [(t, 1, 2, rng.integers(0, 100), rng.integers(0, 100), f"1n2i_rep_di_{x}") for x in range(100)]
    din1i3_alt_seeds = [(t, 1, 3, 200, 200, rng.integers(0,100), rng.integers(0, 100), f"1n3i_alt_di_{x}") for x in range(100)]
    din1i3_async_seeds = [(t, 1, 3, rng.integers(0, 100), rng.integers(0, 100), f"1n3i_async_di_{x}") for x in range(100)]
    din1i3_rep_seeds = [(t, 1, 3, rng.integers(0, 100), rng.integers(0, 100), f"1n3i_rep_di_{x}") for x in range(100)]
    din4i2_alt_seeds = [(t, 4, 2, 200, 200, rng.integers(0,100), rng.integers(0, 100), f"4n2i_alt_di_{x}") for x in range(100)]
    din4i2_async_seeds = [(t, 4, 2, rng.integers(0, 100), rng.integers(0, 100), f"4n2i_async_di_{x}") for x in range(100)]
    din4i2_rep_seeds = [(t, 4, 2, rng.integers(0, 100), rng.integers(0, 100), f"4n2i_rep_di_{x}") for x in range(100)]


    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_rep, din1i2_rep_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_alt, din1i2_alt_seeds)
    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(sim.run_xnxi_async, din1i2_async_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_rep, din1i3_rep_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, din1i3_alt_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, din1i3_async_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_rep, din4i2_rep_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, din4i2_alt_seeds)
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, din4i2_async_seeds)



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


pop = Population.Population((10,Population.RS))
pop.create_directional_ring_lattice_connections(2,[round(x/10,1) for x in range(1,60)], 16, trainable=True, seed=10)
inp = pop.create_input(p=0.1)
pop.create_synapse(inp.ID, 0, w=60)
pop.create_synapse(inp.ID, 4, w=60)
pop.run(1000, plot_network=True)
pop.show_network(save=True)
pop.plot_raster()
pop.plot_delays()
pop.plot_membrane_potential()
pop.create_video(30)
'''