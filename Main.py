import time
import Data
from Population import *
import Simulations as sim
#import Population
import numpy as np
import multiprocessing as mp
import itertools
import os
import re
import pandas as pd
import shutil
import json




#dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000"
#dir = "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSoff\\1n3i\\t10000"

#dirs = ["Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSoff\\1n3i\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSoff\\1n2i\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSoff\\1n3i w8\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSfrequency\\1n2i\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSfrequency\\1n3i\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\async rep alt"]


#dirs = ["Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSfrequency\\1n2i\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSfrequency\\1n3i\\t10000",
#        "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\async rep alt"]

#for d in dirs:
#    Data.reduce_sim_file_size(d)

#if __name__ == "__main__":
#    with mp.Pool(mp.cpu_count()-1) as p:
#        p.map(Data.reduce_sim_file_size, dirs)




'''
    # OK
    # DelayVSoff 1n3i w=8
if __name__ == "__main__":
    t = 10000
    off_range = list(range(0, 4))
    off_combos = list(itertools.product(off_range, off_range, off_range, off_range, off_range, off_range))
    delay_range = list(range(20, 22))
    delay_combos = list(itertools.product(delay_range, delay_range, delay_range))
    combos = list(itertools.product(off_combos, delay_combos))
    print("Combo length before: ", len(combos))
    dir = "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\\delayVSoff\\1n3i w8\\t10000\\"
    start = time.time()
    for dirs in os.listdir(dir):
        if dirs.startswith("delay"):
            off11 = int(re.findall(r'off11-([0-9]+)', dirs)[0])
            off12 = int(re.findall(r'off12-([0-9]+)', dirs)[0])
            off21 = int(re.findall(r'off21-([0-9]+)', dirs)[0])
            off22 = int(re.findall(r'off22-([0-9]+)', dirs)[0])
            off31 = int(re.findall(r'off31-([0-9]+)', dirs)[0])
            off32 = int(re.findall(r'off32-([0-9]+)', dirs)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', dirs)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', dirs)[0])
            d3 = int(re.findall(r'd3-([0-9]+)', dirs)[0])
            if ((off11, off12, off21, off22, off31, off32), (d1, d2, d3)) in combos:
                combos.remove(((off11, off12, off21, off22, off31, off32), (d1, d2, d3)))
    print("Combo length after: ", len(combos))
    start = time.time()
    params = [(dir, t, 1, 3, 500, 500, 1, 1, [x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]], [x[1][0], x[1][1], x[1][2]], [], f"delayVSoffw8_off11-{x[0][0]}_off12-{x[0][1]}_off21-{x[0][2]}_off22-{x[0][3]}_off31-{x[0][4]}_off32-{x[0][5]}_d1-{x[1][0]}_d2-{x[1][1]}_d3-{x[1][2]}") for x in combos]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, params)
    stop = round((time.time()-start)/60,1)
    print(stop)
'''

'''
    # OK
    # DelayVSfreq 1n3i w=8
if __name__ == "__main__":
    t = 10000
    freq_range = list(range(20, 26))
    freq_combos = list(itertools.product(freq_range, freq_range, freq_range))
    delay_range = list(range(18, 23))
    delay_combos = list(itertools.product(delay_range, delay_range, delay_range))
    combos = list(itertools.product(freq_combos, delay_combos))
    print("Combo length before: ", len(combos))
    dir = "Z:\\MASTER THESIS - SIMULATION RESULTS\\feed forward\delayVSfrequency\\1n3i w8\\t10000\\"
    for dirs in os.listdir(dir):
        if not "." in dirs:
            param = dirs
            f1 = int(re.findall(r'f1-([0-9]+)', param)[0])
            f2 = int(re.findall(r'f2-([0-9]+)', param)[0])
            f3 = int(re.findall(r'f3-([0-9]+)', param)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', param)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', param)[0])
            d3 = int(re.findall(r'd3-([0-9]+)', param)[0])
            if ((f1, f2, f3), (d1, d2, d3)) in combos:
                combos.remove(((f1, f2, f3), (d1, d2, d3)))
    print("Combo length after: ", len(combos))
    start = time.time()
    params = [(dir, t, 1, 3, 1, 1, [x[0][0], x[0][1], x[0][2]], [x[1][0], x[1][1], x[1][2]], [], [], f"delayVSfreqw8_f1-{x[0][0]}_f2-{x[0][1]}_f3-{x[0][2]}_d1-{x[1][0]}_d2-{x[1][1]}_d3-{x[1][2]}") for x in combos]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, params)
    stop = round((time.time()-start)/60,1)
    print(stop)
'''




'''
    # OK
    # DelayVSoff 1n3i
if __name__ == "__main__":
    t = 10000
    off_range = list(range(0, 4))
    off_combos = list(itertools.product(off_range, off_range, off_range, off_range, off_range, off_range))
    delay_range = list(range(20, 22))
    delay_combos = list(itertools.product(delay_range, delay_range, delay_range))
    combos = list(itertools.product(off_combos, delay_combos))
    print("Combo length before: ", len(combos))
    dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000/"
    start = time.time()
    for dirs in os.listdir(dir):
        if not "." in dirs:
            param = dirs
            off11 = int(re.findall(r'off11-([0-9]+)', param)[0])
            off12 = int(re.findall(r'off12-([0-9]+)', param)[0])
            off21 = int(re.findall(r'off21-([0-9]+)', param)[0])
            off22 = int(re.findall(r'off22-([0-9]+)', param)[0])
            off31 = int(re.findall(r'off31-([0-9]+)', param)[0])
            off32 = int(re.findall(r'off32-([0-9]+)', param)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', param)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', param)[0])
            d3 = int(re.findall(r'd3-([0-9]+)', param)[0])
            if ((off11, off12, off21, off22, off31, off32), (d1, d2, d3)) in combos:
                combos.remove(((off11, off12, off21, off22, off31, off32), (d1, d2, d3)))
    print("Combo length after: ", len(combos))
    start = time.time()
    params = [(dir, t, 1, 3, 500, 500, 1, 1, [x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5]], [x[1][0], x[1][1], x[1][2]], [], f"delayVSoff_off11-{x[0][0]}_off12-{x[0][1]}_off21-{x[0][2]}_off22-{x[0][3]}_off31-{x[0][4]}_off32-{x[0][5]}_d1-{x[1][0]}_d2-{x[1][1]}_d3-{x[1][2]}") for x in combos]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, params)
    stop = round((time.time()-start)/60,1)
    print(stop)
'''


'''
    # OK
    # DelayVSoff 1n2i
if __name__ == "__main__":
    t = 10000
    off_range = list(range(0, 6))
    off_combos = list(itertools.product(off_range, off_range, off_range, off_range))
    delay_range = list(range(18, 23))
    delay_combos = list(itertools.product(delay_range, delay_range))
    combos = list(itertools.product(off_combos, delay_combos))
    print("Combo length before: ", len(combos))
    dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000/"
    for dirs in os.listdir(dir):
        if not "." in dirs:
            param = dirs
            off11 = int(re.findall(r'off11-([0-9]+)', param)[0])
            off12 = int(re.findall(r'off12-([0-9]+)', param)[0])
            off21 = int(re.findall(r'off21-([0-9]+)', param)[0])
            off22 = int(re.findall(r'off22-([0-9]+)', param)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', param)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', param)[0])
            if ((off11, off12, off21, off22), (d1, d2)) in combos:
                combos.remove(((off11, off12, off21, off22), (d1, d2)))
    print("Combo length after: ", len(combos))
    start = time.time()
    params = [(dir, t, 1, 2, 500, 500, 1, 1, [x[0][0], x[0][1], x[0][2], x[0][3]], [x[1][0], x[1][1]], [], f"delayVSoff_off11-{x[0][0]}_off12-{x[0][1]}_off21-{x[0][2]}_off22-{x[0][3]}_d1-{x[1][0]}_d2-{x[1][1]}") for x in combos]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_alt, params)
    stop = round((time.time()-start)/60,1)
    print(stop)
'''

'''
    # OK
    # DelayVSfreq 1n3i
if __name__ == "__main__":
    t = 10000
    freq_range = list(range(20, 26))
    freq_combos = list(itertools.product(freq_range, freq_range, freq_range))
    delay_range = list(range(18, 23))
    delay_combos = list(itertools.product(delay_range, delay_range, delay_range))
    combos = list(itertools.product(freq_combos, delay_combos))
    print("Combo length before: ", len(combos))
    dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n3i/t10000/"
    for dirs in os.listdir(dir):
        if not "." in dirs:
            param = dirs
            f1 = int(re.findall(r'f1-([0-9]+)', param)[0])
            f2 = int(re.findall(r'f2-([0-9]+)', param)[0])
            f3 = int(re.findall(r'f3-([0-9]+)', param)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', param)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', param)[0])
            d3 = int(re.findall(r'd3-([0-9]+)', param)[0])
            if ((f1, f2, f3), (d1, d2, d3)) in combos:
                combos.remove(((f1, f2, f3), (d1, d2, d3)))
    print("Combo length after: ", len(combos))
    start = time.time()
    params = [(dir, t, 1, 3, 1, 1, [x[0][0], x[0][1], x[0][2]], [x[1][0], x[1][1], x[1][2]], [], [], f"delayVSfreq_f1-{x[0][0]}_f2-{x[0][1]}_f3-{x[0][2]}_d1-{x[1][0]}_d2-{x[1][1]}_d3-{x[1][2]}") for x in combos]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, params)
    stop = round((time.time()-start)/60,1)
    print(stop)
'''


'''
    # OK
    # DelayVSfreq 1n2i
if __name__ == "__main__":
    t = 10000
    freq_range = list(range(20, 31))
    freq_combos = list(itertools.product(freq_range, freq_range))
    delay_range = list(range(10,31))
    delay_combos = list(itertools.product(delay_range, delay_range))
    combos = list(itertools.product(freq_combos, delay_combos))
    print("Combo length before: ", len(combos))
    dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n2i/t10000/"
    for dirs in os.listdir(dir):
        param = dirs
        if not "." in param:
            f1 = int(re.findall(r'f1-([0-9]+)', param)[0])
            f2 = int(re.findall(r'f2-([0-9]+)', param)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', param)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', param)[0])
            if ((f1, f2), (d1, d2)) in combos:
                combos.remove(((f1, f2), (d1, d2)))
    print("Combo length after: ", len(combos))
    start = time.time()
    params = [(dir, t, 1, 2, 1, 1, [x[0][0], x[0][1]], [x[1][0], x[1][1]], [], [], f"delayVSfreq_f1-{x[0][0]}_f2-{x[0][1]}_d1-{x[1][0]}_d2-{x[1][1]}") for x in combos]
    with mp.Pool(mp.cpu_count() - 1) as pool:
        pool.starmap(sim.run_xnxi_async, params)
    stop = round((time.time()-start)/60,1)
    print(stop)
'''



'''
comp_dirs = ["Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000",
             "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000",
             "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i w8/t10000",
             "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n2i/t10000",
             "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n3i/t10000",
             "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n3i w8/t10000"]
'''
'''
for dr in os.listdir("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n2i/t10000"):
    if "." not in dr:
        with open(os.path.join("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n2i/t10000", dr, "synapse_data.json"), "r") as file:
            try:
                data = json.load(file)
            except:
                print(dr)
            if str(data)[-1] != "}":
                print(dr)
'''


'''
def do_stuff(dr):
    for f in os.listdir(dr):
        if "." not in f:
            with open(os.path.join(dr,f, "synapse_data.json"), "r") as file:
                data = json.loads(file.read())
                for k in data.keys():
                    if type(data[k]["d_hist"]) == dict:
                        l = data[k]["d_hist"]["d"]
                    else:
                        l = data[k]["d_hist"]


if __name__ == "__main__":
    with mp.Pool(os.cpu_count() - 1) as p:
        p.map(do_stuff, comp_dirs)

'''






'''

if __name__ == "__main__":
    with mp.Pool(os.cpu_count() - 1) as p:
        p.map(Data.compile_simulation_data, comp_dirs)


if __name__ == "__main__":
    with mp.Pool(os.cpu_count() - 1) as p:
        p.map(Data.sum_simulation_data, comp_dirs)

'''

#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i w8/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n2i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n3i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n3i w8/t10000")

#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n2i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i w8/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000")



#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000", file_title="delayVSfreq1n2i_delayplot", identifier_title="Spike intervals (ms)", identifiers=["f1", "f2"], nd=2)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000", file_title="delayVSfreq1n3i_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000", file_title="delayVSfreq1n3i_w8_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n2i/t10000", file_title="delayVSoff1n2i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22"], nd=2)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i/t10000", file_title="delayVSoff1n3i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i w8/t10000", file_title="delayVSoff1n3i_w8_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)


#Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000", file_title="delayVSoff1n2i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22"], nd=2)
#Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000", file_title="delayVSoff1n3i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i w8/t10000", file_title="delayVSoff1n3i_w8_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000", file_title="delayVSfreq1n2i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2"], nd=2)
#Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000", file_title="delayVSfreq1n3i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)
#Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000", file_title="delayVSfreq1n3i_w8_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)

'''
pop = Population.Population((10, Population.RS), name="ring_n10i1_k1_1conn_lowsr")
pop.create_ring_lattice_connections(k = 2, d=list(range(10,31)),w=30, trainable=True, seed=2)
spike_t = [t for t in range(1, 10000) if t % 50 == 0]

pop.create_input(spike_t, j=[5], wj=30)
pop.run(os.getcwd(), 1000)
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()
pop.show_network(save=True)
'''




'''
#def just_do_it(i,j):
t = 50000
k = 2
n = 10
#internal_d = 20
internal_d = [5 for x in range(20)]

internal_w = 16
input_d = [1, 2]
input_w = [32, 16]
input_connections = [0, 1]
input_freq = 5

pop = Population.Population((n, Population.RS), path="./network_plots", name=f"dir_RL_t{t}_{n}n1i_k{k}_d{internal_d}_w{internal_w}_inpd{input_d}_inpw{input_w}_freq{input_freq}_conn{input_connections}")
pop.create_directional_ring_lattice_connections(k=k, d=internal_d, w=internal_w, trainable=True)
pop.create_synapse(3,6, w=16,d=5)
pop.create_synapse(1,7, w=16,d=5)
spike_train = [x for x in range(0, t) if x % input_freq == 0]
pop.create_input(spike_train, j=input_connections, wj=input_w, dj=input_d)
pop.run(t, save_post_model=True)
pop.plot_topology()
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()

#i = [x for x in range(1,20)]
#if __name__ == "__main__":
#    with mp.Pool(os.cpu_count() - 1) as p:
#        p.map(just_do_it, i)
'''


'''


post_model = Data.load_model("Z:\\MASTER THESIS - SIMULATION RESULTS\\ring lattice\homogenous delays\\dir_RL_t50000_10n1i_k2_d[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]_w16_inpd[29, 60]_inpw[32, 16]_freq5_conn[0, 1]/post_sim_model.pkl")
post_model.plot_raster(duration=[30000,31000])
'''

'''
pre_model = Data.load_model("network_plots/directional_ring_lattice_10n1i_k2_d[29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60]_w16_inpd[29, 60]_inpw[30, 16]_freq[0, 600]_conn[0, 1]_1/pre_sim_model.pkl")
post_model = Data.load_model("network_plots/directional_ring_lattice_10n1i_k2_d[29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60]_w16_inpd[29, 60]_inpw[30, 16]_freq[0, 600]_conn[0, 1]_1/post_sim_model.pkl")

print("Spikes n8: ", post_model.neurons["8"].spikes)
print("Delay 8-0 at t=277.2: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 8 and syn.j == 0][0][2772])

print("Spikes n9: ", post_model.neurons["9"].spikes)
print("Delay 9-0 at t=308.0: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 9 and syn.j == 0][0][3080])
print("Delay 9-1 at t=308.0: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 9 and syn.j == 1][0][3080])

print("Spikes n0: ", post_model.neurons["0"].spikes)
print("Delay 0-1 at t=338.8: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 0 and syn.j == 1][0][3388])


for v,u,t in zip(post_model.neurons["0"].v_hist["v"][3270:3390],post_model.neurons["0"].u_hist["u"][3270:3390], post_model.neurons["0"].u_hist["t"][3270:3390]):
    print(v,u, t)

print("-------------------")

for v,u,t in zip(post_model.neurons["1"].v_hist["v"][3600:3800],post_model.neurons["1"].u_hist["u"][3600:3800], post_model.neurons["1"].u_hist["t"][3600:3800]):
    print(v,u, t)
'''

'''
pop = Population.Population((1, Population.RS), path="network_plots/", name="TEST")

#pop.create_input(spike_times=[0, 8.9, 8.9+30, 9++30+8.9 ], dj=0.1, wj=16, j=[0])
#pop.create_input(spike_times=[0, 55], dj=0.1, wj=17.8, j=[0])
interval = 90
ofset =
pop.create_input(spike_times=[0], dj=0.1, wj=32, j=[0])
pop.create_input(spike_times=[interval + 0, interval + ofset], dj=0.1, wj=16, j=[0])
pop.run(duration=1000)
pop.plot_membrane_potential(IDs=[0])

#print(pop.neurons["0"].v_hist["v"])

print(pop.neurons["0"].v_hist["v"][-1])
for i, v in zip(pop.neurons["0"].v_hist["t"], pop.neurons["0"].v_hist["v"]):
    print(i,v)
    if v >= -75 and i > 100:
        print(i)
        break
'''


'''
if __name__ == "__main__":
    sim.refractory_period_mt()
'''
'''
interval = 2
offset = 0
pop = Population((1, RS), path="network_plots/", name="TESTYTEST", save_data=True)
pop.create_input(spike_times=[0], dj=0.1, wj=32, j=[0])
pop.create_input(spike_times=[interval], dj=0.1, wj=16, j=[0])
pop.create_input(spike_times=[interval + offset], dj=0.1, wj=16, j=[0])
pop.run(duration=600, show_process=False)
pop.plot_raster()
pop.plot_delays()
pop.plot_membrane_potential()
'''

#if __name__ == "__main__":
#    sim.weight_shift_response_mt()



'''
t = 50000
k = 2
n = 10
internal_d = 5
internal_w = 16
input_d = [1, 2]
input_w = [32, 16]
input_conn = [0, 1]
input_spike_train = Data.create_alternating_input(2, 50000)

pop = Population((n, RS), path="./network_plots",
                 name=f"ringlattice_alternating_input")
pop.create_directional_ring_lattice_connections(k=k, d=internal_d, w=internal_w, trainable=True)
pop.create_synapse(3, 6, w=16, d=5)
pop.create_synapse(1, 7, w=16, d=5)
pop.create_input(input_spike_train[0], j=input_conn, wj=input_w, dj=input_d)
pop.create_input(input_spike_train[1], j=input_conn, wj=input_w, dj=input_d)
pop.run(t, save_post_model=True)
pop.plot_topology()
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()
'''

'''
t = 50000
n = 16
internal_d = list(range(1,10))
internal_w = 16
input_d = 1
input_w = 32

input_spike_train = Data.create_asynchronous_input(4, 50000)

pop = Population((n, RS), path="./network_plots",
                 name=f"ff_asynchronous_input")
pop.create_feed_forward_connections(d=internal_d, w=internal_w, trainable=True, seed=1)

pop.create_input(input_spike_train[0], j=[0], wj=input_w, dj=input_d)
pop.create_input(input_spike_train[1], j=[1], wj=input_w, dj=input_d)
pop.create_input(input_spike_train[2], j=[2], wj=input_w, dj=input_d)
pop.create_input(input_spike_train[3], j=[3], wj=input_w, dj=input_d)
pop.run(t, save_post_model=True)
pop.plot_topology()
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()
'''


t = 10000
n = 20
internal_d = list(range(1,10))
internal_w = 16
input_d = 1
input_w = 32
rng = np.random.default_rng(3)


#input_spike_train = Data.create_repeating_input(4, 50000, 100, seed=1)
#input_spike_train = Data.create_alternating_input(4, 50000, l_pattern=30, l_interm=100)
s1 = 2
s2 = 5
s3 = 8
s4 = 12

spike_train1 = []
spike_train2 = []
spike_train3 = []
spike_train4 = []
for pp in range(0, int(t/2), 200):
    spike_train1.append(s1 + pp + rng.integers(-1,2))
    spike_train2.append(s2 + pp + rng.integers(0,3))
    spike_train3.append(s3 + pp + rng.integers(0,4))
    spike_train4.append(s4 + pp + rng.integers(0,4))

s1 = 12
s2 = 9
s3 = 4
s4 = 0
for pp in range(int(t / 2)+ 200, t, 200):
    spike_train1.append(s1 + pp + rng.integers(0, 2))
    spike_train2.append(s2 + pp + rng.integers(0, 3))
    spike_train3.append(s3 + pp + rng.integers(0, 5))
    spike_train4.append(s4 + pp + rng.integers(0, 4))

    #s2 += 4
    #s3 += 6
    #s4 += 8

#spike_train1 = [5,  204,406]
#spike_train2 = [8, 209, 401]
#spike_train3 = [3, 104, 209, 309, 402]
#spike_train4 = [11, 112, 202, 310, 411]

pop = Population((n, RS), path="./network_plots",
                 name=f"xpolytest")
pop.create_random_connections(p=0.2, d=internal_d, w=internal_w, trainable=True, seed=3)

pop.create_input(spike_train1, j=[int(rng.integers(0, n-1))], wj=input_w, dj=input_d)
pop.create_input(spike_train2, j=[int(rng.integers(0, n-1))], wj=input_w, dj=input_d)
pop.create_input(spike_train3, j=[int(rng.integers(0, n-1))], wj=input_w, dj=input_d)
pop.create_input(spike_train4, j=[int(rng.integers(0, n-1))], wj=input_w, dj=input_d)

pop.run(t, save_post_model=True, show_process=True)
pop.plot_topology()
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()

'''
def findDiff(d1, d2, path=""):
    for k in d1:
        if k in d2:
            if type(d1[k]) is dict:
                findDiff(d1[k],d2[k], "%s -> %s" % (path, k) if path else k)
            if d1[k] != d2[k]:
                result = [ "%s: " % path, " - %s : %s" % (k, d1[k]) , " + %s : %s" % (k, d2[k])]
                print("\n".join(result))
        else:
            print ("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))


def pretty(d, indent=0, s = ""):
   for key, value in d.items():
      s += '\n' + '-' * indent + str(key)
      if isinstance(value, dict):
         s += pretty(value, indent+1, s=s)
      else:
         s += '\n' + '-' * (indent+1) + str(value)
   return s

def check_similarity(l1, l2):
    sim = 0
    tot = 0
    layer1 = l1
    layer2 = l2
    keys_intersect = set.intersection(set(l1.keys()), set(l2.keys()))
    while True:
        if keys_intersect:
            for k in keys_intersect:

            keys1 = layer1.keys()
            keys2 = layer2.keys()

            sim += len(keys_intersect)

        else:
            break
'''




'''

inp1 = [model.neurons[n] for n in model.neurons if isinstance(model.neurons[n], Input)][0]
inp2 = [model.neurons[n] for n in model.neurons if isinstance(model.neurons[n], Input)][1]
#for k in inp.poly_group.keys():
    #print(k)
#findDiff(inp.poly_group[2.0], inp.poly_group[502.0])

print(inp1.poly_group)
print(inp2.poly_group)

k1 = list(inp1.poly_group.keys())[0]
k2 = list(inp2.poly_group.keys())[0]

kk1 = list(inp1.poly_group.keys())[1]
kk2 = list(inp2.poly_group.keys())[1]

tittitotale = {"10": inp1.poly_group[k1], "11": inp2.poly_group[k2]}
tittitotale2 = {"10": inp1.poly_group[kk1], "11": inp2.poly_group[kk2]}
match, unique = Data.compare_poly(tittitotale, tittitotale2)
print(match, unique)
#print(inp1.poly_group[k1])
#print(inp1.poly_group[k2])
#check_similarity(inp.poly_group[2.0], inp.poly_group[502.0])

#print(pretty(inp.poly_group[2.0]))
#print(inp.poly_group[2.0])
#print(model.poly_groups)
#print(model.poly_groups[0])
#print(len(model.poly_groups))
'''

