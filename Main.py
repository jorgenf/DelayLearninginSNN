import time
import Data
import Population
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


#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n2i/t10000", file_title="delayVSoff1n2i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22"], nd=2)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i/t10000", file_title="delayVSoff1n3i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i w8/t10000", file_title="delayVSoff1n3i_w8_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000", file_title="delayVSfreq1n2i_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2"], nd=2)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000", file_title="delayVSfreq1n3i_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000", file_title="delayVSfreq1n3i_w8_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)

'''
Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000/simulation_data.csv", file_title="delayVSoff1n2i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22"], nd=2)
Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000/simulation_data.csv", file_title="delayVSoff1n3i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i w8/t10000/simulation_data.csv", file_title="delayVSoff1n3i_w8_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000/simulation_data.csv", file_title="delayVSfreq1n2i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2"], nd=2)
Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000/simulation_data.csv", file_title="delayVSfreq1n3i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)
Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000/simulation_data.csv", file_title="delayVSfreq1n3i_w8_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)
'''


#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000", file_title="delayVSfreq1n2i_delayplot", identifier_title="Spike intervals (ms)", identifiers=["f1", "f2"], nd=2)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000", file_title="delayVSfreq1n3i_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000", file_title="delayVSfreq1n3i_w8_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n2i/t10000", file_title="delayVSoff1n2i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22"], nd=2)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i/t10000", file_title="delayVSoff1n3i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_delay_catetgories_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i w8/t10000", file_title="delayVSoff1n3i_w8_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
'''

Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000", file_title="delayVSoff1n2i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22"], nd=2)
Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000", file_title="delayVSoff1n3i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i w8/t10000", file_title="delayVSoff1n3i_w8_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000", file_title="delayVSfreq1n2i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2"], nd=2)
Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000", file_title="delayVSfreq1n3i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)
Data.plot_SR_heatmap("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i w8/t10000", file_title="delayVSfreq1n3i_w8_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)
'''
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
t = 1000
k = 2
n = 10
#internal_d = 20
internal_d = [29,60,29,60,29,60,29,60,29,60,29,60,29,60,29,60,29,60,29,60,29,60]
internal_w = 16
input_d = [29, 60]
input_w = [32, 16]
input_connections = [0, 1]
input_freq = [0, 600]

pop = Population.Population((n, Population.RS), path="./network_plots", name=f"directional_ring_lattice_{n}n1i_k{k}_d{internal_d}_w{internal_w}_inpd{input_d}_inpw{input_w}_freq{input_freq}_conn{input_connections}")
pop.create_directional_ring_lattice_connections(k=k, d=internal_d, w=internal_w, trainable=True)

#spike_train = [x for x in range(0, t) if x % input_freq == 0]
pop.create_input(input_freq, j=input_connections, wj=input_w, dj=input_d)
pop.run(t, save_post_model=True)
pop.plot_topology()
pop.plot_delays()
pop.plot_raster()
pop.plot_membrane_potential()
'''
pre_model = Data.load_model("network_plots/directional_ring_lattice_10n1i_k2_d[29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60]_w16_inpd[29, 60]_inpw[30, 16]_freq[0, 600]_conn[0, 1]_1/pre_sim_model.pkl")
post_model = Data.load_model("network_plots/directional_ring_lattice_10n1i_k2_d[29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60, 29, 60]_w16_inpd[29, 60]_inpw[30, 16]_freq[0, 600]_conn[0, 1]_1/post_sim_model.pkl")

print("Spikes n8: ", post_model.neurons["8"].spikes)
print("Delay 8-0 at t=277.2: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 8 and syn.j == 0][0][2772])

print("Spikes n9: ", post_model.neurons["9"].spikes)
print("Delay 9-0 at t=308.0: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 9 and syn.j == 0][0][3080])
print("Delay 9-1 at t=307.0: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 9 and syn.j == 1][0][3070])

print("Spikes n0: ", post_model.neurons["0"].spikes)
print("Delay 0-1 at t=338.8: ", [syn.d_hist["d"] for syn in post_model.synapses if syn.i == 0 and syn.j == 1][0][3388])


for v,u,t in zip(post_model.neurons["0"].v_hist["v"][3270:3390],post_model.neurons["0"].u_hist["u"][3270:3390], post_model.neurons["0"].u_hist["t"][3270:3390]):
    print(v,u, t)

print("-------------------")

for v,u,t in zip(post_model.neurons["1"].v_hist["v"][3600:3800],post_model.neurons["1"].u_hist["u"][3600:3800], post_model.neurons["1"].u_hist["t"][3600:3800]):
    print(v,u, t)



'''
t = sorted([round(np.random.random()*100,1) for x in range(1000)])
print(t)

#t = [2.3, 4.5, 4.7, 4.9, 5.8, 6, 6.8]

h = np.histogram(t, bins=10, range=[0,100])
print(h)
'''