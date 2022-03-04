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

dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000"


'''
for dirs, subdirs, file in os.walk(dir):
    if re.search("t10000.+",os.path.split(dirs)[1]):
        print(os.path.split(dirs)[1].replace("t10000", ""))
        d = os.path.split(dirs)[0]
        sd = os.path.split(dirs)[1].replace("t10000", "")
        dd = os.path.join(d, sd)
        print(dd)
        os.rename(dirs,dd)

'''
'''
def change_name(dir):
    main_dir = "Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000"

    if re.search("t10000.+",dir):
        #d = os.path.split(dir)[0]
        #sd = os.path.split(dir)[1].replace("t10000", "")
        sd = dir.replace("t10000", "")
        dd = os.path.join(main_dir, sd)
        print("before: ", dir)
        try:
            os.rename(os.path.join(main_dir,dir),dd)
            print("after: ", dd)
        except:
            print(dir)

#params = [change_name(x[0]) for x in os.walk(dir)]
params = sorted(os.listdir(dir), reverse=True)

[change_name(x) for x in params]

'''
'''
#params = os.listdir(dir)

#if __name__ == "__main__":
#    with mp.Pool(os.cpu_count()-1) as pool:
#        pool.map(change_name, params)

'''

'''
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
    for dirs, subdirs, files in os.walk(dir):
        if os.path.split(dirs)[1].startswith("delay"):
            param = os.path.split(dirs)[1]
            off11 = int(re.findall(r'off11-([0-9]+)', param)[0])
            off12 = int(re.findall(r'off12-([0-9]+)', param)[0])
            off21 = int(re.findall(r'off21-([0-9]+)', param)[0])
            off22 = int(re.findall(r'off22-([0-9]+)', param)[0])
            off31 = int(re.findall(r'off31-([0-9]+)', param)[0])
            off32 = int(re.findall(r'off32-([0-9]+)', param)[0])
            d1 = int(re.findall(r'd1-([0-9]+)', param)[0])
            d2 = int(re.findall(r'd2-([0-9]+)', param)[0])
            d3 = int(re.findall(r'd2-([0-9]+)', param)[0])
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
    for dirs, subdirs, files in os.walk(dir):
        if os.path.split(dirs)[1].startswith("delay"):
            param = os.path.split(dirs)[1]
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
    for dirs, subdirs, files in os.walk(dir):
        if os.path.split(dirs)[1].startswith("delay"):
            param = os.path.split(dirs)[1]

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
    for dirs, subdirs, files in os.walk(dir):
        param = os.path.split(dirs)[1]
        if param.startswith("delayVSfreq"):
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


#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n3i/t10000")
#Data.compile_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSfrequency/1n2i/t10000")

#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n2i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000")
#Data.sum_simulation_data(dir="Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000")

Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n2i/t10000", file_title="delayVSoff1n2i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22"], nd=2)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSoff/1n3i/t10000", file_title="delayVSoff1n3i_delayplot",identifier_title="Spike offsets (ms)", identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000", file_title="delayVSfreq1n2i_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2"], nd=2)
#Data.plot_delay_categories("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000", file_title="delayVSfreq1n3i_delayplot",identifier_title="Spike intervals (ms)", identifiers=["f1", "f2", "f3"], nd=3)


#Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n2i/t10000/simulation_data.csv", file_title="delayVSoff1n2i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22"], nd=2)
#Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward\delayVSoff/1n3i/t10000/simulation_data.csv", file_title="delayVSoff1n3i_spikeplot", identifier_title="Spike offsets (ms)",identifiers=["off11", "off12", "off21", "off22", "off31", "off32"], nd=3)
#Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n2i/t10000/simulation_data.csv", file_title="delayVSfreq1n2i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2"], nd=2)
#Data.plot_spike_rate_data("Z:/MASTER THESIS - SIMULATION RESULTS/feed forward/delayVSfrequency/1n3i/t10000/simulation_data.csv", file_title="delayVSfreq1n3i_spikeplot", identifier_title="Spike intervals (ms)",identifiers=["f1", "f2", "f3"], nd=3)

