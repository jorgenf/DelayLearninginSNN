import os
import json
import Population as Pop
import numpy as np
from scipy import stats as st
import itertools
import time

def get_connection_data(dir):
    diverging = 0
    converging = 0
    repeating = 0
    uncategorized = 0
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file == "synapse_data.json":
                with open(os.path.join(subdir, file), "r") as file:
                    data = json.loads(file.read())
                    keys = data.keys()
                    pairs = {}
                    for k in keys:
                        j = int(k.split("-")[1])
                        if j not in pairs.keys():
                            pairs[j] = [k]
                        else:
                            pairs[j].append(k)
                    for pair in pairs:
                        combs = itertools.combinations(pairs[pair],2)
                        for comb in combs:
                            l1 = data[comb[0]]["d_hist"]["d"]
                            l2 = data[comb[1]]["d_hist"]["d"]
                            diff = [x-y for x,y in zip(l1,l2)]
                            if check_divergence(diff, 2000):
                                diverging += 1
                            elif check_convergence(diff, 5000):
                                converging += 1
                            elif check_repetitiveness(diff, 5000):
                                repeating += 1
                            else:
                                uncategorized += 1
    print("Diverging: ", diverging)
    print("Converging: ", converging)
    print("Repeating: ", repeating)
    print("Uncategorized: ", uncategorized)

def check_repetitiveness(l, pattern_length):
    pattern = str(l[-pattern_length:]).strip("[]")
    string = str(l[:-pattern_length]).strip("[]")
    if pattern in string:
        return True
    else:
        return False

def check_convergence(l, stable_state):
    lvl = np.round(l[-1],1)
    trend = np.round(np.mean(l[-stable_state:]),1)
    if trend == lvl:
        return True
    else:
        return False

def check_divergence(l, stable_state):
    if abs(np.round(np.mean(l[-stable_state:]),1)) == 19.9:
        return True
    else:
        return False




def get_SR_data(dir):
    spike_rates = []
    reservoir_spike_rates = []
    input_spike_rates = []
    dormant = 0
    reservoir = 0
    inputs = 0
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file == "neuron_data.json":
                with open(os.path.join(subdir, file), "r") as file:
                    data = json.loads(file.read())
                    for id in data:
                        spikes = len(data[id]["spikes"])
                        if spikes == 0:
                            dormant += 1
                        spike_rate = spikes / (data[id]["duration"] / 1000)
                        spike_rates.append(spike_rate)
                        if data[id]["type"] == str(Pop.Input):
                            input_spike_rates.append(spike_rate)
                            inputs += 1
                        else:
                            reservoir_spike_rates.append(spike_rate)
                            reservoir += 1

    avg_sr = np.round(np.mean(spike_rates), 1)
    median_sr = np.round(np.median(spike_rates), 1)
    mode_sr = np.round(list(st.mode(spike_rates)[0]), 1).tolist()
    std_sr = np.round(np.std(spike_rates), 1)

    avg_rsr = np.round(np.mean(reservoir_spike_rates), 1)
    median_rsr = np.round(np.median(reservoir_spike_rates), 1)
    mode_rsr = np.round(list(st.mode(reservoir_spike_rates)[0]), 1).tolist()
    std_rsr = np.round(np.std(reservoir_spike_rates), 1)

    avg_isr = np.round(np.mean(input_spike_rates), 1)
    median_isr = np.round(np.median(input_spike_rates), 1)
    mode_isr = np.round(st.mode(input_spike_rates)[0], 1).tolist()
    std_isr = np.round(np.std(input_spike_rates), 1)

    dormant_rate = np.round(dormant / reservoir if reservoir != 0 else 0, 2)
    SR_data = {
        "inputs": inputs,
        "reservoir": reservoir,
        "dormant neurons": dormant,
        "dormant rate": dormant_rate,
        "average SR": avg_sr,
        "median SR": median_sr,
        "mode SR" : mode_sr,
        "std SR" : std_sr,
        "average reservoir SR": avg_rsr,
        "median reservoir SR" : median_rsr,
        "mode reservoir SR": mode_rsr,
        "std reservoir SR" : std_rsr,
        "average input SR": avg_isr,
        "median input SR" : median_isr,
        "mode input SR" : mode_isr,
        "std input SR" : std_isr
    }
    with open(f"{dir}/SR_data.json", "w") as f:
        json.dump(SR_data, f)

'''
get_SR_data(
    "C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/4n2i di asynchronous pattern/t2000")
get_SR_data(
    "C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/4n2i di repeating pattern/t2000")
get_SR_data(
    "C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/4n2i di alternating pattern/t2000")

get_SR_data(
    "C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/1n3i di repeating pattern/t2000")
get_SR_data(
    "C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/1n3i di asynchronous pattern/t2000")
get_SR_data(
    "C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/1n3i di alternating pattern/t2000")
'''
start = time.time()
#et_connection_data("C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/1n3i di alternating pattern/t2000")
#get_connection_data("C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/1n3i di asynchronous pattern/t2000")
get_connection_data("C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/4n2i di alternating pattern/t2000")


print(time.time()-start)