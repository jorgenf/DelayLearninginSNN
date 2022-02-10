import os
import json
import Population as Pop
import numpy as np
from scipy import stats as st
import itertools
import pandas as pd
import csv
from collections import Counter


def compile_simulation_data(dir, t_folder):
    for dirs, subdirs, files in os.walk(dir):
        SR_dir = os.path.join(dirs, "SR_data.json")
        if t_folder == os.path.split(dirs)[1] and not os.path.exists(SR_dir):
            get_SR_data(dirs)
        if t_folder == os.path.split(os.path.split(dirs)[0])[1]:
            data_dict = {}
            data_dict["name"] = os.path.basename(os.path.normpath(dirs))
            for file in files:
                if file == "neuron_data.json":
                    with open(os.path.join(dirs, file), "r") as file:
                        data = json.loads(file.read())
                        for id in data:
                            spikes = len(data[id]["spikes"])
                            spike_rate = spikes / (data[id]["duration"] / 1000)
                            data_dict[f"n{id}_SR"] = spike_rate
                if file == "synapse_data.json":
                    with open(os.path.join(dirs, file), "r") as file:
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
                            combs = itertools.combinations(pairs[pair], 2)
                            for comb in combs:
                                l1 = data[comb[0]]["d_hist"]["d"]
                                l2 = data[comb[1]]["d_hist"]["d"]
                                diff = [x - y for x, y in zip(l1, l2)]
                                if check_divergence(diff, 2000):
                                    data_dict[f"{comb[0]}_{comb[1]}"] = "diverging"
                                elif check_convergence(diff, 5000):
                                    data_dict[f"{comb[0]}_{comb[1]}"] = "converging"
                                elif check_repetitiveness(diff, 5000):
                                    data_dict[f"{comb[0]}_{comb[1]}"] = "repeating"
                                else:
                                    data_dict[f"{comb[0]}_{comb[1]}"] = "uncategorized"
            path = os.path.join(os.path.split(dirs)[0], "simulation_data.csv")
            exists = os.path.isfile(path)
            with open(path, 'a' if exists else 'w', newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
                if not exists:
                    writer.writeheader()
                writer.writerow(data_dict)
    sum_simulation_data(dir, t_folder)


def check_repetitiveness(l, pattern_length):
    pattern = str(l[-pattern_length:]).strip("[]")
    string = str(l[:-pattern_length]).strip("[]")
    if pattern in string:
        return True
    else:
        return False


def check_convergence(l, stable_state):
    lvl = np.round(l[-1], 1)
    trend = np.round(np.mean(l[-stable_state:]), 1)
    if trend == lvl:
        return True
    else:
        return False


def check_divergence(l, stable_state):
    if abs(np.round(np.mean(l[-stable_state:]), 1)) == 19.9:
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

    avg_sr = float(np.round(np.mean(spike_rates), 1))
    median_sr = float(np.round(np.median(spike_rates), 1))
    mode_sr = np.round(list(st.mode(spike_rates)[0]), 1).tolist()
    std_sr = float(np.round(np.std(spike_rates), 1))

    avg_rsr = float(np.round(np.mean(reservoir_spike_rates), 1))
    median_rsr = float(np.round(np.median(reservoir_spike_rates), 1))
    mode_rsr = np.round(list(st.mode(reservoir_spike_rates)[0]), 1).tolist()
    std_rsr = float(np.round(np.std(reservoir_spike_rates), 1))

    avg_isr = float(np.round(np.mean(input_spike_rates), 1))
    median_isr = float(np.round(np.median(input_spike_rates), 1))
    mode_isr = np.round(st.mode(input_spike_rates)[0], 1).tolist()
    std_isr = float(np.round(np.std(input_spike_rates), 1))

    dormant_rate = float(np.round(dormant / reservoir if reservoir != 0 else 0, 2))
    SR_data = {
        "inputs": inputs,
        "reservoir": reservoir,
        "dormant neurons": dormant,
        "dormant rate": dormant_rate,
        "average SR": avg_sr,
        "median SR": median_sr,
        "mode SR": mode_sr,
        "std SR": std_sr,
        "average reservoir SR": avg_rsr,
        "median reservoir SR": median_rsr,
        "mode reservoir SR": mode_rsr,
        "std reservoir SR": std_rsr,
        "average input SR": avg_isr,
        "median input SR": median_isr,
        "mode input SR": mode_isr,
        "std input SR": std_isr
    }
    with open(f"{dir}/SR_data.json", "w") as f:
        json.dump(SR_data, f)


def sum_simulation_data(dir, t_folder):
    for dirs, subdirs, files in os.walk(dir):
        if t_folder == os.path.split(dirs)[1]:
            df = pd.read_csv(os.path.join(dirs, "simulation_data.csv"))
            if df[df["name"] == "count"].first_valid_index():
                first_id = df[df["name"] == "count"].first_valid_index()
                df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
            count = {"name": "count"}
            mean = {"name": "mean"}
            std = {"name": "std"}
            min = {"name": "min"}
            pros25 = {"name": "25%"}
            pros50 = {"name": "50%"}
            pros75 = {"name": "75%"}
            max = {"name": "max"}

            cols = list(df.columns)
            for col in cols:
                if not col == "name":
                    if df[col].dtypes == object:
                        counts = Counter(list(df[col]))
                        count[col] = dict(counts)
                        mean[col] = counts.most_common()[0]
                    else:
                        stats = round(df[col].describe(), 1)
                        count[col] = stats["count"]
                        mean[col] = stats["mean"]
                        std[col] = stats["std"]
                        min[col] = stats["min"]
                        pros25[col] = stats["25%"]
                        pros50[col] = stats["50%"]
                        pros75[col] = stats["75%"]
                        max[col] = stats["max"]
            df.to_csv(os.path.join(dirs, "simulation_data.csv"), index=False)
            data_list = [count, mean, std, min, pros25, pros50, pros75, max]
            with open(os.path.join(dirs, "simulation_data.csv"), 'a', newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=cols)
                for d in data_list:
                    writer.writerow(d)
            break


# compile_simulation_data("C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/feed forward", t_folder="t5000")
sum_simulation_data("C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/feed forward",
                    t_folder="t5000")
