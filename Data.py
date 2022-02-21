import itertools
import os
import json
import Population as Pop
import numpy as np
from scipy import stats as st
import pickle
import pandas as pd
import csv
from collections import Counter
from scipy.stats import linregress
import matplotlib.pyplot as plt
import re
import Constants as C


def compile_simulation_data(dir, t_folder):
    existing_sims = []
    sim_data = os.path.join(dir, t_folder, "simulation_data.csv")
    if os.path.exists(sim_data):
        print("Simulation data exists. Adding to existing data...")
        df = pd.read_csv(sim_data)
        if df[df["name"] == "count"].first_valid_index():
            print("Previous summation data found. Removing summation data from: ", os.path.split(dir)[1])
            first_id = df[df["name"] == "count"].first_valid_index()
            df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
        df.dropna(inplace=True)
        existing_sims = list(df["name"])
        df.to_csv(os.path.join(dir, "simulation_data.csv"), index=False)
    else:
        print("Creating new simulation data file...")
    print(f"\rCompiling simulation data...", end="")
    for dirs, subdirs, files in os.walk(dir):
        SR_dir = os.path.join(dirs, "SR_data.json")
        #if t_folder == os.path.split(dirs)[1] and not os.path.exists(SR_dir):
        #    get_SR_data(dirs)
        if t_folder == os.path.split(os.path.split(dirs)[0])[1]:
            if os.path.split(dirs)[1] not in existing_sims:
                data_dict = {}
                data_dict["name"] = os.path.basename(os.path.normpath(dirs))
                neuron_fp = os.path.join(dirs, "neuron_data.json")
                if os.path.exists(neuron_fp):
                    try:
                        with open(neuron_fp, "r") as file:
                            data = json.loads(file.read())
                            for id in data:
                                spikes = len(data[id]["spikes"])
                                spike_rate = spikes / (data[id]["duration"] / 1000)
                                type = "i" if data[id]["type"] == "<class 'Population.Input'>" else "n"
                                data_dict[f"{type}{id}_SR"] = spike_rate
                    except:
                        print(f"Unable to open: ", {os.path.join(dirs, "neuron_data.json")})
                synapse_fp = os.path.join(dirs, "synapse_data.json")
                if os.path.exists(synapse_fp):
                    try:
                        with open(synapse_fp, "r") as file:
                            data = json.loads(file.read())
                            keys = data.keys()
                            for k in keys:
                                l = data[k]["d_hist"]["d"]
                                saturation = check_saturation(l, C.SATURATION_LENGTH)
                                if saturation:
                                    data_dict[k] = saturation
                                else:
                                    if check_convergence(l, C.CONVERGENT_LENGTH):
                                        if k in data_dict.keys():
                                            data_dict[k] += "-converging"
                                        else:
                                            data_dict[k] = "converging"
                                    elif check_repetitiveness(l, C.REPETITIVE_LENGTH):
                                        if k in data_dict.keys():
                                            data_dict[k] += "-repeating"
                                        else:
                                            data_dict[k] = "repeating"
                                    divergence = check_divergence(data[k]["d_hist"])
                                    if divergence:
                                        data_dict[k] = divergence
                                if k not in data_dict.keys():
                                    data_dict[k] = "uncategorized"
                    except:
                        print(f"Unable to open: ", synapse_fp)
                path = os.path.join(os.path.split(dirs)[0], "simulation_data.csv")
                exists = os.path.isfile(path)
                with open(path, 'a' if exists else 'w', newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
                    if not exists:
                        writer.writeheader()
                    writer.writerow(data_dict)

def check_repetitiveness(l, pattern_length, use_correlation = False):
    if use_correlation:
        pattern = l[-pattern_length:]
        string = l[-pattern_length * 4: -pattern_length]
        max_corr = 0
        for i in range(len(string) - len(pattern)):
            corr = np.corrcoef(string[i:i + len(pattern)], pattern)[0][1]
            max_corr = max(max_corr, corr)
        if max_corr > C.CORRELATION_THRESHOLD:
            return True
        else:
            return False
    else:
        pattern = str(l[-pattern_length:]).strip("[]")
        string = str(l[: -pattern_length]).strip("[]")
        if pattern in string:
            return True
        else:
            return False

def check_convergence(l, pattern_length):
    if np.std(l[-pattern_length:]) < C.STD_THRESHOLD:
        return True
    else:
        return False


def check_divergence(l):
    slope = linregress(l["t"], l["d"])[0]
    if slope > C.SLOPE_THRESHOLD:
        return "increasing"
    elif slope < -C.SLOPE_THRESHOLD:
        return "decreasing"
    else:
        return False


def check_saturation(l, pattern_length):
    mean_val = np.mean(l[-pattern_length:])
    if C.MIN_DELAY - C.STD_THRESHOLD < mean_val < C.MIN_DELAY + C.STD_THRESHOLD:
        return "min"
    elif C.MAX_DELAY - C.STD_THRESHOLD < mean_val < C.MAX_DELAY + C.STD_THRESHOLD:
        return "max"
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
                print(f"\rCompiling SR data for: {os.path.split(dirs)[1]}", end="")
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
    print(f"\rSumming simulation data...", end="")
    for dirs, subdirs, files in os.walk(dir):
        if t_folder == os.path.split(dirs)[1]:
            df = pd.read_csv(os.path.join(dirs, "simulation_data.csv"))
            if df[df["name"] == "count"].first_valid_index():
                print(f"\rDeleting previous summation data...", end="")
                first_id = df[df["name"] == "count"].first_valid_index()
                df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
                print(f"\rSumming simulation data...", end="")
            df.dropna(inplace=True)
            count = {"name": "count"}
            mean = {"name": "mean"}
            std = {"name": "std"}
            min = {"name": "min"}
            pros25 = {"name": "25%"}
            pros50 = {"name": "50%"}
            pros75 = {"name": "75%"}
            max = {"name": "max"}
            dormant = {"name" : "dormant"}
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
                        dormant[col] = (df[col] == 0).sum()
            df.to_csv(os.path.join(dirs, "simulation_data.csv"), index=False)
            data_list = [count, mean, std, min, pros25, pros50, pros75, max, dormant]
            with open(os.path.join(dirs, "simulation_data.csv"), 'a', newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=cols)
                for d in data_list:
                    writer.writerow(d)


def delete_simulation_data(dir, t_folder):
    for dirs, subdirs, files in os.walk(dir):
        if t_folder == os.path.split(dirs)[1]:
            if os.path.exists(os.path.join(dirs,"simulation_data.csv")):
                os.remove(os.path.join(dirs,"simulation_data.csv"))
                print("DELETED simulation data for: ", os.path.split(os.path.split(dirs)[0])[1])
            else:
                print("No simulation data found for: ", os.path.split(os.path.split(dirs)[0])[1])

def save_model(object, dir):
    with open(dir, "wb") as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)

def load_model(dir):
    with open(os.path.join(dir), 'rb') as file:
        return(pickle.load(file))

def plot_delay_categories(dir, t_folder, topology):
    cat_list = list(itertools.combinations_with_replacement(C.DELAY_CATEGORIES_SHORTLIST, 2))
    cat_combos = [sorted(x) for x in cat_list]
    print(cat_combos)
    possible_colors = C.COLORS[:len(cat_combos)]
    for dirs, subdirs, files in os.walk(dir):
        top_temp = os.path.split(os.path.split(dirs)[0])[1].split(" ")[0]
        if t_folder == os.path.split(dirs)[1] and topology == top_temp:
            if os.path.exists(os.path.join(dirs, "simulation_data.csv")):
                df = pd.read_csv(os.path.join(dirs, "simulation_data.csv"))
                index =  list(df.index[df["name"]=="count"])[0]
                df_rows = df.loc[:index-1]
                data = [[] for x in range(df_rows.shape[0])]
                for col in df_rows.keys():
                    if col == "name":
                        names = list(df_rows[col])
                        for i, name in enumerate(names):
                            f1 = re.findall(r'f1-([0-9]+)', name)[0]
                            f2 = re.findall(r'f2-([0-9]+)', name)[0]
                            d1 = re.findall(r'd1-([0-9]+)', name)[0]
                            d2 = re.findall(r'd2-([0-9]+)', name)[0]
                            config = (int(f1), int(f2), int(d1), int(d2))
                            [data[i].append(x) for x in config]
                    elif re.match("[0-9]+_[0-9]+$", col):
                        conn = list(df_rows[col])
                        for i, c in enumerate(conn):
                            data[i].append(c)
                data = sorted(data, key=lambda element: (element[0], element[1], element[2], element[3]))
                x = []
                y = []
                z = []
                for row in data:
                    x.append(f"{row[0]}-{row[1]}")
                    y.append(f"{row[2]}-{row[3]}")
                    z.append(sorted([C.CATEGORY_CONVERSION[row[4]], C.CATEGORY_CONVERSION[row[5]]]))
                fig, ax = plt.subplots()
                colors = []
                for p in z:
                    if [p[0],p[1]] in cat_combos:
                        index = cat_combos.index([p[0],p[1]])
                    elif [p[1],p[0]] in cat_combos:
                        index = cat_combos.index([p[1], p[0]])
                    else:
                        raise Exception(f"Category combination not found: [{p[0]}, {p[1]}]")
                    color = possible_colors[index]
                    colors.append(color)
                ax.scatter(y, x, s=0.1, c=colors)
                ax.set_ylabel("Frequencies")
                ax.set_xlabel("Delays")
                plt.xticks(rotation=90)
                ax.xaxis.set_major_locator(plt.MaxNLocator(50))
                #plt.locator_params(axis='x', nbins=10)
                path = os.path.join(os.getcwd(), "delayVSfreq.png")
                plt.legend()
                print("Saving data to: ", path)
                plt.tight_layout()
                plt.savefig(path)
                plt.clf()


def plot_spike_rate_data(dir, t_folder, topology):
    fig, ax = plt.subplots()
    for dirs, subdirs, files in os.walk(dir):
        top_temp = os.path.split(os.path.split(dirs)[0])[1].split(" ")[0]
        if t_folder == os.path.split(dirs)[1] and topology == top_temp:
            df = pd.read_csv(os.path.join(dirs, "simulation_data.csv"))
            data = df.loc[df["name"].str.startswith(topology)]
            for k in df.keys():
                if str(k).endswith("SR"):
                    pass
            return False






