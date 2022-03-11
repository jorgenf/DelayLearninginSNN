import itertools
import os
import json
import matplotlib.colors
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
import matplotlib.patches as mpatches
import shutil
from sklearn.preprocessing import StandardScaler


def compile_simulation_data(dir):
    existing_sims = []
    sim_data = os.path.join(dir, "simulation_data.csv")
    if os.path.exists(sim_data):
        print(f"\rSimulation data exists. Adding to existing data...", end="\t")
        df = pd.read_csv(sim_data)
        if df[df["name"] == "count"].first_valid_index():
            print(f"\rPrevious summation data found. Removing summation data from: {os.path.split(dir)[1]}",  end="\t")
            first_id = df[df["name"] == "count"].first_valid_index()
            df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
        df.dropna(inplace=True)
        existing_sims = list(df["name"])
        df.to_csv(sim_data, index=False)
    else:
        print(f"\rCreating new simulation data file...", end="\t")
    print(f"\rCompiling simulation data for {dir}", end="")
    compiled_data = []
    for dirs, subdirs, files in os.walk(dir):
        #SR_dir = os.path.join(dirs, "SR_data.json")
        # if t_folder == os.path.split(dirs)[1] and not os.path.exists(SR_dir):
        #    get_SR_data(dirs)
        if os.path.split(dirs)[1] not in existing_sims and os.path.split(os.path.split(dirs)[0])[1].startswith("t"):
            data_dict = {}
            data_dict["name"] = os.path.basename(os.path.normpath(dirs))
            neuron_fp = os.path.join(dirs, "neuron_data.json")
            if os.path.exists(neuron_fp):
                with open(neuron_fp, "r") as file:
                    data = json.load(file)
                    for id in data:
                        spikes = len(data[id]["spikes"])
                        spike_rate = spikes / (data[id]["duration"] / 1000)
                        type = "i" if data[id]["type"] == "<class 'Population.Input'>" else "n"
                        data_dict[f"{type}{id}_SR"] = spike_rate
            synapse_fp = os.path.join(dirs, "synapse_data.json")
            if os.path.exists(synapse_fp):
                with open(synapse_fp, "r") as file:
                    data = json.load(file)
                    keys = data.keys()
                    for k in keys:
                        try:
                            l = data[k]["d_hist"]["d"]
                        except:
                            l = data[k]["d_hist"]
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
                            try:
                                divergence = check_divergence(data[k]["d_hist"]["d"])
                            except:
                                divergence = check_divergence(data[k]["d_hist"])
                            if divergence:
                                data_dict[k] = divergence
                        if k not in data_dict.keys():
                            data_dict[k] = "uncategorized"
            compiled_data.append(data_dict)
    save_sim_data(dir, compiled_data)
    print("Finished compiling data for: ", dir)


def save_sim_data(dir, data):
    path = os.path.join(dir, "simulation_data.csv")
    exists = os.path.isfile(path)
    with open(path, 'a' if exists else 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        if not exists:
            writer.writeheader()
        writer.writerows(data)


def check_repetitiveness(l, pattern_length, use_correlation=False):
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
    slope = linregress(np.round(np.arange(0, np.round(len(l)/10, 1), 0.1), 1), l)[0]
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


def sum_simulation_data(dir):
    print(f"\rSumming simulation data...", end="")
    df = pd.read_csv(os.path.join(dir, "simulation_data.csv"))
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
    dormant = {"name": "dormant"}
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
    df.to_csv(os.path.join(dir, "simulation_data.csv"), index=False)
    data_list = [count, mean, std, min, pros25, pros50, pros75, max, dormant]
    with open(os.path.join(dir, "simulation_data.csv"), 'a', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cols)
        for d in data_list:
            writer.writerow(d)
    print("Finished summing data for: ", dir)

def delete_simulation_data(dir):
    if os.path.exists(os.path.join(dir, "simulation_data.csv")):
        os.remove(os.path.join(dir, "simulation_data.csv"))
        print("DELETED simulation data for: ", os.path.split(os.path.split(dir)[0])[1])
    else:
        print("No simulation data found for: ", os.path.split(os.path.split(dir)[0])[1])


def save_model(object, dir):
    with open(dir, "wb") as file:
        pickle.dump(object, file, pickle.HIGHEST_PROTOCOL)


def load_model(dir):
    with open(os.path.join(dir), 'rb') as file:
        return (pickle.load(file))


def plot_delay_categories(dir, file_title, identifier_title, identifiers, nd):
    cat_list = C.DELAY_CATEGORIES_SHORTLIST
    possible_colors = C.DCAT_COLORS[:len(cat_list)]
    for dirs, subdirs, files in os.walk(dir):
        if os.path.exists(os.path.join(dirs, "simulation_data.csv")):
            df = pd.read_csv(os.path.join(dirs, "simulation_data.csv"))
            index = list(df.index[df["name"] == "count"])[0]
            df_rows = df.loc[:index - 1]
            data = [[] for x in range(df_rows.shape[0])]
            for col in df_rows.keys():
                if col == "name":
                    names = list(df_rows[col])
                    for i, name in enumerate(names):
                        f = []
                        d = []
                        for fx in identifiers:
                            f.append(int(re.findall(f'{fx}-([0-9]+)', name)[0]))
                        for dx in range(nd):
                            d.append(int(re.findall(f'd{dx + 1}-([0-9]+)', name)[0]))
                        config = (f + d)
                        [data[i].append(x) for x in config]
                elif re.match("[0-9]+_[0-9]+$", col):
                    conn = list(df_rows[col])
                    for i, c in enumerate(conn):
                        data[i].append(c)


            data = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                      range(len(identifiers))], 2)]),
                                                  sum(element[x] for x in
                                                      range(len(identifiers), len(identifiers) + nd)),
                                                  element[0], element[len(identifiers)]))
            '''

            data = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                                       range(len(identifiers))], 2)]),
            np.mean([abs(x[0] - x[1]) for x in
                     itertools.combinations([element[x] for x in
                                             range(len(identifiers), len(identifiers) + nd)], 2)]),
            element[0], element[len(identifiers)]))
            '''

            x = []
            y = []
            z = []
            for row in data:
                x_str = ""
                y_str = ""
                cat = []
                for yid in range(len(identifiers)):
                    if yid < len(identifiers) - 1:
                        y_str += f"{row[yid]}-"
                    else:
                        y_str += f"{row[yid]}"
                for xid in range(len(identifiers), nd + len(identifiers)):
                    if xid < nd + len(identifiers) - 1:
                        x_str += f"{row[xid]}-"
                    else:
                        x_str += f"{row[xid]}"
                x.append(x_str)
                y.append(y_str)
                for catid in range(nd + len(identifiers), len(row)):
                    cat.append(row[catid])
                if any(i in cat for i in ["increasing", "decreasing", "min", "max"]):
                    z.append("diverging")
                elif "uncategorized" in cat:
                    z.append("uncategorized")
                elif "repeating" in cat:
                    z.append("repeating")
                elif "converging" in cat:
                    z.append("converging")
                else:
                    raise Exception(f"Category {cat} not found!")
            fig, ax = plt.subplots()
            colors = []
            for p in z:
                if p in cat_list:
                    index = cat_list.index(p)
                else:
                    raise Exception(f"Category not found: {p}")
                color = possible_colors[index]
                colors.append(color)
            ax.scatter(y, x, s=0.1, c=colors)
            ax.set_xlabel(identifier_title)
            ax.set_ylabel("Delays (ms)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(min(50, len(set(y)))))
            ax.yaxis.set_major_locator(plt.MaxNLocator(min(30, len(set(x)))))
            plt.xticks(rotation=90)
            path = os.path.join(os.getcwd(), f"{file_title}.png")
            patches = [mpatches.Patch(color=col, label=cat) for col, cat in zip(possible_colors, cat_list)]
            plt.legend(handles=patches, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
            print("Saving data to: ", path)
            plt.tight_layout()
            plt.savefig(path)
            plt.clf()


def plot_spike_rate_data(path, file_title, identifier_title, identifiers, nd):
    fig, ax = plt.subplots()
    df = pd.read_csv(path)
    index = list(df.index[df["name"] == "count"])[0]
    df_rows= df.loc[:index - 1]
    #data = df.loc[df["name"].str.startswith(identifiers)]
    names = list(df_rows["name"])
    data = [[] for x in range(df_rows.shape[0])]
    SR_keys = [k for k in list(df.keys()) if k.endswith("SR") and k.startswith("n")]
    for i, name in enumerate(names):
        f = []
        d = []
        for fx in identifiers:
            f.append(int(re.findall(f'{fx}-([0-9]+)', name)[0]))
        for dx in range(nd):
            d.append(int(re.findall(f'd{dx + 1}-([0-9]+)', name)[0]))
        sr = sum([x for x in [df_rows.iloc[i][srk] for srk in SR_keys]])
        config = (f + d)
        config.append(sr)
        [data[i].append(x) for x in config]

    data_sort = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                      range(len(identifiers))], 2)]),
                                                  sum(element[x] for x in
                                                      range(len(identifiers), len(identifiers) + nd)),
                                                  element[0], element[len(identifiers)]))
    '''
    data_sort = sorted(data, key=lambda element: (np.mean([abs(x[0] - x[1]) for x in itertools.combinations([element[x] for x in
                                                      range(len(identifiers))], 2)]),
                                                  np.mean([abs(x[0] - x[1]) for x in
                                                           itertools.combinations([element[x] for x in
                                                                                   range(len(identifiers), len(identifiers) + nd)], 2)]),
                                                  element[0], element[len(identifiers)]))
    '''
    x = []
    y = []
    z = []
    for row in data_sort:
        x_str = ""
        y_str = ""
        for yid in range(len(identifiers)):
            if yid < len(identifiers) - 1:
                y_str += f"{row[yid]}-"
            else:
                y_str += f"{row[yid]}"
        for xid in range(len(identifiers), nd + len(identifiers)):
            if xid < nd + len(identifiers) - 1:
                x_str += f"{row[xid]}-"
            else:
                x_str += f"{row[xid]}"
        x.append(x_str)
        y.append(y_str)
        z.append(row[-1])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(z))
    ax.scatter(y, x, s=0.1, c=z, cmap=plt.cm.get_cmap("Reds"))
    ax.set_ylabel("Delays (ms)")
    ax.set_xlabel(identifier_title)
    plt.xticks(rotation=90)
    ax.xaxis.set_major_locator(plt.MaxNLocator(min(50, len(set(y)))))
    ax.yaxis.set_major_locator(plt.MaxNLocator(min(30, len(set(x)))))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap("Reds")), ax=ax)
    plt.tight_layout()
    path = os.path.join(os.getcwd(), f"{file_title}.png")
    print("Saving data to: ", path)
    plt.savefig(path)

def plot_categories_PCA(path):
    df = pd.read_csv(path)
    first_id = df[df["name"] == "count"].first_valid_index()
    df.drop(df.tail(df.shape[0] - first_id).index, inplace=True)
    df.dropna(inplace=True)
    names = list(df["name"])
    data_unsort = [[] for x in range(df.shape[0])]
    labels = df.keys()
    connections = [x for x in labels if re.search("^\d+_\d+$",str(x))]
    z = [df[c] for c in connections]

    for i, name in enumerate(names):
        f1 = re.findall(r'f1-([0-9]+)', name)[0]
        f2 = re.findall(r'f2-([0-9]+)', name)[0]
        f3 = re.findall(r'f3-([0-9]+)', name)[0]
        d1 = re.findall(r'd1-([0-9]+)', name)[0]
        d2 = re.findall(r'd2-([0-9]+)', name)[0]
        d3 = re.findall(r'd3-([0-9]+)', name)[0]
        config = [int(d1), int(d2), int(d3), int(f1), int(f2), int(f3)]
        cat = [c[i] for c in z]
        if any(i in cat for i in ["increasing", "decreasing", "min", "max"]):
            config.append("diverging")
        elif "uncategorized" in cat:
            config.append("uncategorized")
        elif "repeating" in cat:
            config.append("repeating")
        elif "converging" in cat:
            config.append("converging")
        else:
            raise Exception(f"Category {cat} not found!")
        data_unsort[i] = config

    data_sort = sorted(data_unsort, key=lambda element: (element[0], element[1], element[2], element[3]))
    for d in data_sort:
        print(d)

def reduce_sim_file_size(dir):
    for dirs, subdirs, files in os.walk(dir):
        if "synapse_data.json" in files:
            file = os.path.join(dirs, "synapse_data.json")
            new_data = {}
            with open(file, "r") as json_file:
                try:
                    data = json.load(json_file)
                except:
                    print("Unable to load:", file)
                if any([k for k in data.keys() if type(data[k]["d_hist"]) == dict]):
                    for k in data.keys():
                        if type(data[k]["d_hist"]) == dict:
                            data[k]["d_hist"] = data[k]["d_hist"]["d"]
                    new_data = data
            if new_data:
                os.remove(file)
                with open(file, "w") as new_file:
                    json.dump(data, new_file)
                    print("Made changes to: ", file)

def change_name(dir, string_to_remove):
    # MÅ FIKSES!!!
    main_dir = os.path.split(dir)[0]
    if re.search(string_to_remove,dir):
        sd = dir.replace(string_to_remove, "")
        dd = os.path.join(main_dir, sd)
        print("before: ", dir)
        try:
            os.rename(os.path.join(main_dir,dir),dd)
            print("after: ", dd)
        except:
            print(dir)