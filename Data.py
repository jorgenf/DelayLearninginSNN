import os
import json

def get_average_connectivity(dir):
    n = 0
    out = 0
    in_ = 0
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file == "":
                print(os.path.join(subdir, file))

def get_average_spike_rate(dir):
    spike_rates = 0
    dormant = 0
    spike_rates_per_ID = []
    n = 0
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file == "neuron_data.json":
                with open(os.path.join(subdir, file), "r") as file:
                    data = json.loads(file.read())
                    for id in data:
                        spikes = len(data[id]["spikes"])
                        if spikes == 0:
                            dormant += 1
                        spike_rate = spikes/(data[id]["duration"]/1000)
                        spike_rates += spike_rate
                        n += 1
    return spike_rates/n, dormant




res = get_average_spike_rate("C:/Users/jorge/OneDrive - OsloMet/Master thesis - Jørgen Farner/Simulation results/4n2i di asynchronous pattern/t2000")
print(res)