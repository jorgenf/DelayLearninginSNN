import os
from datetime import datetime
import numpy as np
from collections import deque, Counter
import time
import math as m
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import itertools
import json
import Constants
import Data
import matplotlib.colors
sns.set()


COLORS = Constants.COLORS
cm = 1/2.54
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=14)
plt.rcParams['figure.figsize'] = (15*cm, 15*cm)
MAX_DELAY = Constants.MAX_DELAY
MIN_DELAY = Constants.MIN_DELAY
mpl.use("Agg")




class Population:

    def __init__(self, *populations, path, name=None, save_data=True):
        global T, ID, DT
        ID = 0
        self.t = None
        self.PG_duration = None
        self.PG_match_th = None
        self.neurons = {}
        self.synapses = []
        for population in populations:
            for n in range(population[0]):
                if len(population) == 3:
                    ref_t = population[2]
                else:
                    ref_t = 0
                neuron = population[1](ref_t=ref_t)
                self.neurons[neuron.ID] = neuron
        self.duration = 0
        self.name = name
        self.structure = False
        self.poly_groups = {}
        self.poly_group_history = []
        self.poly_ID_counts = {}
        self.save_data = save_data
        self.registered_inputs = []
        if save_data:
            date = datetime.now().strftime("%d-%B-%Y_%H-%M-%S")
            if self.name is not None:
                self.dir = os.path.join(path, self.name)
            else:
                self.dir = os.path.join(path, date)
            cnt = 1
            while os.path.exists(self.dir):
                if self.name:
                    self.dir = os.path.join(path, f"{self.name}_{cnt}")
                else:
                    self.dir = os.path.join(path, f"{date}_{cnt}")
                cnt += 1
            os.makedirs(self.dir, exist_ok=True)


    def update(self):
        for neuron in reversed(self.neurons):
            self.neurons[neuron].update(self.neurons)

    def create_input(self, spike_times=[], p=0.0, j=None, wj=None, dj=None, seed=None, trainable=False):
        inp = self.Input(spike_times, p, seed)
        self.add_neuron(inp)
        if j is not None:
            j = list(j)
            for n,ij in enumerate(j):
                if isinstance(wj, list):
                    w = wj[n]
                elif wj:
                    w = wj
                else:
                    w = Constants.W
                if isinstance(dj, list):
                    d = dj[n]
                elif dj:
                    d = dj
                else:
                    d = 1
                self.create_synapse(inp.ID, ij, w=w, d=d, trainable=trainable)
        return inp

    def add_neuron(self, neuron):
        self.neurons[neuron.ID] = neuron

    def create_synapse(self, i, j, w=10, d=1, trainable=True, partial=False):
        syn = self.Synapse(i, j, w, d, trainable, partial=partial)
        try:
            self.neurons[str(i)].down.append(syn)
        except:
            raise Exception(f"Neuron {i} not part of population")
        try:
            self.neurons[str(j)].up.append(syn)
        except:
            raise Exception(f"Neuron {j} not part of population")
        self.synapses.append(syn)
        return syn

    def delete_synapse(self, i, j):
        [self.synapses.remove(syn) for syn in self.synapses if syn.i == str(i) and syn.j == str(j)]
        [self.neurons[str(i)].down.remove(syn) for syn in self.neurons[str(i)].down if syn.i == str(i) and syn.j == str(j)]
        [self.neurons[str(j)].up.remove(syn) for syn in self.neurons[str(j)].up if syn.i == str(i) and syn.j == str(j)]

    def create_grid(self, diagonals, d, w, trainable, seed=None):
        dim = m.sqrt(len(self.neurons))
        if dim != int(dim):
            raise Exception("Error: population size must be a perfect square.")
        else:
            dim = int(dim)
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        matrix = np.reshape(list(self.neurons.keys()), (dim,dim))
        for row in range(dim):
            for col in range(dim):
                for i in range(-1, 2):
                    x = i + row
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        y = j + col
                        exp = (lambda x,y,row,col: (row == x or col == y) if not diagonals else True)
                        if exp(x, y, row, col) and 0 <= x < dim and 0 <= y < dim:
                            self.create_synapse(matrix[row][col],matrix[x][y], w=rng.choice(w), d=rng.choice(d), trainable=trainable)

    def create_random_connections(self, p, d, w, trainable, seed=None):
        self.structure = "grid"
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        for id in self.neurons:
            for id_2 in self.neurons:
                if rng.random() < p and id != id_2:
                    self.create_synapse(id, id_2, d=rng.choice(d), w=rng.choice(w), trainable=trainable)

    def create_random_distance_delayed_connections(self, p, w, trainable, seed=None):
            self.structure = "grid"
            if seed is not None:
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.default_rng()
            if not isinstance(w, list):
                w = [w]
            n = len(self.neurons)
            side = m.ceil(m.sqrt(n))
            steps = range(int(side))
            prod = list(itertools.product(steps, steps))
            pos = prod[:n]
            for neuron, p1 in zip(self.neurons, pos):
                for neuron2, p2 in zip(self.neurons, pos):
                    if rng.random() < p and type(self.neurons[neuron2]) != self.Input and neuron != neuron2:
                        d = round(np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)), 1)
                        self.create_synapse(neuron, neuron2, w=rng.choice(w), d=d, trainable=trainable)

    def create_watts_strogatz_connections(self, k, p, d, w, trainable, seed=None):
        # Must be fixed
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        if k >= len(self.neurons):
            raise Exception("Degree k cannot equal or exceed number of neurons.")
        n = len(self.neurons)
        side = m.ceil(m.sqrt(n))
        steps = range(int(side))
        prod = list(itertools.product(steps, steps))
        pos = prod[:n]
        pos_node = {}
        for neuron, ps in zip(self.neurons, pos):
            pos_node[str(ps)] = neuron
        for neuron, ps in zip(self.neurons, pos):
            neighbors = []
            diff = 1
            while len(neighbors) < k:
                temp = []
                for x in range(ps[0] - diff, ps[0] + diff + 1):
                    for y in range(ps[1] - diff, ps[1] + diff + 1):
                        if ps != (x,y) and (x,y) not in neighbors and (x,y) in pos:
                            temp.append((x,y))
                choice = rng.choice(range(len(temp)), min(k-len(neighbors), len(temp)), replace=False)
                [neighbors.append(temp[x]) for x in choice]
                diff += 1
            for ngb in neighbors:
                self.create_synapse(neuron, pos_node[str(ngb)], w=rng.choice(w), d=rng.choice(d), trainable=trainable)
        if k < len(self.neurons) - 1:
            for syn in self.synapses:
                if rng.random() < p:
                    neurons = self.neurons.copy()
                    neurons.pop(syn.i)
                    pairs = [(syn.i,syn.j) for syn in self.synapses]
                    while True:
                        new_j = rng.choice(list(neurons.keys()))
                        if (syn.i, new_j) not in pairs:
                            break
                    syn.j = new_j

    def create_barabasi_albert_connections(self, d, w, trainable, seed=None):
        # Must be fixed
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        ids = list(self.neurons.copy().keys())
        added_ids = []
        id1 = ids.pop(0)
        added_ids.append(id1)
        id2 = ids.pop(0)
        added_ids.append(id2)
        self.create_synapse(id2, id1, w=rng.choice(w), d=rng.choice(d), trainable=trainable)
        for i in range(len(ids)):
            id = ids.pop(0)
            n_syn = len(self.synapses)
            for added_id in added_ids:
                c = len(self.neurons[added_id].down)
                if rng.random() < c/n_syn:
                    self.create_synapse(id, added_id, w=rng.choice(w), d=rng.choice(d), trainable=trainable)
            added_ids.append(id)

    def create_ring_lattice_connections(self, k, d, w, trainable, seed=None):
        self.structure = "ring"
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        for neuron in self.neurons:
            neuron = int(neuron)
            for i in range(-k, k + 1):
                j = (neuron + i) % len(self.neurons)
                if neuron != j:
                    self.create_synapse(neuron, j, w=rng.choice(w), d=rng.choice(d), trainable=trainable)

    def create_directional_ring_lattice_connections(self, k, d, w, skip_n, trainable, seed=None):
        self.structure = "ring"
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        for neuron in self.neurons:
            neuron = int(neuron)
            for i in range(1, k + 1):
                j = (neuron + i) % len(self.neurons)
                if neuron != j:
                    if seed is not None:
                        self.create_synapse(neuron, j, w=rng.choice(w), d=rng.choice(d), trainable=trainable)
                    else:
                        if len(d) > 1:
                            self.create_synapse(neuron, j, w=w[0], d=d.pop(0), trainable=trainable)
                        else:
                            self.create_synapse(neuron, j, w=w[0], d=d[0], trainable=trainable)
        skips = []
        for skip in range(skip_n):
            i = rng.choice(len(self.neurons))
            j = rng.choice(len(self.neurons))
            while (i, j) in skips or i == j:
                i = rng.choice(len(self.neurons))
                j = rng.choice(len(self.neurons))
            skips.append((i, j))
            if seed is not None:
                self.create_synapse(i, j, w=rng.choice(w), d=rng.choice(d), trainable=trainable)
            else:
                if len(d) > 1:
                    self.create_synapse(i, j, w=w[0], d=d.pop(0), trainable=trainable)
                else:
                    self.create_synapse(i, j, w=w[0], d=d[0], trainable=trainable)


    def create_feed_forward_connections(self, d, w, trainable, p=None, n_layers=None, partial=False, seed=False):
        self.structure = "grid"
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        if n_layers is None:
            n_layers = int(np.ceil(np.sqrt(len(self.neurons))))
            n_row = n_layers
        else:
            n_row = len(self.neurons) / n_layers
            if n_row % 1 != 0:
                raise Exception("n neurons not divisible by layers")
            else:
                n_row = int(n_row)
        layers = [[] for x in range(n_layers)]
        pop_copy = list(self.neurons.copy())
        for layer in range(n_layers):
            for row in range(n_row):
                if pop_copy:
                    neuron = pop_copy.pop(0)
                    layers[layer].append(neuron)
        for layer in layers:
            if layers.index(layer) < len(layers) - 1:
                for neuron_i in layer:
                    for neuron_j in layers[layers.index(layer) + 1]:
                        if p is None:
                            self.create_synapse(neuron_i, neuron_j, w=rng.choice(w), d=rng.choice(d), trainable=trainable, partial=partial)
                        else:
                            if rng.random() < p:
                                self.create_synapse(neuron_i, neuron_j, w=rng.choice(w), d=rng.choice(d),
                                                    trainable=trainable, partial=partial)

    def plot_topology(self):
        global T
        plt.figure()
        G = nx.DiGraph()
        colors = []
        for n in self.neurons:
            G.add_node(self.neurons[n].ID, type="input" if isinstance(self.neurons[n], self.Input) else "neuron")
            if round(T-DT,1) in self.neurons[n].spikes:
                colors.append("r")
            elif isinstance(self.neurons[n], self.Input):
                colors.append("g")
            else:
                colors.append("b")
        for syn in self.synapses:
            G.add_edge(str(syn.j), str(syn.i), d=syn.d)
        if self.structure == "grid":
            neurons = []
            inputs = []
            [neurons.append(i) if G.nodes[i]["type"] == "neuron" else inputs.append(i) for i in G.nodes]
            if inputs:
                v_pos = np.linspace(0, 1, len(inputs))
                for input, p in zip(inputs, v_pos):
                    G.nodes[input]["pos"] = (0, p)
            side = m.ceil(m.sqrt(len(neurons)))
            v_steps = np.linspace(0,1,side)
            if inputs:
                h_steps = np.linspace(0,1, side + 1)
                h_steps = h_steps[1:]
            else:
                h_steps = v_steps
            prod=list(itertools.product(h_steps, v_steps))
            pos = prod[:len(neurons)]
            for node, p in zip(G.nodes, pos):
                G.nodes[node]["pos"] = p
            pos = nx.get_node_attributes(G, 'pos')
        elif self.structure == "ring":
            pos = nx.circular_layout(G)
        elif self.structure == "reservoir":
            pos = nx.kamada_kawai_layout(G)
        else:
            raise Exception("No valid network structure.")
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=150, alpha=1, label=False)
        labels = {}
        for node in G.nodes:
            labels[node] = int(node)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        ax = plt.gca()
        colors = plt.cm.brg(np.linspace(0, 1, int(MAX_DELAY / DT)))
        color_id = {}
        for d, c in enumerate(colors):
            color_id[round(d*DT + DT,1)] = c
        for e in G.edges(data=True):
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=color_id[round(e[2]["d"],1)],
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle=f"arc3,rad=0.2"), zorder=0)
        plt.axis('off')
        patches = []
        for i in range(1, MAX_DELAY + 1):
            patches.append(mpatches.Patch(color=color_id[i], label=f'd={i}'))
        #plt.legend(loc="lower left", fancybox=True, bbox_to_anchor=(1.06, 0), handles=patches, prop={'size': 8})
        norm = matplotlib.colors.Normalize(vmin=0, vmax=MAX_DELAY + 1)
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap("brg")), ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, "network.png"))
        return plt


    def plot_delays(self):
        fig, sub = plt.subplots()
        handles = []
        ls = ["solid", "dashed", "dotted", "dashdot"]
        for i, syn in enumerate(self.synapses):
            sub.plot(syn.d_hist["t"], syn.d_hist["d"], linestyle=ls[int((i/10) % 4)], linewidth=0.8)
            handles.append((syn.i, syn.j))

        sub.set_xlabel("Time (ms)")
        sub.set_ylabel("Delay (ms)")
        sub.set_ylim(0, MAX_DELAY)
        sub.set_yticks(np.arange(0, MAX_DELAY + 1, 2))
        sub.legend(handles, bbox_to_anchor=(1, 1), prop={"size":8}, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, "delays.png"))
        plt.close()

    def plot_mean_delays(self):
        fig, sub = plt.subplots()
        mean = []
        std = []
        for i in range(0, round(self.duration / DT)):
            d = [syn.d_hist["d"][i] for syn in self.synapses]
            mean.append(np.mean(d))
            std.append(np.std(d))
        sub.plot(np.arange(0, self.duration, DT), mean)
        sub.fill_between(np.arange(0, self.duration, DT), [m-s for m,s in zip(mean, std)], [m+s for m, s in zip(mean,std)], alpha=.3)
        sub.set_xlabel("Time (ms)")
        sub.set_ylabel("Delay (ms)")
        sub.set_ylim(0, MAX_DELAY)
        sub.set_yticks(np.arange(0, MAX_DELAY + 1, 2))
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, "mean_delays.png"))
        plt.close()

    def plot_raster(self, duration=None, classes=None, plot_pg=True, legend=True):
        fig, sub = plt.subplots()
        if duration is not None:
            spikes = []
            for n in self.neurons:
                sp = [x for x in list(self.neurons[n].spikes) if duration[0] < x < duration[1]]
                spikes.append(sp)
        else:
            spikes = [self.neurons[x].spikes for x in self.neurons]
        c = ["red" if isinstance(self.neurons[_],self.Input) else "black" for _ in self.neurons]
        sub.eventplot(spikes, colors=c, lineoffsets=1, linelengths=1, linewidths=0.5)
        sub.set_xlabel("Time (ms)")
        sub.set_ylabel("Neuron ID")
        sub.set_ylim([-1, len(self.neurons)])
        sub.set_yticks(range(len(self.neurons)))
        sub.set_yticklabels([x if type(self.neurons[x]) != self.Input else f"{x} (Input)" for x in self.neurons])
        sub.yaxis.set_major_locator(plt.MaxNLocator(20))
        count = Counter([tpg["poly_index"] for tpg in self.poly_group_history])
        sorted_count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True))
        if plot_pg:
            for key, value in zip(sorted_count.keys(), sorted_count.values()):
                self.poly_ID_counts[key] = value
            sorted_poly_group_history = sorted(self.poly_group_history, key=lambda x: list(sorted_count.keys()).index(x['poly_index']))
            for pg in sorted_poly_group_history:
                pg_start = pg["pattern_start"]
                if duration is None or (duration is not None and pg_start + self.PG_duration < duration[1]):
                    pg_ID = pg["poly_index"]
                    linewidth = max(2, int(np.ceil(len(self.neurons) / 100) + 1))
                    sub.hlines(y=-0.5, xmin=pg_start, xmax=pg_start + self.PG_duration, linewidth=linewidth, color=Constants.COLORS[list(sorted_count.keys()).index(pg_ID)])
            if legend:
                max_keys = list(sorted_count.keys())[0: classes if classes is not None else 2]
                patches = []
                for uid in max_keys:
                    patches.append(mpatches.Patch(color=Constants.COLORS[list(sorted_count.keys()).index(uid)], label=f'ID={uid} ({self.poly_ID_counts[uid]})'))
                if len(list(self.poly_ID_counts.values())) > (classes if classes is not None else 2):
                    patches.append(mpatches.Patch(color='none', label=f'Others ({sum(list(self.poly_ID_counts.values())[classes if classes is not None else 2:])})'))
                if len(patches) > 0:
                    if len(patches) <= 12:
                        plt.legend(handles=patches, ncol=1, loc="best", title="PG IDs")
                    else:
                        plt.legend(handles=patches, ncol=6, loc="upper center", columnspacing=0.5, prop={"size": 8}, title="PG IDs")
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, "spikes.png"))
        plt.close()

    def plot_membrane_potential(self, IDs=None):
        if IDs is not None:
            IDs = list(IDs)
        else:
            IDs = self.neurons.copy()
        fig, sub = plt.subplots()
        handles = []
        ls = ["solid", "dashed", "dotted", "dashdot"]
        for i, id in enumerate(IDs):
            if type(self.neurons[str(id)]) != self.Input:
                sub.plot(self.neurons[str(id)].v_hist["t"], self.neurons[str(id)].v_hist["v"], linestyle=ls[int((i/10) % 4)], linewidth=0.8)
                handles.append(id)
        sub.set_xlabel("Time (ms)")
        sub.set_ylabel("Potential (mV)")
        sub.set_ylim(-90, 40)
        sub.legend(handles, bbox_to_anchor=(1.05, 1), prop={"size": 8})
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, "potentials.png"))
        plt.close()

    def plot_spike_rate(self):
        fig, subs = plt.subplots(len(self.neurons))
        spikes = [np.histogram(self.neurons[x].spikes, bins=round(self.duration/10), range=[0, self.duration])[0] for x in self.neurons]
        for sub, sr in zip(subs, spikes):
            sub.plot(sr)
        '''
        sub.set_xlabel("Time (ms)")
        sub.set_ylabel("Neuron ID")
        sub.set_ylim([-1, len(self.neurons)])
        sub.set_yticks(range(len(self.neurons)))
        sub.set_yticklabels([x if type(self.neurons[x]) != Input else f"{x} (Input)" for x in self.neurons])
        '''
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, "SR.png"))
        plt.close()

    def create_video(self, fps):
        os.system(f"ffmpeg -y -r {fps} -i {self.dir}/t%10d.png -vcodec libx264 {self.dir}/output.mp4")
        os.system(f"ffmpeg -y -r {fps} -i {self.dir}/t%10d.png -vcodec msmpeg4 {self.dir}/output.wmv")

    def build_poly_groups(self, poly_index):
        inputs = [self.neurons[neuron] for neuron in self.neurons if isinstance(self.neurons[neuron], self.Input)]
        if all(len(inputs[x].poly_group.keys()) == len(inputs[x - 1].poly_group.keys()) and all([inputs[x].poly_group.keys(),inputs[x - 1].poly_group.keys()]) for x in
               range(1, len(inputs))):
            max_keys = [max(inp.poly_group.keys()) for inp in inputs]
            poly_id_t = [(inp.ID, max(inp.poly_group.keys())) for inp in inputs]
            if all(T - mk >= self.PG_duration for mk in max_keys) and all(pkv not in self.registered_inputs for pkv in poly_id_t):
                [self.registered_inputs.append(pkv) for pkv in poly_id_t]
                poly = {}
                for inp in inputs:
                    poly[inp.ID] = inp.poly_group[max(inp.poly_group.keys())]
                highest_match = 0
                highest_match_ID = None
                for poly_ID in self.poly_groups:
                    match, unique = Data.compare_poly(poly, self.poly_groups[poly_ID])
                    if unique:
                        if match / unique > highest_match:
                            highest_match = match / unique
                            highest_match_ID = poly_ID
                if highest_match < self.PG_match_th:
                    if max(max_keys) not in [tmpx["pattern_start"] for tmpx in self.poly_group_history]:
                        self.poly_groups[poly_index] = poly
                        self.poly_group_history.append({"pattern_start": max(max_keys), "poly_index": poly_index})
                        return 1
                elif max(max_keys) not in [tmpx["pattern_start"] for tmpx in self.poly_group_history]:
                    self.poly_group_history.append({"pattern_start": max(max_keys), "poly_index": highest_match_ID})
                    return 0
        return 0

    def run(self, duration, dt=0.1, save_post_model=False, record_topology=False, record_PG=True, PG_duration = 200, PG_match_th=0.6, show_process=True, save_plots=True, n_classes=2, raster_legend=True):
        global T, DT
        if self.t is not None:
            T = self.t
            self.t += duration
            duration += T
        else:
            self.t = duration
            T = 0.0
        DT = dt
        self.duration = duration
        self.PG_duration = PG_duration
        self.PG_match_th = PG_match_th
        tot_start = time.time()
        if self.save_data:
            Data.save_model(self, os.path.join(self.dir, "pre_sim_model.pkl"))
        poly_index = 0
        while T < self.duration:
            start = time.time()
            self.update()
            if record_PG:
                poly_index += self.build_poly_groups(poly_index)
            if record_topology:
                fig = self.plot_topology()
                fig.title(f"Time={T}ms")
                file_name = "t" + str(T).replace(".","").rjust(10,"0")
                fig.savefig(os.path.join(self.dir, f"{file_name}.png"))
                fig.close()
            if show_process:
                stop = time.time() - start
                prog = (T / duration) * 100
                print("\r |" + "#" * int(prog) + f"  {round(prog, 1) if T < duration - DT else 100}%| t={T}ms | Time per step: {round(stop,4)} sec", end="")
            T = round(T + DT,3)
        if self.save_data:
            self.save_neuron_data()
            self.save_synapse_data()
        if save_post_model:
            Data.save_model(self, os.path.join(self.dir, "post_sim_model.pkl"))
        if save_plots:
            self.plot_raster(classes=n_classes, legend=raster_legend)
            self.plot_topology()
            self.plot_delays()
        stop = time.time()
        if show_process:
            print(f"\nSimulation finished: {self.name}")
            print(f"\nElapsed time: {round((stop-tot_start)/60,1)}min")




    def save_neuron_data(self):
        global T
        data = {}
        for id in self.neurons:
            neuron = self.neurons[id]
            type_ = str(type(neuron))
            n_up = None if isinstance(neuron, self.Input) else len(neuron.up)
            n_down = len(neuron.down)
            up = None if isinstance(neuron, self.Input) else [syn.i for syn in neuron.up]
            down = [syn.j for syn in neuron.down]
            ref_t = None if isinstance(neuron, self.Input) else neuron.ref_t
            spikes = list(neuron.spikes)
            data[id] = {"duration" : T, "type" : type_, "n_up" : n_up, "n_down" : n_down, "up" : up, "down" : down, "ref_t" : ref_t, "spikes": spikes}
        with open(os.path.join(self.dir, "neuron_data.json"), "w") as file:
            json.dump(data, file)

    def save_synapse_data(self):
        global T
        data = {}
        for synapse in self.synapses:
            w = float(synapse.w)
            pre = synapse.pre_window
            post = synapse.post_window
            trainable = synapse.trainable
            d_hist = synapse.d_hist
            data[f"{synapse.i}_{synapse.j}"] = {"duration" : T, "w" : w, "pre_window" : pre, "post_window" : post, "trainable" : trainable, "d_hist" : d_hist["d"]}
        with open(os.path.join(self.dir, "synapse_data.json"), "w") as file:
            json.dump(data, file)

    class Synapse:
        def __init__(self, i, j, w, d, trainable, partial=False):
            self.i = i
            self.j = j
            self.w = w
            self.d = float(d)
            self.d_hist = {"t":[], "d":[]}
            self.pre_window = Constants.PRE_WINDOW
            self.post_window = Constants.POST_WINDOW
            self.spikes = []
            self.trainable = trainable
            self.partial = partial

        def add_spike(self):
            self.spikes.append({"t": T, "d": self.d, "w": self.w})

        def get_spikes(self):
            self.d_hist["t"].append(T)
            self.d_hist["d"].append(self.d)
            count = 0
            for spike in self.spikes:
                if round(spike["t"] + spike["d"], 1) == T:
                    count += spike["w"]
                    self.spikes.remove(spike)
            return count

        def F(self, delta_tdist):
            if self.trainable:
                dd = -3*m.tanh(delta_tdist/3)
                self.d += dd
                self.d = round(max(self.d, DT), 1 if DT==0.1 else 0)
                self.d = min(self.d, MAX_DELAY)

        def G(self, delta_t):
            if self.trainable and not self.partial:
                dd = (3/2)*m.tanh(2.5625-0.625*delta_t)+1.5
                self.d += dd
                self.d = round(self.d, 1 if DT == 0.1 else 0)
                self.d = min(self.d, MAX_DELAY)

    class Input:
        global T
        def __init__(self, spike_times=False, p=0.0, seed=False):
            global ID
            self.ID = str(ID)
            ID += 1
            self.spikes = deque()
            self.spike_times = spike_times
            self.p = p
            self.down = []
            self.poly_group = {}
            if seed:
                self.rng = np.random.default_rng(seed)
            else:
                self.rng = np.random.default_rng()

        def update(self, neurons=None, record_PG=None):
            if self.spike_times and T in self.spike_times:
                [syn.add_spike() for syn in self.down]
                self.spikes.append(T)
                if T not in self.poly_group.keys():
                    self.poly_group[T] = {}
            elif self.rng.random() < self.p and not self.spike_times:
                [syn.add_spike() for syn in self.down]
                self.spikes.append(T)

    class Neuron:
        def __init__(self, a, b, c, d, u, ref_t=0, v_init=False):
            global ID, DT
            self.ID = str(ID)
            ID += 1
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.v = c if not v_init else v_init
            self.u = u
            self.th = 30
            self.ref_t = ref_t
            self.refractory = 0
            self.v_hist = {"t":[], "v": []}
            self.u_hist = {"t":[], "u": []}
            self.spikes = deque()
            self.up = []
            self.down = []
            self.inputs = []
            self.poly_group = {}

        def update(self, neurons, record_pg=False):
            for syn in self.up:
                i = syn.get_spikes()
                if not self.refractory:
                    self.inputs.append({"I": i, "counter": 1})
            if self.inputs and not self.refractory:
                I = 0
                for inp in range(len(self.inputs)):
                    I += self.inputs[inp]["I"]
                    self.inputs[inp]["counter"] = round(self.inputs[inp]["counter"] - DT,3)
                self.inputs = [x for x in self.inputs if x["counter"] > 0]
            else:
                I = 0
            self.v += 0.5 * (0.04*self.v**2+5*self.v+140-self.u + I) * DT
            self.v += 0.5 * (0.04*self.v**2+5*self.v+140-self.u + I) * DT
            self.u += self.a*(self.b*self.v-self.u) * DT
            self.v = min(self.th, self.v)
            self.v_hist["t"].append(T)
            self.v_hist["v"].append(self.v)
            self.u_hist["t"].append(T)
            self.u_hist["u"].append(self.u)
            if self.th <= self.v:
                self.spikes.append(T)
                self.v = self.c
                self.u += self.d
                self.refractory = self.ref_t
                [syn.add_spike() for syn in self.down]
                self.inputs.clear()
                syn_list = []
                if T not in self.poly_group.keys():
                    self.poly_group[T] = {}
                for syn in self.up:
                    spikes = neurons[str(syn.i)].spikes
                    if spikes:
                        pre_spikes = []
                        pre_spike_t = []
                        post_spikes = []
                        for spike in spikes:
                            index = syn.d_hist["t"].index(spike)
                            delta_t = round((spike + syn.d_hist["d"][index]) - T, 1)
                            if syn.pre_window <= delta_t <= 0:
                                pre_spikes.append(delta_t)
                                pre_spike_t.append(spike)
                            elif syn.post_window >= delta_t > 0:
                                post_spikes.append(delta_t)
                        if pre_spikes:
                            min_pre = max(pre_spikes)
                            syn_list.append((syn,min_pre))
                            min_pre_index = pre_spikes.index(max(pre_spikes))
                            min_pre_t = pre_spike_t[min_pre_index]
                            if record_pg:
                                neurons[str(syn.i)].poly_group[min_pre_t][self.ID] = self.poly_group[T]
                        elif post_spikes:
                            min_post = min(post_spikes)
                            syn.G(min_post)
                if syn_list:
                    avg_delta_t = round(sum(x[1] for x in syn_list)/len(syn_list), 1 if DT == 0.1 else 0)
                    [syn[0].F(syn[1]-avg_delta_t) for syn in syn_list]
            else:
                self.refractory = max(0, np.round(self.refractory - DT, 1))


    class FS(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.1, b=0.2, c=-65, d=2, u=-14, ref_t=ref_t)


    class RS(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.02, b=0.2, c=-65, d=8, u=-14, ref_t=ref_t, v_init=-70)


    class RZ(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.1, b=0.26, c=-65, d=2, u=-16, ref_t=ref_t)


    class LTS(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.02, b=0.25, c=-65, d=2, u=-16, ref_t=ref_t)


    class TC(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.02, b=0.25, c=-65, d=0.05, u=-16, ref_t=ref_t)


    class IB(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.02, b=0.2, c=-55, d=4, u=-14, ref_t=ref_t)


    class CH(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.02, b=0.2, c=-50, d=2, u=-14, ref_t=ref_t)

    class POLY(Neuron):
        def __init__(self, ref_t):
            super().__init__(a=0.02, b=0.2, c=-65, d=2, u=-14, ref_t=ref_t)



    def save_fig(i):
        fig = plt.figure(i)
        file_name = "t" + str(i).replace(".", "").rjust(10, "0")
        fig.savefig(f"network_plots/{file_name}.png")
        fig.clf()
        plt.close(fig)


