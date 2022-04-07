import os
from datetime import datetime
import numpy as np
from collections import deque
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
from scipy.stats import binned_statistic
sns.set()
global T, DT, ID

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
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def update(self, neurons=None):
        if self.spike_times and T in self.spike_times:
            [syn.add_spike() for syn in self.down]
            self.spikes.append(T)
        elif self.rng.random() < self.p and not self.spike_times:
            [syn.add_spike() for syn in self.down]
            self.spikes.append(T)


class Population:
    def __init__(self, *populations, path, name=False, save_data=True):
        global T, ID
        T = 0.0
        ID = 0
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
        self.save_data = save_data
        if save_data:
            date = datetime.now().strftime("%d-%B-%Y_%H-%M-%S")
            if self.name:
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

    def create_input(self, spike_times=[], p=0.0, j=False, wj=False, dj=False, seed=False, trainable=False):
        inp = Input(spike_times, p, seed)
        self.add_neuron(inp)
        if j:
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

    def create_synapse(self, i, j, w=10, d=1, trainable=True):
        syn = self.Synapse(i, j, w, d, trainable)
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

    def create_grid(self, diagonals, d, w, trainable, seed=False):
        dim = m.sqrt(len(self.neurons))
        if dim != int(dim):
            raise Exception("Error: population size must be a perfect square.")
        else:
            dim = int(dim)
        if seed:
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

    def create_random_connections(self, p, d, w, trainable, seed=False):
        self.structure = "grid"
        if seed:
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

    def create_random_distance_delayed_connections(self, p, w, trainable, seed=False):
            self.structure = "grid"
            if seed:
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
                    if rng.random() < p and type(self.neurons[neuron2]) != Input and neuron != neuron2:
                        d = round(np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)), 1)
                        self.create_synapse(neuron, neuron2, w=rng.choice(w), d=d, trainable=trainable)

    def create_watts_strogatz_connections(self, k, p, d, w, trainable, seed=False):
        # Must be fixed
        if seed:
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

    def create_barabasi_albert_connections(self, d, w, trainable, seed=False):
        # Must be fixed
        if seed:
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

    def create_ring_lattice_connections(self, k, d, w, trainable, seed=False):
        self.structure = "ring"
        if seed:
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

    def create_directional_ring_lattice_connections(self, k, d, w, trainable, seed=False):
        self.structure = "ring"
        if seed:
            rng = np.random.default_rng(seed)
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        for neuron in self.neurons:
            neuron = int(neuron)
            for i in range(1, k + 1):
                j = (neuron + i) % len(self.neurons)
                if neuron != j:
                    if seed:
                        self.create_synapse(neuron, j, w=rng.choice(w), d=rng.choice(d), trainable=trainable)
                    else:
                        if len(d) > 1:
                            self.create_synapse(neuron, j, w=w[0], d=d.pop(0), trainable=trainable)
                        else:
                            self.create_synapse(neuron, j, w=w[0], d=d[0], trainable=trainable)

    def create_feed_forward_connections(self, d, w, trainable, seed=False):
        self.structure = "grid"
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        if not isinstance(d, list):
            d = [d]
        if not isinstance(w, list):
            w = [w]
        n_layers = int(np.ceil(np.sqrt(len(self.neurons))))
        layers = [[] for x in range(n_layers)]
        pop_copy = list(self.neurons.copy())
        for layer in range(n_layers):
            for row in range(n_layers):
                if pop_copy:
                    neuron = pop_copy.pop(0)
                    layers[layer].append(neuron)
        for layer in layers:
            if layers.index(layer) < len(layers) - 1:
                for neuron_i in layer:
                    for neuron_j in layers[layers.index(layer) + 1]:
                        self.create_synapse(neuron_i, neuron_j, w=rng.choice(w), d=rng.choice(d), trainable=trainable)

    def plot_topology(self):
        global T
        plt.figure()
        G = nx.DiGraph()
        colors = []
        for n in self.neurons:
            G.add_node(self.neurons[n].ID, type="input" if isinstance(self.neurons[n], Input) else "neuron")
            if round(T-DT,1) in self.neurons[n].spikes:
                colors.append("r")
            elif isinstance(self.neurons[n], Input):
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
        plt.close()


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

    def plot_raster(self, duration=False):
        fig, sub = plt.subplots()
        if duration:
            spikes = []
            for n in self.neurons:
                sp = [x for x in list(self.neurons[n].spikes) if duration[0] < x < duration[1]]
                spikes.append(sp)
        else:
            spikes = [self.neurons[x].spikes for x in self.neurons]
        sub.eventplot(spikes, colors='black', lineoffsets=1, linelengths=1, linewidths=0.5)
        sub.set_xlabel("Time (ms)")
        sub.set_ylabel("Neuron ID")
        sub.set_ylim([-1, len(self.neurons)])
        sub.set_yticks(range(len(self.neurons)))
        sub.set_yticklabels([x if type(self.neurons[x]) != Input else f"{x} (Input)" for x in self.neurons])
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, "spikes.png"))
        plt.close()

    def plot_membrane_potential(self, IDs=False):
        if IDs:
            IDs = list(IDs)
        else:
            IDs = self.neurons.copy()
        fig, sub = plt.subplots()
        handles = []
        ls = ["solid", "dashed", "dotted", "dashdot"]
        for i, id in enumerate(IDs):
            if type(self.neurons[str(id)]) != Input:
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

    def run(self, duration, dt=0.1, save_post_model=False, record=False, show_process=True):
        self.duration = duration
        global T, DT
        DT = dt
        tot_start = time.time()
        if self.save_data:
            Data.save_model(self, os.path.join(self.dir, "pre_sim_model.pkl"))
        while T < self.duration:
            start = time.time()
            self.update()
            if show_process:
                stop = time.time() - start
                prog = (T / duration) * 100
                print("\r |" + "#" * int(prog) + f"  {round(prog, 1) if T < duration - DT else 100}%| t={T}ms | Time per step: {round(stop,4)} sec", end="")
            T = round(T + DT,3)
            if record:
                fig = self.plot_topology()
                fig.title(f"Time={T}ms")
                file_name = "t" + str(T).replace(".","").rjust(10,"0")
                fig.savefig(os.path.join(self.dir, f"{file_name}.png"))
                fig.close()
        if self.save_data:
            self.save_neuron_data()
            self.save_synapse_data()
        if save_post_model:
            Data.save_model(self, os.path.join(self.dir, "post_sim_model.pkl"))
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
            n_up = None if isinstance(neuron, Input) else len(neuron.up)
            n_down = len(neuron.down)
            up = None if isinstance(neuron, Input) else [syn.i for syn in neuron.up]
            down = [syn.j for syn in neuron.down]
            ref_t = None if isinstance(neuron, Input) else neuron.ref_t
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
        def __init__(self, i, j, w, d, trainable):
            self.i = i
            self.j = j
            self.w = w
            self.d = float(d)
            self.d_hist = {"t":[], "d":[]}
            self.pre_window = -10
            self.post_window = 7
            self.spikes = []
            self.trainable = trainable

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
            if self.trainable:
                dd = (3/2)*m.tanh(2.5625-0.625*delta_t)+1.5
                self.d += dd
                self.d = round(self.d, 1 if DT == 0.1 else 0)
                self.d = min(self.d, MAX_DELAY)


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
        self.v_hist = {"t":[], "v":[]}
        self.u_hist = {"t":[], "u":[]}
        self.spikes = deque()
        self.up = []
        self.down = []
        self.inputs = []

    def update(self, neurons):
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
            syn_list  = []
            for syn in self.up:
                spikes = neurons[str(syn.i)].spikes
                if spikes:
                    pre_spikes = []
                    post_spikes = []
                    for spike in spikes:
                        delta_t = (spike + syn.d) - T
                        if syn.pre_window <= delta_t <= 0:
                            pre_spikes.append(delta_t)
                        elif syn.post_window >= delta_t > 0:
                            post_spikes.append(delta_t)
                    if pre_spikes:
                        min_pre = max(pre_spikes)
                        syn_list.append((syn,min_pre))
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


