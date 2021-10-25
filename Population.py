import numpy as np
from collections import deque
import time
import random
import math as m
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
global t, DT, ID


class Population:
    def __init__(self, *populations):
        global t, ID
        t = 0.0
        ID = 0
        self.neurons = {}
        self.synapses = []
        for population in populations:
            for n in range(population[0]):
                neuron = population[1]()
                self.neurons[neuron.ID] = neuron
        self.n = len(self.neurons)

    def update(self):
        for neuron in self.neurons:
            self.neurons[neuron].update(self.neurons)

    def add_neuron(self, neuron):
        self.neurons[neuron.ID] = neuron

    def create_synapse(self, i, j, w=10, d=1, trainable=False):
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

    def create_grid(self, corners=False):
        dim = m.sqrt(self.n)
        if dim != int(dim):
            raise Exception("Error: population size must be a perfect square.")
        else:
            dim = int(dim)
        matrix = np.reshape(list(self.neurons.keys()), (dim,dim))
        for row in range(dim):
            for col in range(dim):
                for i in range(-1, 2):
                    x = i + row
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        y = j + col
                        exp = (lambda x,y,row,col: (row == x or col == y) if not corners else True)
                        if exp(x, y, row, col) and 0 <= x < dim and 0 <= y < dim:
                            syn = self.create_synapse(matrix[row][col],matrix[x][y])
                            self.synapses.append(syn)

    def run(self, duration, dt=1):
        global t, DT
        DT = dt
        start = time.time()
        while t < duration:
            start = time.time()
            self.update()
            stop = time.time() - start
            prog = (t / duration) * 100
            print("\r |" + "#" * int(prog) + f"  {round(prog, 1) if t < duration - 1 else 100}%| Time per step: {stop}", end="")
            t = round(t + DT,3)
        stop = time.time()
        print(f"\nElapsed time: {stop-start}")

    class Synapse:
        def __init__(self, i, j, w, d, trainable):
            self.i = i
            self.j = j
            self.w = w
            self.d = d
            self.d_hist = [d]
            self.pre_window = 1
            self.post_window = 5
            self.counters = []
            self.trainable = trainable

        def add_spike(self):
            self.counters.append(self.d)

        def update(self):
            self.d_hist.append(self.d)
            if self.counters:
                count = self.counters.count(DT)
                self.counters = [round(x - DT,3) for x in self.counters if x > DT]
                return count * self.w
            else:
                return 0

        def change(self, change):
            self.d = max(self.d + change, DT)


class Neuron:
    def __init__(self,a,b,c,d,u,ref_t=2):
        global ID, DT
        self.ID = str(ID)
        ID += 1
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = c
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
            i = syn.update()
            if not self.refractory:
                self.inputs.append({"I": i, "counter": 1})
        if self.refractory:
            I = 0
        elif self.inputs:
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
        self.v_hist["t"].append(t)
        self.v_hist["v"].append(self.v)
        self.u_hist["t"].append(t)
        self.u_hist["u"].append(self.u)
        if self.th <= self.v:
            self.spikes.append(t)
            self.v = self.c
            self.u += self.d
            self.refractory = self.ref_t
            self.inputs = []
            [syn.add_spike() for syn in self.down]
            for syn in self.up:
                if syn.trainable:
                    spikes = neurons[str(syn.i)].spikes
                    if spikes:
                        pre_spikes = []
                        post_spikes = []
                        for spike in spikes:
                            diff = t - (spike + syn.d)
                            if diff >= 0:
                                pre_spikes.append(diff)
                            else:
                                post_spikes.append(abs(diff))
                        if pre_spikes:
                            min_pre = min(pre_spikes)
                            if min_pre <= syn.pre_window:
                                syn.change(min_pre - syn.pre_window)
                        elif post_spikes:
                            min_post = min(post_spikes)
                            if min_post <= syn.post_window:
                                syn.change(-(min_post-syn.post_window))
        else:
            self.refractory = max(0, self.refractory - DT)


class FS(Neuron):
    def __init__(self):
        super().__init__(a=0.1, b=0.2, c=-65, d=2, u=-14)


class RS(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-65, d=8, u=-14)


class RZ(Neuron):
    def __init__(self):
        super().__init__(a=0.1, b=0.26, c=-65, d=2, u=-16)


class LTS(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.25, c=-65, d=2, u=-16)


class TC(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.25, c=-65, d=0.05, u=-16)


class IB(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-55, d=4, u=-14)


class CH(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-50, d=2, u=-14)

class POLY(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-65, d=2, u=-14)

class Input:
    global t

    def __init__(self, spike_times=[], p=0.0, seed=None):
        global ID
        self.ID = str(ID)
        ID += 1
        self.spikes = deque()
        self.spike_times = spike_times
        self.p = p
        self.down = []
        #fix seeding
        if seed is not None:
            random.seed(seed)
    def update(self, neurons=None):
        if random.random() < self.p or t in self.spike_times:
            [syn.add_spike() for syn in self.down]
            self.spikes.append(t)







