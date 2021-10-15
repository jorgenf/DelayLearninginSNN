import numpy as np
from collections import deque
import time
import random
import math as m
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
global t
ID = 0


class Population():
    def __init__(self, *populations):
        global t
        t = 0
        self.neurons = {}
        self.synapses = []
        for population in populations:
            for n in range(population[0]):
                neuron = population[1]()
                self.neurons[neuron.ID] = neuron
        self.n = len(self.neurons)

    def update(self):
        for neuron in self.neurons:
            self.neurons[neuron].update()

    def add_neuron(self, neuron):
        self.neurons[neuron.ID] = neuron

    def create_synapse(self, i, j, w=10, d=1):
        syn = self.Synapse(i,j,w, d)
        try:
            self.neurons[str(i)].down.append(syn)
        except:
            raise Exception(f"Neuron {i} not part of population")
        try:
            self.neurons[str(j)].up.append(syn)
        except:
            raise Exception(f"Neuron {j} not part of population")
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

    def run(self, duration):
        global t
        while t < duration:
            prog = (t / duration) * 100
            print("\r |" + "#" * int(prog) + f"  {round(prog, 1) if t < duration - 1 else 100}%| ", end="")
            self.update()
            t += 1

    class Synapse:
        def __init__(self, i, j, w, d):
            self.i = i
            self.j = j
            self.w = w
            self.d = d
            self.que = deque(False for x in range(d))

        def pop(self):
            return self.que.pop()

        def push(self, spike):
            self.que.appendleft(spike)

        def update(self, change):
            for i in range(abs(change)):
                if change > 0:
                    self.que.appendleft(False)
                elif change < 0:
                    self.que.popleft()


class Neuron:
    def __init__(self,a,b,c,d,u=-14):
        global ID
        self.ID = str(ID)
        ID += 1
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = c
        self.u = u
        self.th = 30
        self.ref_t = 3
        self.refractory = 0
        self.v_hist = deque()
        self.u_hist = deque()
        self.spikes = deque()
        self.up = []
        self.down = []

    def update(self):
        I = 0
        if self.up and not self.refractory:
            I += sum([syn.pop() * syn.w for syn in self.up])
        else:
            [syn.pop() * syn.w for syn in self.up]
        self.v += 0.5*(0.04*self.v**2+5*self.v+140-self.u+I)
        self.v += 0.5*(0.04*self.v**2+5*self.v+140-self.u+I)
        self.v = min(self.th, self.v)
        self.u += self.a*(self.b*self.v-self.u)
        self.v_hist.append(self.v)
        self.u_hist.append(self.u)
        if self.th <= self.v:
            self.spikes.append(t)
            self.v = self.c
            self.u += self.d
            self.refractory = self.ref_t
            [syn.push(True) for syn in self.down]
        else:
            self.refractory = max(0, self.refractory - 1)
            [syn.push(False) for syn in self.down]

class FS(Neuron):
    def __init__(self):
        super().__init__(a=0.1, b=0.2, c=-65, d=2)


class RS(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-65, d=8, u=-14)


class RZ(Neuron):
    def __init__(self):
        super().__init__(a=0.1, b=0.26, c=-65, d=2)


class LTS(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.25, c=-65, d=2)


class TC(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.25, c=-65, d=0.05)


class IB(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-55, d=4)


class CH(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-50, d=2)

class Input:
    global t

    def __init__(self, spike_times=[], input_array=[], p=0.0, seed=None):
        global ID
        self.ID = str(ID)
        ID += 1
        self.spikes = deque()
        self.spike_times = spike_times
        self.p = p
        self.input_array = input_array
        self.down = []
        #fix seeding
        if seed is not None:
            random.seed(seed)
    def update(self):
        if random.random() < self.p or t in self.spike_times:

            [syn.push(True) for syn in self.down]
            self.spikes.append(t)
        elif self.input_array:
            val = self.input_array.pop(0)
            [syn.push(val) for syn in self.down]
        else:
            [syn.push(False) for syn in self.down]






