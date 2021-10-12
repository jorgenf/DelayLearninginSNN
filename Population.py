import numpy as np
from collections import deque
import time

global t
ID = 0

class Population():
    def __init__(self, *populations):
        self.neurons = {}
        for population in populations:
            for n in range(population[0]):
                neuron = population[1]()
                self.neurons[neuron.ID] = neuron


class Neuron:
    def __init__(self,a,b,c,d):
        global ID
        self.ID = str(ID)
        ID += 1
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = c
        self.u = -14
        self.th = 30
        self.v_hist = deque()
        self.u_hist = deque()
        self.spikes = deque()

    def update(self, I):
        self.v += 0.5*(0.04*self.v**2+5*self.v+140-self.u+I)
        self.v += 0.5*(0.04*self.v**2+5*self.v+140-self.u+I)
        self.u += self.a*(self.b*self.v-self.u)
        self.v_hist.append(self.v)
        self.u_hist.append(self.u)
        if self.th <= self.v:
            self.spikes.append(t)
            self.v = self.c
            self.u += self.d


class FS(Neuron):
    def __init__(self):
        super().__init__(a=0.1, b=0.2, c=-65, d=2)


class RS(Neuron):
    def __init__(self):
        super().__init__(a=0.02, b=0.2, c=-65, d=8)


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

start = time.time()
pop = Population((10000, CH),(5000,IB), (1000,TC))

stop = time.time()
print(stop-start)
print(len(pop.neurons))

