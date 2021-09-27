from brian2 import *
import numpy as np
import izhikevich as iz
from matplotlib import pyplot as plt
import time
import seaborn as sns
# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 1000
defaultclock.dt = 0.1*ms
N = 4
I_N = 2
D_MAX = 1
D_MIN = 10
D_WND = 0.1 * ms
CONN_P = 0.5
F_P = 0.2
## Iz parameters
a = iz.a
b = iz.b
c = iz.c
d = iz.d
v_th = iz.v_max
I = 10 * mV
ref_t = 2 * ms
reset = iz.reset

# Izikevich population
def create_population(N, eqs, threshold, reset, method, refractory):
    iz_pop = NeuronGroup(N, iz.eqs,
                    threshold='v>v_th', reset=reset,
                    method='euler', refractory=ref_t)
    iz_pop.v = iz.c
    return iz_pop

def create_input(input_neurons, duration, fire_probability):
    input = []
    for i in range(input_neurons):
        inp = [(i, x*ms) for x in range(duration) if np.random.random() < fire_probability]
        input += inp
    input = sorted(input, key=lambda x: x[1])
    input_pop = SpikeGeneratorGroup(I_N, [x[0] for x in input], [x[1] for x in input])
    return input_pop


def create_grid_connections(synapse, dim):
    x = dim[0]
    y = dim[1]
    pairs = []
    mat = np.arange(x*y).reshape(x,y)
    i = 0
    for row in range(x):
        for col in range(y):
            for x in range(-1,2):
                for y in range(-1,2):
                    x = x + row
                    y = y + col
                    if (row != x or col != y) and x >= 0 and y >= 0:
                        try:
                            pairs.append((mat[row][col],mat[x][y]))
                        except:
                            pass
    return pairs

    synapse.connect(i=[0,0,1,1,2,2,3,3,0,3,1,2], j=[1,2,0,3,0,3,1,2,3,0,2,1])
    s_id = list(zip(synapse.get_states()["i"], synapse.get_states()["j"]))
    synapse.delay = 5*ms


# Monitor
im = SpikeMonitor(input_pop, record=True)
nm = SpikeMonitor(iz_pop, record=True)
s = StateMonitor(iz_pop, variables=("v","u"), record=True)

print("RUNNING SIMULATION:")
for t in range(DURATION):
    run(1*ms)
    prog = (t/DURATION)*100
    print("\r |" + "#"*int(prog) + f"  {round(prog,1) if t < DURATION - 1 else 100}%| ", end="")
    syn_data = synapse.get_states()


nspike_data = [[] for _ in range(N)]
for i, t in zip(nm.i, nm.t):
    nspike_data[i].append(t)

ispike_data = [[] for _ in range(N)]
for i, t in zip(im.i, im.t):
    ispike_data[i].append(t)

end_time = time.time()
print("Simulation time: " + str(round(end_time-start_time, 2)))

sns.set()
fig, (sub1, sub2, sub3, sub4) = subplots(4,1)
#colors = [(np.random.random(), np.random.random(), np.random.random()) for x in range(N)]
colors = ["red", "blue","green","indigo", "royalblue", "yellow", "peru", "palegreen"]
sub1.eventplot(nspike_data, colors=colors[:N])
sub1.set_ylim([-0.5,N - 0.5])
sub1.set_xlim([0,DURATION/1000])
sub1.set_title("Neuron spikes")
for i in range(N):
    sub2.plot(s.v[i], color=colors[i])
    sub2.set_title("Membrane potential")
sub2.set_ylim([-0.1, 0.1])
sub2.set_xlim([0,len(s.v[0])])
for i in range(N):
    sub3.plot(s.u[i], color=colors[i])
    sub3.set_title("Neuron u-variable")
    sub3.set_xlim([0, len(s.u[0])])
sub4.eventplot(ispike_data, color=colors[:1])
sub4.set_ylim([-0.5,I_N - 0.5])
sub4.set_xlim([0,DURATION/1000])
sub4.set_title("Input spikes")
plt.show()
'''