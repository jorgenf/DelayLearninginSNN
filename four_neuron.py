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
defaultclock.dt = 1*ms
N = 4
I_N = 2
D_MAX = 1
D_MIN = 10
D_WND = 0.1 * ms
CONN_P = 0.5
F_P = 0.1
## Iz parameters
a = iz.a
b = iz.b
c = iz.c
d = iz.d
v_th = iz.v_max
I = 10 * mV
ref_t = 10 * ms

# Izikevich population
iz_pop = NeuronGroup(N, iz.eqs
                     ,
                    threshold='v>v_th', reset=iz.reset,
                    method='euler', refractory=ref_t)
iz_pop.v = c

# Create input
in_0 = [(0,x*ms) for x in range(DURATION) if np.random.random() < F_P]
in_1 = [(1,x*ms) for x in range(DURATION) if np.random.random() < F_P]
input = sorted(in_0 + in_1, key=lambda x: x[1])

input_pop = SpikeGeneratorGroup(I_N, [x[0] for x in input], [x[1] for x in input])
# Create connections
input_syn = Synapses(input_pop, iz_pop, on_pre="v+=I")
input_syn.connect(i=[0,1], j=[0,1])
synapse = Synapses(iz_pop, iz_pop, on_pre="v+=I")

synapse.connect(i=[0,0,1,1,2,2,3,3], j=[1,2,0,3,0,3,1,2])
s_id = list(zip(synapse.get_states()["i"], synapse.get_states()["j"]))
synapse.delay = 1*ms


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