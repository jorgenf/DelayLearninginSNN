from brian2 import *
import numpy as np
import izhikevich as iz
from matplotlib import pyplot as plt
import time
import seaborn as sns
# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 500
defaultclock.dt = 1*ms
N = 1
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
v_max = iz.v_max
I = 10 * mV
ref_t = 10 * ms

# Izikevich population
iz_pop = NeuronGroup(N, iz.eqs
                     ,
                    threshold='v>v_max', reset=iz.reset,
                    method='euler', refractory=ref_t)
iz_pop.v = c
iz_pop.u = -15.0*mV/second

# Create input
input = {"index": [], "time": []}
input["time"] = [x*ms for x in range(DURATION) if np.random.random() < F_P]
input["index"] = list(np.zeros(len(input["time"])))

input_pop = SpikeGeneratorGroup(1, input["index"], input["time"])
# Create connections
input_syn = Synapses(input_pop, iz_pop, on_pre="v+=I")
input_syn.connect(i=[0], j=[0])

# Monitor
im = SpikeMonitor(input_pop, record=True)
nm = SpikeMonitor(iz_pop, record=True)
s = StateMonitor(iz_pop, variables=("v","u"), record=True)

print("RUNNING SIMULATION:")

for t in range(DURATION):
    run(1*ms)
    prog = (t/DURATION)*100
    print("\r |" + "#"*int(prog) + f"  {round(prog,1) if t < DURATION - 1 else 100}%| ", end="")
    ns = iz_pop.get_states()

nspike_data = [[] for _ in range(N)]
for i, t in zip(nm.i, nm.t):
    nspike_data[i].append(t)

ispike_data = [[] for _ in range(N)]
for i, t in zip(im.i, im.t):
    ispike_data[i].append(t)

sns.set()
fig, (sub1, sub2, sub3, sub4) = subplots(4,1)
color = "blue"
colors = [(np.random.random(), np.random.random(), np.random.random()) for x in range(N)]
sub1.eventplot(nspike_data, colors=color)
#sub1.set_ylim([-0.5,N + 0.5])
sub1.set_xlim([0,DURATION/1000])
sub1.set_title("Neuron spikes")
for i in range(N):
    sub2.plot(s.v[i], color=color)
    sub2.set_title("Membrane potential")
sub2.set_ylim([-0.1, 0.1])
sub2.set_xlim([0,len(s.v[0])])
sub3.plot(s.u[0], color=color)
sub3.set_title("Neuron u-variable")
sub3.set_xlim([0, len(s.u[0])])
sub4.eventplot(ispike_data, color=color)
#sub3.set_ylim([-0.5,N + 0.5])
sub4.set_xlim([0,DURATION/1000])
sub4.set_title("Input spikes")
end_time = time.time()
print("Simulation time: " + str(round(end_time-start_time, 2)))
plt.show()