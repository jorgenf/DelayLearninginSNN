from brian2 import *
import numpy as np
from matplotlib import pyplot as plt
import time
import seaborn as sns
# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 100
defaultclock.dt = 1*ms
N = 1
D_MAX = 1
D_MIN = 10
D_WND = 0.1 * ms
CONN_P = 0.5
F_P = 0.1
## Iz parameters
a = 0.02/ms
b = 0.2/ms
c = -65*mV
d = 8*mV/ms
I = 15*mV
ref_t = 2*ms
v_th = 30*mV



# Izikevich population
iz_pop = NeuronGroup(N, '''dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u : volt
du/dt = a*(b*v-u)                              : volt/second
                     ''',
                    threshold="v>v_th", reset='''
v = c
u = u + d
''',
                    method='euler', refractory=ref_t)

iz_pop.v = c
iz_pop.u = -14 * mV/ms
# Create input
input = {"index": [0,0,0,0,0,0,0,0,0,0], "time": [2.0,10.0,11.0,12.0, 22.0,21.0, 34.0,35.0, 66.0, 78.0]*ms}
#input["time"] = [x*ms for x in range(DURATION) if np.random.random() < F_P]
#input["index"] = list(np.zeros(len(input["time"])))

input_pop = SpikeGeneratorGroup(1, input["index"], input["time"])
# Create connections
input_syn = Synapses(input_pop, iz_pop, delay=1*ms, on_pre="v+=I")
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

end_time = time.time()
print("Simulation time: " + str(round(end_time-start_time, 2)))

nspike_data = [[] for _ in range(N)]
for i, t in zip(nm.i, nm.t):
    nspike_data[i].append(t)

ispike_data = [[] for _ in range(N)]
for i, t in zip(im.i, im.t):
    ispike_data[i].append(t)

sns.set()
fig, (sub1, sub2, sub3, sub4) = subplots(4,1,figsize=(25,15))
color = "blue"
colors = [(np.random.random(), np.random.random(), np.random.random()) for x in range(N)]
sub1.eventplot(nspike_data, colors=color)
#sub1.set_ylim([-0.5,N + 0.5])
sub1.set_xlim([0,DURATION/1000])
sub1.set_title("Neuron spikes")
sub1.set_ylabel("Neuron ID")
sub1.set_yticks([1])
for i in range(N):
    sub2.plot(s.v[i])
    sub2.set_title("Membrane potential")
sub2.set_ylim([-0.1, 0.05])
sub2.set_xlim([0,len(s.v[0])])
sub2.set_xticks(range(0,DURATION,2))
sub2.set_ylabel("V")
sub3.plot(s.u[0])
sub3.set_title("U-variable")
sub3.set_xlim([0, len(s.u[0])])
sub3.set_ylabel("u")
sub4.eventplot(ispike_data)
#sub3.set_ylim([-0.5,N + 0.5])
sub4.set_xlim([0,DURATION/1000])
sub4.set_yticks([1])
sub4.set_title("Input spikes")
sub4.set_xlabel("Time (ms)")
fig.suptitle("Brian2")
plt.savefig("../output/brian2", dpi=200)
#plt.show()