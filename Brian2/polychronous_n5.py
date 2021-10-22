from brian2 import *
import numpy as np
from matplotlib import pyplot as plt
import time
from Brian2 import plots, izhikevich as iz
import seaborn as sns
# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 30
TIMESTEP = 1
defaultclock.DT = 1 * ms
N = 5
I_N = 5
D_MAX = 1
D_MIN = 10
D_WND = 0.1 * ms
CONN_P = 0.5
F_P = 0.2
## Iz parameters
a = 0.02/ms
b = 0.2/ms
c = -65*mV
d = 2*mV/ms
v_th = 30*mV
I = 30 * mV
I_inp = 40 * mV
ref_t = 2 * ms

# Izikevich population
iz_pop = NeuronGroup(N,
                     '''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u : volt
du/dt = a*(b*v-u)                              : volt/second
                                                 '''
                     ,
                    threshold='v>v_th', reset='''
v = c
u = u + d
''',
                    method='euler')
iz_pop.v = c
iz_pop.u = -12 * mV/second

# Pattern 1
patterns = {"pattern1": [(0,5*ms),(1,0*ms)], "pattern2":[(2,0*ms),(0,5*ms)], "pattern3": [(3,0*ms),(0,1*ms)]}

pattern = patterns["pattern1"]
input = sorted(pattern, key=lambda x: x[1])

input_pop = SpikeGeneratorGroup(I_N, [x[0] for x in input], [x[1] for x in input])
# Create connections
input_syn = Synapses(input_pop, iz_pop, on_pre="v+=I_inp")
input_syn.connect(i=[0,1,2,3,4], j=[0,1,2,3,4])
input_syn.delay = 0
synapse = Synapses(iz_pop, iz_pop, on_pre="v+=I")

synapse.connect(i=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4], j=[1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3])
delays = [6,4,2,2,2,7,7,2,4,2,2,7,10,3,5,6,5,4,4,4]*ms

s_id = list(zip(synapse.get_states()["i"], synapse.get_states()["j"]))
synapse.delay = delays



# Monitor
im = SpikeMonitor(input_pop, record=True)
nm = SpikeMonitor(iz_pop, record=True)
s = StateMonitor(iz_pop, variables=("v","u"), record=True)

plots.plot_connectivity_nx((synapse.get_states()["i"], synapse.get_states()["j"], synapse.delay))

for t in range(int(DURATION/TIMESTEP)):
    run(TIMESTEP*ms)
    prog = (t/int(DURATION/TIMESTEP))*100
    print("\r |" + "#"*int(prog) + f"  {round(prog,1) if t < DURATION - 1 else 100}%| ", end="")

nspike_data = [[] for _ in range(N)]
for i, t in zip(nm.i, nm.t):
    nspike_data[i].append(t)

ispike_data = [[] for _ in range(N)]
for i, t in zip(im.i, im.t):
    ispike_data[i].append(t)

end_time = time.time()
print("Simulation time: " + str(round(end_time-start_time, 2)))


#args = {"plot":{"mthd": plt.eventplot, "data":nspike_data, "title": "Neuron Spikes", "y_label": "Neuron ID", "x_label": "Time"}}
#plots.plot_data(args)



sns.set()
fig, sub1 = subplots(1,1)
np.random.seed(1)
colors = ["red", "blue","green","indigo", "royalblue", "peru", "palegreen", "yellow"]
colors += [(np.random.random(), np.random.random(), np.random.random()) for x in range(N)]
nspike_data.insert(0,[])
sub1.eventplot(nspike_data, colors=colors[:len(nspike_data)], linewidths=3)
sub1.set_ylim([0.5,N + 0.5])
sub1.set_yticks(range(1,6))
#sub1.set_xlim([0,DURATION/1000])
sub1.set_title("Neuron spikes")
sub1.set_xlabel("Time (s)")
sub1.set_ylabel("Neuron ID")
plt.show()
'''
for i in range(N):
   sub2.plot(s.u[i], color=colors[i])
   sub2.set_title("Neuron u-variable")
   sub2.set_xlim([0, len(s.u[0])])
plt.show()

for i in range(N):
    sub1.plot(s.v[i], color=colors[i])
    sub1.set_title("Membrane potential")
sub1.set_ylim([-0.1, 0.04])
sub1.set_xlim([0,len(s.v[0])])
sub1.set_xlabel(f"Time (dt={defaultclock.dt}s)")
sub1.set_ylabel("Membrane potential (V)")
for i in range(N):
    sub3.plot(s.u[i], color=colors[i])
    sub3.set_title("Neuron u-variable")
    sub3.set_xlim([0, len(s.u[0])])
#sub4.eventplot(ispike_data, color=colors[:1])
#sub4.set_ylim([-0.5,I_N - 0.5])
#sub4.set_xlim([0,DURATION/1000])
#sub4.set_title("Input spikes")

plt.show()
'''