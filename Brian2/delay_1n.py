from brian2 import *
#set_device('cpp_standalone')
#prefs.devices.cpp_standalone.openmp_threads = 16
import numpy as np
from Brian2 import izhikevich as iz
from matplotlib import pyplot as plt
import time
import seaborn as sns
# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 1000
defaultclock.DT = 0.1 * ms
N = 1
I_N = 3
D_MAX = 20 * ms
D_MIN = 1 * ms
D_WND = 6 * ms
CONN_P = 0.5
F_P = 0.02
## Iz parameters
a = iz.a
b = iz.b
c = iz.c
d = iz.d
v_th = iz.v_max
I = 10 * mV
ref_t = 2 * ms

def update_d():
    syn_data = input_syn.get_states()
    neuron_data = iz_pop.get_states()
    for i,j,syn_ind in zip(syn_data["i"], syn_data["j"], range(len(syn_data["i"]))):
        post = neuron_data["lastspike"][j]
        delays[i].append(input_syn.delay[i])
        if post == t*ms:
            spike_times = [x[1] for x in input if x[0] == i and x[1] <= t*ms]
            pre = min(spike_times, key=lambda x: abs(x - post))
            if -D_WND <= (post - (pre + input_syn.delay[syn_ind])) < 0 and input_syn.delay[syn_ind] < D_MAX:
                input_syn.delay[syn_ind] += 1 * ms
            elif 0 < (post - (pre + input_syn.delay[syn_ind])) <= D_WND and input_syn.delay[syn_ind] > D_MIN:
                input_syn.delay[syn_ind] -= 1 * ms

# Izikevich population
iz_pop = NeuronGroup(N, iz.eqs
                     ,
                    threshold='v>v_th', reset=iz.reset,
                    method='euler', refractory=ref_t, events="sp: v>v_th")
iz_pop.v = c
iz_pop.u = -12*mV/second
iz_pop.run_on_event("sp", update_d())



# Create input
in_0 = [(0,x*ms) for x in range(DURATION) if np.random.random() < F_P]
in_1 = [(1,x*ms) for x in range(DURATION) if np.random.random() < F_P]
in_2 = [(2,x*ms) for x in range(DURATION) if np.random.random() < F_P]
input = sorted(in_0 + in_1 + in_2, key=lambda x: x[1])

input_pop = SpikeGeneratorGroup(I_N, [x[0] for x in input], [x[1] for x in input])
# Create connections
input_syn = Synapses(input_pop, iz_pop, on_pre="v+=I")
input_syn.connect(i=[0,1,2], j=[0,0,0])


s_id = list(zip(input_syn.get_states()["i"], input_syn.get_states()["j"]))
for i in range(I_N):
    input_syn.delay[i] = np.random.randint(2,9)*ms


# Monitor
im = SpikeMonitor(input_pop, record=True)
nm = SpikeMonitor(iz_pop, record=True)
s = StateMonitor(iz_pop, variables=("v","u"), record=True)
delays = [[] for _ in range(len(s_id))]
print("RUNNING SIMULATION:")



for t in range(DURATION):
    run(1*ms)
    prog = (t/DURATION)*100
    print("\r |" + "#"*int(prog) + f"  {round(prog,1) if t < DURATION - 1 else 100}%| ", end="")
    syn_data = input_syn.get_states()
    neuron_data = iz_pop.get_states()
    for i,j,syn_ind in zip(syn_data["i"], syn_data["j"], range(len(syn_data["i"]))):
        post = neuron_data["lastspike"][j]
        delays[i].append(input_syn.delay[i])
        if post == t*ms:
            spike_times = [x[1] for x in input if x[0] == i and x[1] <= t*ms]
            pre = min(spike_times, key=lambda x: abs(x - post))
            if -D_WND <= (post - (pre + input_syn.delay[syn_ind])) < 0 and input_syn.delay[syn_ind] < D_MAX:
                input_syn.delay[syn_ind] += 1 * ms
            elif 0 < (post - (pre + input_syn.delay[syn_ind])) <= D_WND and input_syn.delay[syn_ind] > D_MIN:
                input_syn.delay[syn_ind] -= 1 * ms



nspike_data = [[] for _ in range(N)]
for i, t in zip(nm.i, nm.t):
    nspike_data[i].append(t)
ispike_data = [[] for _ in range(I_N)]
for i, t in zip(im.i, im.t):
    ispike_data[i].append(t)


end_time = time.time()
print("Simulation time: " + str(round(end_time-start_time, 2)))

sns.set()
fig, (sub1, sub2, sub3) = subplots(3,1)
#colors = [(np.random.random(), np.random.random(), np.random.random()) for x in range(N)]

colors = ["red", "blue","green","indigo", "royalblue", "yellow", "peru", "palegreen"]
sub1.eventplot(nspike_data)
#sub1.set_ylim([-0.5,N - 0.5])
sub1.set_xlim([0,DURATION/1000])
sub1.set_title("Neuron spikes")
sub2.eventplot(ispike_data, color=colors[:len(ispike_data)])
sub2.set_ylim([-0.5,I_N - 0.5])
sub2.set_xlim([0,DURATION/1000])
sub2.set_title("Input spikes")
for i, ind in zip(delays, range(len(delays))):
    sub3.plot(i, color=colors[ind])
    sub3.set_title("Delays")
#sub2.set_ylim([-0.1, 0.1])
sub3.set_xlim([0,len(delays[0])])


plt.show()