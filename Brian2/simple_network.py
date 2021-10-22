from brian2 import *
import numpy as np
from Brian2 import izhikevich as iz
from matplotlib import pyplot as plt
import time
# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 1000
defaultclock.DT = 1 * ms
N = 25
DELAY_MIN = 1
DELAY_MAX = 10
D_WND = 0.1 * ms
CONN_P = 0.5
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
input = {"index": [], "time": []}
input["time"] = [x*ms for x in range(DURATION) if np.random.random() < 0.2]

input["index"] = [np.random.randint(0,2) for x in range(len(input["time"]))]
input_pop = SpikeGeneratorGroup(2, input["index"], input["time"])
# Create connections
input_syn = Synapses(input_pop, iz_pop, on_pre="v+=I")
input_syn.connect(i=[0,1,0,1,0,1,0,1,0,1], j=list(range(10)))
synapse = Synapses(iz_pop, iz_pop, on_pre="v+=I")

#i = [x for x in range(0, N - 1, 2)]
#j = [x for x in range(1, N, 2)]
#synapse.connect(i=i, j=j)

synapse.connect(p=CONN_P, condition="i!=j")
s_id = list(zip(synapse.get_states()["i"], synapse.get_states()["j"]))
synapse.delay = [int(rand()*DELAY_MAX-DELAY_MIN)*ms for _ in range(len(s_id))]


# Monitor
m = SpikeMonitor(iz_pop, "v", record=True)
s = StateMonitor(iz_pop, variables="v", record=True)
print("RUNNING SIMULATION:")
delays = [[] for _ in range(len(s_id))]
for t in range(DURATION):
    run(1*ms)
    prog = (t/DURATION)*100
    print("\r |" + "#"*int(prog) + f"  {round(prog,1) if t < DURATION - 1 else 100}%| ", end="")
    syn_data = synapse.get_states()
    t = syn_data["t"]

    for ind in range(len(syn_data["lastspike"])):
        pre = syn_data["lastspike_pre"][ind]
        post = syn_data["lastspike_post"][ind]
        if post == t:
            print(post, t)
        delays[ind].append(synapse.delay[ind])
        if post == t:
            if 0 <= (post - pre + synapse.delay[ind]) <= D_WND:
                synapse.delay[ind] -= 1 * ms
                print(synapse.delay[ind])
            else:
                synapse.delay[ind] += 1 * ms



spike_data = [[] for _ in range(N)]
for i, t in zip(m.i, m.t):
    spike_data[i].append(t)

fig, (sub1, sub2, sub3) = subplots(3,1)
colors = [(np.random.random(), np.random.random(), np.random.random()) for x in range(N)]

sub1.eventplot(spike_data, colors=colors[:N])
sub1.set_ylim([-0.5,N + 0.5])
sub1.set_xlim([0,DURATION/1000])
for i in range(N):
    sub2.plot(s.v[i], color=colors[i])
end_time = time.time()
print("Simulation time: " + str(round(end_time-start_time, 2)))

sub2.set_ylim([-0.1, 0.1])
sub2.set_xlim([0,len(s.v[0])])

for d in delays:
    sub3.plot(d)

sub3.set_xlim(0,DURATION)
plt.show()