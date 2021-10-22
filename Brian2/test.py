from brian2 import *
import numpy as np
from matplotlib import pyplot as plt
import time
from Brian2 import plots, izhikevich as iz

# https://brian.discourse.group/t/adapting-synaptic-delay-on-postsynaptic-spike/380


start_time = time.time()
# Parameters
DURATION = 50
TIMESTEP = 1
defaultclock.DT = 0.0001 * ms
N = 10
I_N = 2
D_MAX = 1
D_MIN = 10
D_WND = 0.1 * ms
CONN_P = 0.5
F_P = 0.9
## Iz parameters

print(iz.standard["eqs"])
# Izikevich population
iz_pop = NeuronGroup(N, iz.standard["eqs"]
                     ,
                    threshold='v>=v_max', reset=iz.standard["reset"],
                    method='euler', refractory="ref_t")

iz_pop.v = -65*mV
iz_pop.u = -14*volt/second

# Create input
np.random.seed(2)
in_0 = [(0,x*ms) for x in np.arange(0, DURATION, TIMESTEP) if np.random.random() < F_P]
in_1 = [(1,x*ms) for x in np.arange(0, DURATION, TIMESTEP) if np.random.random() < F_P]
input = sorted(in_0 + in_1, key=lambda x: x[1])

input_pop = SpikeGeneratorGroup(I_N, [x[0] for x in input], [x[1] for x in input])
# Create connections
input_syn = Synapses(input_pop, iz_pop, on_pre="v_post+=I")
input_syn.connect(p=CONN_P)
synapse = Synapses(iz_pop, iz_pop, on_pre="v_post+=I")

synapse.connect(i=[0,0,1,1,2,2,3,3,0,3,1,2], j=[1,2,0,3,0,3,1,2,3,0,2,1])
s_id = list(zip(synapse.get_states()["i"], synapse.get_states()["j"]))
synapse.delay = 5*ms



# Monitor
im = SpikeMonitor(input_pop, record=True)
nm = SpikeMonitor(iz_pop, record=True)
s = StateMonitor(iz_pop, variables=("v","u"), record=True)

print("RUNNING SIMULATION:")
for t in range(int(DURATION/TIMESTEP)):
    run(TIMESTEP*ms)
    prog = (t/int(DURATION/TIMESTEP))*100
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



plot_args = {"v":{"data":s.v, "mthd":plt.plot, "title": "Membrane potential", "y_label": "Volt", "x_label": "Time"},"u":{"data":s.u, "mthd":plt.plot, "title":"U-variable","x_label":"Time", "y_label":"U"}, "spikes":{"data":nspike_data, "mthd":plt.eventplot, "title":"Spikes", "x_label":"Time","y_label":"NeuronID"}, "input":{"data":ispike_data,"mthd":plt.eventplot,"title":"Input", "x_label":"Time","y_label":"NeuronID"}}
plots.plot_data(plot_args, DURATION)
