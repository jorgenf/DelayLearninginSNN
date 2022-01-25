from Population import *
import time
import numpy as np
import multiprocessing as mp
import itertools
import os
import random
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator

COLORS = ["red", "blue", "green", "indigo", "royalblue", "peru", "palegreen", "yellow"]
COLORS += [(np.random.random(), np.random.random(), np.random.random()) for x in range(20)]

cm = 1/2.54
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', titlesize=14)
plt.rcParams['figure.figsize'] = (15*cm, 10*cm)
#plt.rcParams['figure.figsize'] = (18/2*cm, 15/2*cm)



def high_input_test():
    DURATION = 100
    pop = Population((1,RS))
    input1 = Input(spike_times=[2.0,4.0,6.0,18.0,19.0])
    pop.add_neuron(input1)
    input2 = Input(spike_times=[2.0,4.0,6.0,18.0,19.0])
    pop.add_neuron(input2)
    input3 = Input(spike_times=[2.0,3,4.0,6.0,18.0,19.0])
    pop.add_neuron(input3)
    input4 = Input(spike_times=[2.0,3,4.0,6.0,18.0,19.0])
    pop.add_neuron(input4)
    input5 = Input(spike_times=[2.0,3,4.0,6.0,18.0,19.0])
    pop.add_neuron(input5)
    input6 = Input(spike_times=[2.0,3,4.0,6.0,18.0,19.0])
    pop.add_neuron(input6)
    pop.create_synapse(input1.ID, "0")
    pop.create_synapse(input2.ID, "0")
    pop.create_synapse(input3.ID, "0")
    pop.create_synapse(input4.ID, "0")
    pop.create_synapse(input5.ID, "0")
    pop.create_synapse(input6.ID, "0")
    pop.run(DURATION)
    fig, (sub1,sub2,sub3,sub4) = plt.subplots(4,1,figsize=(25,15))
    sub1.eventplot(pop.neurons["0"].spikes)
    sub1.set_ylabel("Neuron ID")
    sub1.set_xlim([0,DURATION])
    sub1.set_yticks([1])
    sub1.set_title("Neuron spikes")
    sub2.plot(pop.neurons["0"].v_hist)
    sub2.set_xlim([0,DURATION])
    sub2.set_ylim([-100,50])
    sub2.set_ylabel("mV")
    sub2.set_title("Membrane potential")
    sub2.set_xticks(range(0,DURATION,2))
    sub3.plot(pop.neurons["0"].u_hist)
    sub3.set_xlim([0,DURATION])
    sub3.set_title("U-variable")
    sub3.set_yticks(range(int(min(pop.neurons["0"].u_hist)), int(max(pop.neurons["0"].u_hist))))
    sub3.set_ylabel("u")
    events = sorted(pop.neurons[input1.ID].spikes)
    sub4.eventplot(events)
    sub4.set_xlim([0,DURATION])
    sub4.set_yticks([1])
    sub4.set_title("Input spikes")
    sub4.set_xlabel("Time (ms)")
    fig.suptitle("Jørgen2")
    plt.show()


def neuron_response(start, end, step, duration, dt):
    delays = []
    for W in np.arange(start,end,step):
        DURATION = duration
        dt = dt
        pop = Population((1, RS))
        input = Input(spike_times=[21.0])
        pop.add_neuron(input)
        pop.create_synapse(input.ID, 0, w=W, d=1)
        pop.run(DURATION, dt=dt)
        delays.append(round(pop.neurons["0"].spikes[0]-(pop.neurons["1"].spikes[0] + 1),3))
    plt.plot(np.arange(start,end, step), delays)
    plt.yticks(np.arange(0,14,0.5 if dt < 1 else 1))
    plt.ylim([0,14])
    plt.xticks(np.arange(round(start), round(end) + 1,1))
    plt.xlabel("Weight")
    plt.ylabel("Neuron response (ms)")
    plt.tight_layout()
    plt.show()

def dt_comparison():
    duration = 100
    inp = [23,24,41,52,58,68,69,70,71,72, 94,95]
    v = []
    u= []
    for dt in [1,0.1]:
        pop = Population((1, RS, 0))
        input = Input(inp)
        pop.add_neuron(input)
        pop.create_synapse(input.ID, 0, w=10, d=1)
        pop.run(duration, dt)
        v.append(pop.neurons["0"].v_hist)
        u.append(pop.neurons["0"].u_hist)


    fig, sub = plt.subplots(3,2)
    sub[0,0].plot(v[0]["t"], v[0]["v"])
    sub[1,0].plot(u[0]["t"], u[0]["u"])
    sub[2,0].eventplot(inp)

    sub[0,1].plot(v[1]["t"], v[1]["v"])
    sub[1,1].plot(u[1]["t"], u[1]["u"])
    sub[2,1].eventplot(inp)

    sub[2,0].set_yticks([])
    sub[0,0].set_ylabel("mV")
    sub[1,0].set_ylabel("U")
    sub[0,0].set_title("Membrane potential")
    sub[0,1].set_title("Membrane potential")
    sub[1,0].set_title("U-variable")
    sub[1,1].set_title("U-variable")
    sub[2,0].set_title("Input spikes")
    sub[2,1].set_title("Input spikes")
    sub[2,0].set_xlabel("Time (ms)")
    sub[2, 1].set_xlabel("Time (ms)")

    sub[1,0].sharex(sub[0,0])
    sub[2, 0].sharex(sub[0, 0])

    sub[1,1].sharex(sub[0,1])
    sub[2, 1].sharex(sub[0, 1])

    sub[0,1].sharey(sub[0,0])
    sub[1, 1].sharey(sub[1, 0])
    sub[2,1].sharey(sub[2,0])

    plt.tight_layout()
    plt.show()


def single_neuron():
    DURATION = 50
    dt = 0.1
    pop = Population((1, RS))
    input = Input(spike_times=[20])
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, w=1, d=1)
    pop.run(DURATION, dt=dt)
    print(pop.neurons["0"].v_hist["v"][-1])
    print(pop.neurons["0"].u_hist["u"][-1])


    fig, (sub2,sub3, sub4) = plt.subplots(3,1, sharex=True)
    #sub1.eventplot(pop.neurons["0"].spikes)
    #sub1.set_ylabel("Neuron ID")
    #sub1.set_xlim([0,DURATION])
    #sub1.set_yticks([1])
    #sub1.set_title("Neuron spikes")
    sub2.plot(pop.neurons["0"].v_hist["t"], pop.neurons["0"].v_hist["v"])
    sub2.set_xlim([0,DURATION])
    #sub2.set_ylim([-120,50])
    sub2.set_ylabel("mV")
    sub2.set_title("Membrane potential")

    sub3.plot(pop.neurons["0"].u_hist["t"], pop.neurons["0"].u_hist["u"])
    sub3.set_xlim([0,DURATION])
    sub3.set_title("U-variable")
    #sub3.set_xlabel("Time")
    #sub3.set_yticks(range(int(min(pop.neurons["0"].u_hist)), int(max(pop.neurons["0"].u_hist))))
    sub3.set_ylabel("u")
    events = sorted(pop.neurons[input.ID].spikes)
    sub4.eventplot(events)
    sub4.set_xlim([0,DURATION])
    sub4.set_yticks([1])
    sub4.set_title("Input spikes")
    sub4.set_xlabel("Time (ms)")
    sub4.set_xticks(range(0,DURATION,int(DURATION/50)))
    fig.suptitle(f"dt={dt}ms", size=35)
    #plt.savefig(f"output/dt/" + ("01" if dt == 0.1 else "1"), dpi=200)
    plt.show()

def poly():
    pattern1 = [[],[5,19],[0,11],[19],[7,19],[12,25]]
    pattern2 = [[],[5,13],[11],[0,18],[],[6]]
    pattern3 = [[],[1,9],[8],[5],[0],[11]]
    pattern4 = [[],[10],[16],[5],[0],[1,11]]
    pattern5 = [[],[0],[6],[13],[3,13],[9,20]]
    pattern6 = [[],[4],[2,11],[0,9],[11],[7]]
    pattern7 = [[],[5],[3],[1,10],[0,10],[6,18]]
    pattern8 = [[],[6],[4],[2,10],[4],[0,9]]
    pattern9 = [[],[0,8],[6],[4,13],[15],[11]]
    pattern10 = [[],[3,11],[0,9],[7],[],[13]]
    pattern11 = [[],[12,23],[0,10,21],[7,17,29],[7,17,29],[3,13,24]]
    pattern12 = [[],[16,27],[4,14,25],[11,22],[0,11,22],[3,17,28]]
    pattern13 = [[],[5],[3,12],[10],[],[0]]
    pattern14 = [[],[],[4],[11],[1,11],[0,6,18]]
    patterns = [pattern1,pattern2,pattern3,pattern4,pattern5,pattern6,pattern7,pattern8,pattern9,pattern10,pattern11,pattern12,pattern13,pattern14]
    input_patterns = {"1":[(0,5),(1,0)], "2":[(0,5),(2,0)], "3":[(0,1),(3,0)], "4":[(3,0),(4,1)], "5":[(0,0),(3,3)], "6":[(1,2),(2,0)],"7":[(2,1),(3,0)],"8":[(2,2),(4,0)],"9":[(0,0),(2,4)],"10":[(0,3),(1,0)],"11":[(1,0),(4,3)],"12":[(3,0),(4,3)],"13":[(1,3),(4,0)],"14":[(3,1),(4,0)]}
    for input_pattern, pattern in zip(input_patterns, patterns):
        val = input_patterns[input_pattern]
        DURATION = 30
        pop = Population((5, RS, 0))
        wc = 16
        dc = 0
        pop.create_synapse(0,1,d=6+dc, w=wc)
        pop.create_synapse(0,2,d=4+dc, w=wc)
        pop.create_synapse(0,3,d=2+dc, w=wc)
        pop.create_synapse(0,4,d=2+dc, w=wc)

        pop.create_synapse(1,0,d=2+dc, w=wc)
        pop.create_synapse(1,2,d=7+dc, w=wc)
        pop.create_synapse(1,3,d=7+dc, w=wc)
        pop.create_synapse(1,4,d=2+dc, w=wc)

        pop.create_synapse(2,0,d=4+dc, w=wc)
        pop.create_synapse(2,1,d=2+dc, w=wc)
        pop.create_synapse(2,3,d=2+dc, w=wc)
        pop.create_synapse(2,4,d=7+dc, w=wc)

        pop.create_synapse(3,0,d=10+dc, w=wc)
        pop.create_synapse(3,1,d=3+dc, w=wc)
        pop.create_synapse(3,2,d=5+dc, w=wc)
        pop.create_synapse(3,4,d=6+dc, w=wc)

        pop.create_synapse(4,0,d=5+dc, w=wc)
        pop.create_synapse(4,1,d=4+dc, w=wc)
        pop.create_synapse(4,2,d=4+dc, w=wc)
        pop.create_synapse(4,3,d=4+dc, w=wc)

        input = Input(spike_times=[val[0][1]])
        input2 = Input(spike_times=[val[1][1]])
        pop.add_neuron(input)
        pop.add_neuron(input2)
        pop.create_synapse(input.ID, str(val[0][0]), w=200, d=0)
        pop.create_synapse(input2.ID, str(val[1][0]), w=200, d=0)

        pop.run(DURATION, dt=1)



        fig, (sub1) = plt.subplots(1, 1,sharex=True)
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        spikes.insert(0,[])
        sub1.eventplot(spikes, colors="black", linewidths=1)
        sub1.set_xlim([-1, DURATION])
        sub1.set_ylim([0.5,len(pop.neurons)-1.5])
        sub1.set_yticks(range(1, 6))

        sub1.set_ylabel("Neuron ID")
        sub1.set_title(f"Polychronous group: {input_pattern}")

        #sub2.eventplot(pattern, colors="black", linewidths=1)
        #sub2.set_xlim([-1, DURATION])
        #sub2.set_ylim([0.5,len(pop.neurons)-1.5])
        #sub2.set_yticks(range(1, 6))
        #sub2.set_xticks(range(0, DURATION + 1, 2))
        #sub2.set_ylabel("Neuron ID")
        #sub2.set_xlabel("Time (ms)")
        #sub2.set_title("Reference spikes")

        #fig.suptitle(f"Polychronous group: {input_pattern}")
        plt.savefig(f"output/polychronous_groups/{input_pattern}.png")
        #plt.show()
        plt.clf()

        if input_pattern == "3":
            fig, (sub1) = plt.subplots(1,1)
            sub1.plot(pop.neurons["4"].v_hist["t"],pop.neurons["4"].v_hist["v"])
            sub1.set_xticks(range(30))
            sub1.set_ylabel("Membrane potential (mV)")
            sub1.set_xlabel("Time (ms)")
            sub1.set_title("Membrane potential for neuron 5")
            plt.tight_layout()
            plt.show()
            plt.clf()

            fig, sub = plt.subplots(1)
            spikes = []
            [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]

            spikes.insert(0, [])
            sub.eventplot(spikes, colors="black", linewidths=1)
            sub.set_xlim([-1, DURATION])
            sub.set_ylim([0.5, len(pop.neurons) - 1.5])
            sub.set_yticks(range(1, 6))

            sub.set_ylabel("Neuron ID")
            sub.set_xlabel("Time (ms)")
            sub.set_title("Simulated spikes")
            plt.tight_layout()
            plt.show()
            plt.clf()


def random():
    input_patterns = {}
    for i in range(1,15):
        ran_pattern = [(np.random.randint(0,5),np.random.randint(0,6)),(np.random.randint(0,5),np.random.randint(0,6))]
        while ran_pattern[0][0] == ran_pattern[1][0]:
            ran_pattern = [(np.random.randint(0, 5), np.random.randint(0, 6)),
                       (np.random.randint(0, 5), np.random.randint(0, 6))]
        input_patterns[str(i)] = ran_pattern
    for input_pattern in input_patterns:
        val = input_patterns[input_pattern]
        DURATION = 30
        start = time.time()
        pop = Population((5, RS))
        wc = 16

        pop.create_synapse(0,1,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(0,2,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(0,3,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(0,4,d=np.random.randint(1,11), w=wc)

        pop.create_synapse(1,0,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(1,2,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(1,3,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(1,4,d=np.random.randint(1,11), w=wc)

        pop.create_synapse(2,0,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(2,1,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(2,3,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(2,4,d=np.random.randint(1,11), w=wc)

        pop.create_synapse(3,0,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(3,1,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(3,2,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(3,4,d=np.random.randint(1,11), w=wc)

        pop.create_synapse(4,0,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(4,1,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(4,2,d=np.random.randint(1,11), w=wc)
        pop.create_synapse(4,3,d=np.random.randint(1,11), w=wc)

        input = Input(spike_times=[val[0][1]])
        input2 = Input(spike_times=[val[1][1]])
        pop.add_neuron(input)
        pop.add_neuron(input2)
        pop.create_synapse(input.ID, str(val[0][0]), d=0,w=100)
        pop.create_synapse(input2.ID, str(val[1][0]), d=0,w=100)

        pop.run(DURATION)

        stop = time.time()
        print(stop-start)
        fig, sub1 = plt.subplots(1, 1)
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        spikes.insert(0, [])
        sub1.eventplot(spikes, colors="black", linewidths=1)
        sub1.set_xlim([-1, DURATION])
        sub1.set_ylim([0.5, len(pop.neurons) - 1.5])
        sub1.set_yticks(range(1, 6))
        #sub1.set_xticks(range(DURATION + 1))
        sub1.set_ylabel("Neuron ID")
        #sub1.set_title(f"Random delays: {input_pattern}")
        plt.tight_layout()
        plt.savefig(f"output/random/{input_pattern}")
        # plt.show()
        plt.clf()
        pop.show_network(input_pattern)


def delay_learning():
    COLORS = ["g", "r", "b","y", "m"]
    DURATION = 20
    pop = Population((1, RS))
    input1 = Input(spike_times=[5])
    input2 = Input(spike_times=[6])
    input3 = Input(spike_times=[10])
    #input4 = Input(spike_times=[9])
    pop.add_neuron(input1)
    pop.add_neuron(input2)
    pop.add_neuron(input3)
    #pop.add_neuron(input4)
    pop.create_synapse(input1.ID, "0", w=9, d=5.1,trainable=True)
    pop.create_synapse(input2.ID, "0", w=9, d=4.1, trainable=True)
    pop.create_synapse(input3.ID, "0", w=9, d=0.1, trainable=True)
    #pop.create_synapse(input4.ID, "0", w=9, d=7, trainable=True)
    pop.run(DURATION, dt=0.1)
    for syn in pop.synapses:
        print(syn.d_hist)
    fig, (sub1, sub2) = plt.subplots(2, 1, sharex=True)
    spikes = []
    [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
    sub1.eventplot(spikes, linewidths=1, colors="black")
    #sub1.set_xlim([-1, DURATION])
    #sub1.set_ylim([0.5, len(pop.neurons) - 1.5])
    sub1.set_yticks(range(len(pop.neurons)))
    sub1.set_yticklabels(["RS neuron","Input 1", "Input 2", "Input 3"])
    #sub1.set_xticks(range(DURATION + 1))

    sub1.set_title("Spikes")
    arrival_t = []
    arrival_t.append([])
    for syn in pop.synapses:
        spikes = [t+syn.d_hist["d"][0] for t in pop.neurons[str(syn.i)].spike_times]
        arrival_t.append(spikes)
    sub1.eventplot(arrival_t, colors=COLORS[:len(pop.neurons)])
    i = 1
    for syn in pop.synapses:
        sub2.plot(syn.d_hist["t"], syn.d_hist["d"], color=COLORS[i])
        i += 1
    sub2.set_title("Delay")
    sub2.set_yticks(range(0, 10))
    sub2.set_ylabel("ms")
    #sub3.set_ylabel("mV")
    sub2.set_xticks(range(0,21,1))
    #sub3.set_title("Membrane potential")
    #sub3.plot(pop.neurons["0"].v_hist["t"], pop.neurons["0"].v_hist["v"], color=COLORS[0])
    plt.tight_layout()
    plt.show()


def spike_shift(L, D, W):
    DURATION = 50
    dt = 0.1
    pop = Population((1, RS))
    spike_times1 = [21.0]
    spike_times2 = [21.0 + D]
    input = Input(spike_times1)
    input2 = Input(spike_times2)
    pop.add_neuron(input)
    pop.add_neuron(input2)
    pop.create_synapse(input.ID, 0, w=W, d=1)
    pop.create_synapse(input2.ID, 0, w=W, d=1)
    pop.run(DURATION, dt=dt)
    L.append((max(pop.neurons["0"].v_hist["v"]), D, W))

def spike_shift_sensitivity(d_start = 0, d_end = 15, w_start = 8, w_end = 18, d_step = 0.1, w_step = 0.1):
    if __name__ == '__main__':
        m = mp.Manager()
        L = m.list()
        d_start = d_start
        d_end = d_end
        w_start = w_start
        w_end = w_end
        d_step = d_step
        w_step = w_step
        li = list(itertools.product(np.arange(d_start, d_end, d_step), np.arange(w_start, w_end, w_step)))
        li = [(L, round(x[0], 2), round(x[1], 2)) for x in li]
        with mp.Pool(os.cpu_count() - 1) as p:
            p.starmap(spike_shift, li)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        X = [round(x[2], 3) for x in L]
        Y = [round(x[1], 3) for x in L]
        Z = []
        for z in [x[0] for x in L]:
            if z < 30:
                Z.append("red")
            else:
                Z.append("blue")
        ax.scatter(X, Y, c=Z, s=0.5)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Shift (ms)")
        ax.set_yticks(np.arange(d_start, d_end + 1, 1))
        ax.set_xticks(np.arange(w_start, w_end + 1, 1))
        plt.tight_layout()
        plt.show()


def weight_shift_response(weight, shift_start, shift_end):
    DURATION = 50
    dt = 1
    response_delay = []
    for shift in range(shift_start, shift_end):
        pop = Population((1, RS))
        spike_times1 = [21.0]
        spike_times2 = [21.0 + shift]
        input = Input(spike_times1)
        input2 = Input(spike_times2)
        pop.add_neuron(input)
        pop.add_neuron(input2)
        pop.create_synapse(input.ID, 0, w=weight, d=1)
        pop.create_synapse(input2.ID, 0, w=weight, d=1)
        pop.run(DURATION, dt=dt)
        st = pop.neurons["0"].spikes[0]
        response_delay.append(round(st-(spike_times2[0] + 1), 2))
    return response_delay
    '''
    plt.plot(range(shift_start, shift_end), response_delay)
    plt.xlabel("Shift (ms)")
    plt.ylabel("Neuron response (ms)")
    plt.xticks(range(shift_start, shift_end, 1))
    plt.yticks(range(int(min(response_delay)),int(max(response_delay) + 1),1))
    plt.title(f"Shift respone for weight {weight}")
    plt.tight_layout()
    plt.show()
    '''


def spike_effect_duration(weight):
    DURATION = 50
    dt = 1
    pop = Population((1, RS))
    spike_times1 = [21.0]
    input = Input(spike_times1)
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, w=weight, d=1)
    pop.run(DURATION, dt=dt)
    plt.plot(pop.neurons["0"].v_hist["t"], pop.neurons["0"].v_hist["v"])
    plt.vlines(spike_times1[0]+1, ymin=-75,ymax=-55, colors="red")
    plt.vlines(30, ymin=-75, ymax=-55, colors="red")
    plt.hlines(-75, xmin=spike_times1[0]+1, xmax=30, colors="green")
    plt.show()


def double_update_comparison():
    DURATION = 100
    dt = 1
    pop = Population((1, RS))
    input = Input(spike_times=[4,10, 23,24, 38, 45, 47, 76])
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, w=80, d=1)
    pop.run(DURATION, dt=dt)

    fig, (sub1, sub2,sub3) = plt.subplots(3,1, sharex=True)
    sub1.plot(pop.neurons["0"].v_hist["t"], pop.neurons["0"].v_hist["v"])
    sub1.set_xlim([0,DURATION])
    #sub2.set_ylim([-120,50])
    sub1.set_ylabel("mV")
    sub1.set_title("Membrane potential")
    sub1.set_yticks(np.linspace(-100,30,3))

    sub2.plot(pop.neurons["0"].u_hist["t"], pop.neurons["0"].u_hist["u"])
    sub2.set_xlim([0,DURATION])
    sub2.set_title("U-variable")
    #sub2.set_yticks(np.arange(np.floor(min(pop.neurons["0"].u_hist["u"])), np.ceil(max(pop.neurons["0"].u_hist["u"]))))
    sub2.set_ylabel("u")
    events = sorted(pop.neurons[input.ID].spikes)
    sub3.eventplot(events)
    sub3.set_xlim([0,DURATION])
    sub3.set_yticks([])
    sub3.set_title("Input spikes")
    sub3.set_xlabel("Time (ms)")
    sub3.set_xticks(np.linspace(0,DURATION,6))
    plt.tight_layout()
    plt.savefig(f"output/double_update.png")
    plt.show()

def neuron_refractory():
    pop = Population((1,RS))
    st = [5,33]
    input1 = Input(spike_times=st)
    input2 = Input(spike_times=st)
    pop.add_neuron(input1)
    pop.add_neuron(input2)
    pop.create_synapse(input1.ID, "0", w=16, d=1)
    pop.create_synapse(input2.ID, "0", w=16, d=1)
    pop.run(50, dt=1)
    print(pop.neurons["0"].spikes)


def random_large_network():
    inp = [[[5, 4, 8],[7, 1, 8]],[[9, 4, 8],[3, 4, 3]],[[3, 8, 5],[6, 7, 9]]]
    iid = [[34,12],[30,24], [34,23]]
    #for ip, id in zip(inp, iid):
    fig, (sub1,sub2,sub3) = plt.subplots(3,1, sharex=True)
    subs = [sub1,sub2,sub3]
    i = 603
    while True:
        pop = Population((100, RS))
        np.random.seed(1)
        pop.create_random_connections(p=0.1, d=list(range(1,11)), w=16)
        np.random.seed(i)
        input1 = Input(spike_times=sorted([np.random.randint(1,10)]))
        input2 = Input(spike_times=sorted([np.random.randint(1,10)]))
        #input1 = Input(spike_times=ip[0])
        #input2 = Input(spike_times=ip[1])
        choice1 = np.random.choice(list(pop.neurons),10, replace=False)
        choice2 = np.random.choice(list(pop.neurons),10, replace=False)
        #choice1 = id[0]
        #choice2 = id[1]
        pop.add_neuron(input1)
        pop.add_neuron(input2)

        for con1, con2 in zip(choice1, choice2):
            pop.create_synapse(input1.ID, con1)
            pop.create_synapse(input2.ID, con2)
        pop.run(500)
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        n_spikes = 0
        for x in spikes:
            n_spikes += len(x)
        if n_spikes > 20:
            try:
                s = subs.pop()
            except:
                break
            s.eventplot(spikes, colors="black")
            s.set_title(f"ID {choice1}: {input1.spike_times[0]}ms\nID {choice2}: {input2.spike_times[0]}ms")
            print(i)
            print(input1.spike_times)
            print(input2.spike_times)
            print(choice1)
            print(choice2)
            if not subs:
                break
        i += 1
    sub3.set_xlabel("Time (ms)")
    sub1.set_ylabel("Neuron ID")
    sub2.set_ylabel("Neuron ID")
    sub3.set_ylabel("Neuron ID")
    plt.tight_layout()
    plt.show()
    #plt.clf()
    #pop.show_network(show=True)

def spike_shift_sensitivity_mt():
    if __name__ == '__main__':
        m = mp.Manager()
        L = m.list()
        d_start = 0
        d_end = 15
        w_start = 8
        w_end = 18
        d_step = 0.1
        w_step = 0.1
        l = list(itertools.product(np.arange(d_start,d_end,d_step), np.arange(w_start, w_end, w_step)))
        l = [(L,round(x[0],2), round(x[1],2)) for x in l]
        with mp.Pool(os.cpu_count() - 1) as p:
            p.starmap(spike_shift_sensitivity, l)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        X = [round(x[2],3) for x in L]
        Y = [round(x[1],3) for x in L]
        Z = []
        for z in [x[0] for x in L]:
            if z < 30:
                Z.append("red")
            else:
                Z.append("blue")

        ax.scatter(X, Y, c=Z, s=0.5)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Shift (ms)")
        ax.set_yticks(np.arange(d_start,d_end + 1, 1))
        ax.set_xticks(np.arange(w_start, w_end + 1, 1))
        plt.tight_layout()
        plt.show()


def run_xnxi_alt(t, n, i, l_pattern, l_interm, delay_seed, input_seed):
    delay_rng = np.random.default_rng(delay_seed)
    input_rng = np.random.default_rng(input_seed)
    pop = Population((n, RS))
    pop.create_feed_forward_connections(d=[])
    for x in range(i):
        offset1 = input_rng.integers(0, 6)
        period1 = input_rng.integers(10,20)
        pattern1 = [(offset1 + (period1 * rep)) if (offset1 + (period1 * rep) < l_pattern) else None for rep in
                   range(l_pattern)]
        offset2 = input_rng.integers(0, 6)
        period2 = input_rng.integers(10,20)
        pattern2 = [(offset2 + (period1 * rep)) if (offset2 + (period2 * rep) < l_pattern) else None for rep in
                   range(l_pattern)]
        pattern = []
        flip = 1
        for rnd in range(int(t/(l_pattern + l_interm))):
            if rnd % 2 == 0:
                [pattern.append(inp +(l_pattern + l_interm) * rnd) for inp in (pattern1 if flip == 1 else pattern2)]
                flip *= -1
        inp = pop.create_input(pattern)
        for j in pop.neurons[:int(np.ceil(np.sqrt(len(pop.neurons))))]:
            pop.create_synapse(inp.ID, j, w=16, d=delay_rng.integers(1, 60) / 10)

    pop.structure = "grid"
    pop.run(t, dt=0.1, plot_network=False)
    pop.plot_delays()
    pop.plot_raster()
    pop.plot_membrane_potential()
    pop.show_network(save=True)


def run_xnxi_async(t, n, i, delay_seed, input_seed):
    delay_rng = np.random.default_rng(delay_seed)
    input_rng = np.random.default_rng(input_seed)

    pop = Population((n, RS))

    for x in range(i):
        xi = [x for x in range(t) if i % input_rng.integers(10,30) == 0]
        inp = pop.create_input(xi)
        for y in range(int(np.ceil(np.sqrt(n)))):
            pop.create_synapse(inp.ID, y, w=16, d=delay_rng.integers(1, 60) / 10)

    pop.structure = "grid"
    pop.run(t, dt=0.1, plot_network=False)
    pop.plot_delays()
    pop.plot_raster()
    pop.plot_membrane_potential()
    pop.show_network(save=True)

