from Population import *
import time
import numpy as np
import multiprocessing as mp
import itertools
import os
from mpl_toolkits import mplot3d

COLORS = ["red", "blue", "green", "indigo", "royalblue", "peru", "palegreen", "yellow"]
COLORS += [(np.random.random(), np.random.random(), np.random.random()) for x in range(20)]

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=25)



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

def spike_sensitivity():
    delays = []
    start = 17
    end = 30
    step = 0.1
    for W in np.arange(start,end,step):
        DURATION = 50
        dt = 0.1
        pop = Population((1, RS))
        input = Input(spike_times=[21.0])
        pop.add_neuron(input)
        pop.create_synapse(input.ID, 0, w=W, d=1)
        pop.run(DURATION, dt=dt)
        delays.append(round(pop.neurons["0"].spikes[0]-(pop.neurons["1"].spikes[0] + 1),3))

    plt.plot(np.arange(start,end, step), delays)
    plt.yticks(np.arange(0,10,0.5))
    plt.xticks(np.arange(start, end,1))
    plt.xlabel("Weight")
    plt.ylabel("Spike delay (ms)")
    #plt.title("Singel neuron", size=35)
    plt.show()

def Single_Neuron():
    DURATION = 100
    dt = 1
    pop = Population((1, RS))
    input = Input(spike_times=[2.0, 10.0, 11.0, 12.0, 22.0, 21.0, 34.0, 35.0, 66.0, 78.0])
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, w=15, d=1)
    pop.run(DURATION, dt=dt)



    fig, (sub2,sub3, sub4) = plt.subplots(3,1,figsize=(25,15), sharex=True)
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
    #sub2.set_xticks(range(0,DURATION,int(DURATION/50)))
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
    fig.suptitle(f"dt={dt}ms", size=35)
    #plt.savefig(f"output/dt/" + ("01" if dt == 0.1 else "1"), dpi=200)
    plt.show()

def poly():
    pattern1 = [[],[5,19],[0,11],[19],[6,19],[12,25]]
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
        pop = Population((5, RS, 3))
        wc = 25
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



        fig, (sub1,sub2) = plt.subplots(2, 1,figsize=(15,15), sharex=True)
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        spikes.insert(0,[])
        sub1.eventplot(spikes, colors="black", linewidths=1)
        sub1.set_xlim([-1, DURATION])
        sub1.set_ylim([0.5,len(pop.neurons)-1.5])
        sub1.set_yticks(range(1, 6))

        sub1.set_ylabel("Neuron ID")
        sub1.set_title("Model spikes")

        sub2.eventplot(pattern, colors="black", linewidths=1)
        sub2.set_xlim([-1, DURATION])
        sub2.set_ylim([0.5,len(pop.neurons)-1.5])
        sub2.set_yticks(range(1, 6))
        sub2.set_xticks(range(0, DURATION + 1, 2))
        sub2.set_ylabel("Neuron ID")
        sub2.set_title("Reference spikes")

        fig.suptitle(f"Polychronous group: {input_pattern}")
        plt.savefig(f"output/polychronous_groups/{input_pattern}", dpi=200)
        #plt.show()
        plt.clf()

def random():
    pattern1 = [[],[5,19],[0,11],[19],[6,19],[12,25]]
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
    input_patterns = {"1":[(0,5),(1,0)], "2":[(0,5),(2,0)], "3":[(0,1),(3,0)], "4":[(3,0),(4,1)], "5":[(0,0),(3,3)], "6":[(1,2),(2,0)],"7":[(2,1),(3,0)],"8":[(2,2),(4,0)],"9":[(0,0),(2,4)],"10":[(0,3),(1,0)],"11":[(1,0),(4,3)],"12":[(1,4),(3,0)],"13":[(1,3),(4,0)],"14":[(3,2),(4,0)]}
    for input_pattern, pattern in zip(input_patterns, patterns):
        val = input_patterns[input_pattern]
        DURATION = 30
        start = time.time()
        pop = Population((5, RS))
        wc = 22

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

        input = Input(spike_times=[val[0][1]])
        input2 = Input(spike_times=[val[1][1]])
        pop.add_neuron(input)
        pop.add_neuron(input2)
        pop.create_synapse(input.ID, str(val[0][0]), w=100)
        pop.create_synapse(input2.ID, str(val[1][0]), w=100)

        pop.run(DURATION)

        stop = time.time()
        print(stop-start)
        fig, (sub1, sub2) = plt.subplots(2, 1, figsize=(15, 15))
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        spikes.insert(0, [])
        sub1.eventplot(spikes, colors="black", linewidths=1)
        sub1.set_xlim([-1, DURATION])
        sub1.set_ylim([0.5, len(pop.neurons) - 1.5])
        sub1.set_yticks(range(1, 6))
        sub1.set_xticks(range(DURATION + 1))
        sub1.set_ylabel("Neuron ID")
        sub1.set_title("Model spikes")

        sub2.eventplot(pattern, colors="black", linewidths=1)
        sub2.set_xlim([-1, DURATION])
        sub2.set_ylim([0.5, len(pop.neurons) - 1.5])
        sub2.set_yticks(range(1, 6))
        sub2.set_xticks(range(DURATION + 1))
        sub2.set_ylabel("Neuron ID")
        sub2.set_title("Reference spikes")

        fig.suptitle(f"Random delays: {input_pattern}")
        plt.savefig(f"output/random/{input_pattern}", dpi=200)
        # plt.show()
        plt.clf()

def delay_learning():
    DURATION = 1000000
    pop = Population((5, RS))
    input1 = Input(p=0.01)
    input2 = Input(p=0.01)
    pop.add_neuron(input1)
    pop.add_neuron(input2)
    pop.create_synapse(input1.ID, "0", w=40)
    pop.create_synapse(input2.ID, "1", w=40)

    wc = 20
    pop.create_synapse(0, 1, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(0, 2, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(0, 3, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(0, 4, d=np.random.randint(1, 20), w=wc, trainable=True)

    pop.create_synapse(1, 0, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(1, 2, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(1, 3, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(1, 4, d=np.random.randint(1, 20), w=wc, trainable=True)

    pop.create_synapse(2, 0, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(2, 1, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(2, 3, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(2, 4, d=np.random.randint(1, 20), w=wc, trainable=True)

    pop.create_synapse(3, 0, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(3, 1, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(3, 2, d=np.random.randint(1, 20), w=wc, trainable=True)
    pop.create_synapse(3, 4, d=np.random.randint(1, 20), w=wc, trainable=True)

    pop.run(DURATION)
    fig, (sub1, sub2, sub3, sub4) = plt.subplots(4, 1, figsize=(15, 15))
    spikes = []
    [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
    sub1.eventplot(spikes, colors="black", linewidths=1)
    #sub1.set_xlim([-1, DURATION])
    #sub1.set_ylim([0.5, len(pop.neurons) - 1.5])
    #sub1.set_yticks(range(len(pop.neurons)))
    #sub1.set_xticks(range(DURATION + 1))
    sub1.set_ylabel("Neuron ID")
    sub1.set_title("Model spikes")
    input = [input1.spikes, input2.spikes]
    sub2.eventplot(input, colors="black", linewidths=1)
    #sub2.set_xlim([-1, DURATION])
    #sub2.set_ylim([0.5, len(pop.neurons) - 1.5])
    #sub2.set_yticks(range(1, 6))
    #sub2.set_xticks(range(DURATION + 1))
    sub2.set_ylabel("Neuron ID")
    sub2.set_title("Input spikes")
    for syn in pop.synapses:
        sub3.plot(syn.d_hist)
    plt.show()

def spike_shift_sensitivity(L, D, W):
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
    '''
    plt.plot(pop.neurons["0"].v_hist["t"], pop.neurons["0"].v_hist["v"])
    plt.vlines(x=spike_times1[0] + 1, ymin=-100, ymax=100, colors="blue")
    plt.vlines(x=spike_times2[0] + 1, ymin=-100, ymax=100, colors="red")
    #plt.yticks(np.arange(0, 10, 0.5))
    plt.xticks(np.arange(0, DURATION, 2))
    plt.xlabel("Membrane potential (mV)")
    plt.ylabel("Time (ms)")
    plt.ylim([-80,40])
    # plt.title("Singel neuron", size=35)
    plt.show()


spike_shift_sensitivity(None, 8.2, 16.2)
'''
if __name__ == '__main__':
    m = mp.Manager()
    L = m.list()
    d_start = 0
    d_end = 15
    w_start = 8
    w_end = 18
    step = 0.1
    list = list(itertools.product(np.arange(d_start,d_end,step), np.arange(w_start, w_end, step)))
    list = [(L,x[0], x[1]) for x in list]
    with mp.Pool(os.cpu_count() - 1) as p:
        p.starmap(spike_shift_sensitivity, list)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    X = [x[2] for x in L]
    Y = [x[1] for x in L]
    print(X)
    print(Y)
    print(Y.index(8.2))
    Z = []
    for z in [x[0] for x in L]:
        if z == 30:
            Z.append("blue")
        else:
            Z.append("red")

    ax.scatter(X, Y, c=Z)
    ax.set_xlabel("W")
    ax.set_ylabel("D")
    ax.set_yticks(np.arange(d_start,d_end + 1, 1))
    ax.set_xticks(np.arange(w_start, w_end + 1, 1))
    plt.show()
