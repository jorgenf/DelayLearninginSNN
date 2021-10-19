import matplotlib.pyplot as plt
import matplotlib
from Population import *
import time
import numpy as np

COLORS = ["red", "blue", "green", "indigo", "royalblue", "peru", "palegreen", "yellow"]
COLORS += [(np.random.random(), np.random.random(), np.random.random()) for x in range(20)]

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

def Single_Neuron():
    DURATION = 1000
    start = time.time()
    pop = Population((1, RS))
    #input = Input(spike_times=[2.0,10.0,11.0,12.0, 22.0,21.0, 34.0,35.0, 66.0, 78.0])
    input = Input(spike_times=[])
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, w=15, d=1)
    pop.run(DURATION)
    stop = time.time()
    print(f"\n{stop-start}")

    fig, (sub2,sub3) = plt.subplots(2,1,figsize=(25,15))
    #sub1.eventplot(pop.neurons["0"].spikes)
    #sub1.set_ylabel("Neuron ID")
    #sub1.set_xlim([0,DURATION])
    #sub1.set_yticks([1])
    #sub1.set_title("Neuron spikes")
    sub2.plot(pop.neurons["0"].v_hist)
    sub2.set_xlim([0,DURATION])
    sub2.set_ylim([-100,50])
    sub2.set_ylabel("mV")
    sub2.set_title("Membrane potential")
    sub2.set_xticks(range(0,DURATION,2))
    sub3.plot(pop.neurons["0"].u_hist)
    sub3.set_xlim([0,DURATION])
    sub3.set_title("U-variable")
    #sub3.set_yticks(range(int(min(pop.neurons["0"].u_hist)), int(max(pop.neurons["0"].u_hist))))
    sub3.set_ylabel("u")
    events = sorted(pop.neurons[input.ID].spikes)
    #sub4.eventplot(events)
    #sub4.set_xlim([0,DURATION])
    #sub4.set_yticks([1])
    #sub4.set_title("Input spikes")
    #sub4.set_xlabel("Time (ms)")
    fig.suptitle("Jørgen2")
    #plt.savefig("output/single_neuron/jørgen2", dpi=200)
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
    input_patterns = {"1":[(0,5),(1,0)], "2":[(0,5),(2,0)], "3":[(0,1),(3,0)], "4":[(3,0),(4,1)], "5":[(0,0),(3,3)], "6":[(1,2),(2,0)],"7":[(2,1),(3,0)],"8":[(2,2),(4,0)],"9":[(0,0),(2,4)],"10":[(0,3),(1,0)],"11":[(1,0),(4,3)],"12":[(1,4),(3,0)],"13":[(1,3),(4,0)],"14":[(3,2),(4,0)]}
    for input_pattern, pattern in zip(input_patterns, patterns):
        val = input_patterns[input_pattern]
        DURATION = 30
        start = time.time()
        pop = Population((5, RS))
        wc = 22
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
        pop.create_synapse(input.ID, str(val[0][0]), w=100)
        pop.create_synapse(input2.ID, str(val[1][0]), w=100)

        pop.run(DURATION)

        stop = time.time()
        print(stop-start)

        fig, (sub1,sub2) = plt.subplots(2, 1,figsize=(15,15))
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        spikes.insert(0,[])
        sub1.eventplot(spikes, colors="black", linewidths=1)
        sub1.set_xlim([-1, DURATION])
        sub1.set_ylim([0.5,len(pop.neurons)-1.5])
        sub1.set_yticks(range(1, 6))
        sub1.set_xticks(range(DURATION + 1))
        sub1.set_ylabel("Neuron ID")
        sub1.set_title("Model spikes")

        sub2.eventplot(pattern, colors="black", linewidths=1)
        sub2.set_xlim([-1, DURATION])
        sub2.set_ylim([0.5,len(pop.neurons)-1.5])
        sub2.set_yticks(range(1, 6))
        sub2.set_xticks(range(DURATION + 1))
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

random()