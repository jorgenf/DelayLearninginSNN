import matplotlib.pyplot as plt

from Population import *
import time

COLORS = ["red", "blue", "green", "indigo", "royalblue", "peru", "palegreen", "yellow"]
COLORS += [(np.random.random(), np.random.random(), np.random.random()) for x in range(20)]

def Single_Neuron():
    DURATION = 100
    start = time.time()
    pop = Population((1, RS))
    input = Input(spike_times=[2.0,10.0,11.0,12.0, 22.0,21.0, 34.0,35.0, 66.0, 78.0])
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, w=15, d=1)
    pop.run(100)
    stop = time.time()
    print(f"\n{stop-start}")

    fig, (sub1, sub2,sub3, sub4) = plt.subplots(4,1,figsize=(25,15))
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
    sub3.set_ylabel("u")
    events = sorted(pop.neurons[input.ID].spikes)
    sub4.eventplot(events)
    sub4.set_xlim([0,DURATION])
    sub4.set_yticks([1])
    sub4.set_title("Input spikes")
    sub4.set_xlabel("Time (ms)")
    fig.suptitle("Jørgen2")
    plt.savefig("output/single_neuron/jørgen2", dpi=200)
    #plt.show()

def poly():
    patterns = {"1":[(0,5),(1,0)], "2":[(0,5),(2,0)], "3":[(0,1),(2,0)], "4":[(3,0),(4,1)], "5":[(0,0),(3,3)], "6":[(1,2),(2,0)],"7":[(2,1),(3,0)],"8":[(2,2),(4,0)],"9":[(0,4),(1,0)],"10":[(0,3),(1,0)],"11":[(1,0),(4,3)],"12":[(1,4),(3,0)],"13":[(1,3),(4,0)],"14":[(3,2),(4,0)]}
    for pattern in patterns:
        val = patterns[pattern]
        DURATION = 30
        start = time.time()
        pop = Population((5, POLY))
        wc = 20
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

        fig, sub1 = plt.subplots(1, 1,figsize=(25,15))
        spikes = []
        [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]
        spikes.insert(0,[])
        sub1.eventplot(spikes,colors=COLORS[:len(spikes)], linewidths=3)
        sub1.set_xlim([-1, DURATION])
        sub1.set_ylim([0.5,len(pop.neurons)-1.5])
        sub1.set_yticks(range(1, 6))
        sub1.set_xticks(range(DURATION + 1))
        sub1.set_ylabel("Neuron ID")
        sub1.set_title("Spikes")
        fig.suptitle(f"Polychronous group: {pattern}")
        plt.savefig(f"output/polychronous_groups/{pattern}", dpi=200)
        plt.clf()
        #plt.show()

poly()