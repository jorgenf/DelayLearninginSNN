from Population import *
import time



def something():
    DURATION = 100
    start = time.time()
    pop = Population((1, RS))
    input = Input(spike_times=[2, 3, 20, 21, 29, 81, 87, 88, 89])
    pop.add_neuron(input)
    pop.create_synapse(input.ID, 0, d=10)
    pop.run(100)
    stop = time.time()
    print(f"\n{stop-start}")
    print(pop.neurons["0"].v_hist)

    fig, (sub1, sub2,sub3) = plt.subplots(3,1)
    sub1.plot(pop.neurons["0"].v_hist)
    sub1.set_xlim([0,DURATION])
    sub1.set_xticks(range(DURATION + 1))
    sub1.set_ylabel("mV")
    sub1.set_title("Membrane potential")
    sub2.plot(pop.neurons["0"].u_hist)
    sub2.set_xticks(range(DURATION + 1))
    sub2.set_xlim([0,DURATION])
    sub2.set_title("U-variable")
    sub2.set_ylabel("u")
    events = sorted(pop.neurons[input.ID].spikes)
    sub3.eventplot(events)
    sub3.set_xlim([0,DURATION])
    sub3.set_xticks(range(DURATION + 1))
    sub3.set_title("Input spikes")
    sub3.set_xlabel("Time (ms)")
    print(pop.neurons[input.ID].spikes)
    plt.show()

def poly():
    DURATION = 30
    start = time.time()
    pop = Population((5, RS))

    wc = 10
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

    input = Input(spike_times=[0])
    input2 = Input(spike_times=[5])
    pop.add_neuron(input)
    pop.add_neuron(input2)

    pop.create_synapse(input.ID, "1", w=100)
    pop.create_synapse(input2.ID, "0", w=100)

    pop.run(DURATION)

    stop = time.time()
    print(stop-start)

    fig, sub1 = plt.subplots(1, 1)
    spikes = []
    [spikes.append(pop.neurons[n].spikes) for n in pop.neurons]

    sub1.eventplot(spikes)
    sub1.set_xlim([-1, DURATION])
    sub1.set_ylim([-0.5,len(pop.neurons)-0.5])
    sub1.set_xticks(range(DURATION + 1))
    sub1.set_ylabel("Neuron ID")
    sub1.set_title("Spikes")

    plt.show()

poly()