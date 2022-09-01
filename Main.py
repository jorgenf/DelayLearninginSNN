import gzip

import Data
import Population
from Population import *
import numpy as np
import multiprocessing as mp
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img = [10, 12, 15]
layers = [1,2,3]
train_inst = [20]
test_inst = [10]
w = [4, 6, 8]
th = [0.7]
p = [0.05, 0.1, 0.2]
par = [True]
seed = [1]

params = [img, layers, train_inst, test_inst, w, th, p, par, seed]
combos = list(itertools.product(*params))
combos = combos[:int(len(combos)/2)]
def run_training_phase(img, layers, train_inst, test_inst, w, th, p, par, seed):
    interval = 200
    rng = np.random.default_rng(seed)
    name = f"TrainingPhase_img-{img}_layers-{layers}_num-{[0,1]}_train_inst-{train_inst}_test_inst-{test_inst}_w-{w}_th-{th}_p-{p}_par-{par}"
    if name in os.listdir("network_plots"):
        if len(os.listdir(os.path.join("network_plots", name))) == 7:
            return None
    input = Data.create_mnist_sequence_input([0 for _ in range(test_inst)] + [1 for _ in range(test_inst)] +
                                             [rng.integers(0, 2) for _ in range(train_inst)] + [0 for _ in range(test_inst)] + [1 for _ in range(test_inst)], interval, [test_inst, 2*test_inst, train_inst + (2*test_inst), train_inst + (3*test_inst), train_inst + (4*test_inst)], img)
    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=False, seed=1)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if rng.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[rng.integers(1,40) for x in range(len(ij))], trainable=False)

    pop.run(21*interval, path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=False, save_synapse_data=False,
            save_neuron_data=True)
    for syn in pop.synapses:
        syn.trainable = True
    pop.run((train_inst + 1) * interval, path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=False, save_synapse_data=False,
            save_neuron_data=True)
    for syn in pop.synapses:
        syn.trainable = False
    pop.run(max([max(x) for x in input]) + interval - (train_inst * interval + 20 * interval), path='network_plots/', name=name, record_PG=True, save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=False, save_synapse_data=False, save_neuron_data=True)



def run_0_8_0(img, layers, num, inst, w, th, p, par, train, seed):
    name = f"MNIST_img-{img}_layers-{layers}_num-{num}_inst-{inst}_w-{w}_th-{th}_p-{p}_par-{par}_train-{train}"
    input = Data.create_mnist_input(inst, num, 200, image_size=img)
    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=train)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if np.random.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[np.random.randint(1,40) for x in range(len(ij))], trainable=train)
    pop.run(max([max(inp) for inp in input]) + 100, path='network_plots/', name=name, record_PG=True, save_post_model=False, PG_duration=100, PG_match_th=th,  save_delays=False, save_synapse_data=False, save_neuron_data=True)




#if __name__ == '__main__':
#    with mp.Pool(31) as p:
#        p.starmap(run_training_phase, combos)

#Data.compile_results("network_plots")

#run_training_phase(2, 1, 2, 2, 16, 0.7, 1, True, 1)
#m = Data.load_model(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-12_layers-2_num-[0, 1]_train_inst-20_test_inst-10_w-6_th-0.7_p-0.05_par-True\post_sim_model.pkl")
#m.build_pgs()
#m.plot_raster()
#m.save_PG_data()

'''
l1 = {'4': {'3':{}}, '5': {}, '6': {'1':{}}, '7': {'1':{'2':{'4':{}}}}}
l2 = {'4': {'1':{}}, '5': {'1':{'2':{}}}, '6': {'1':{}}, '7': {'1':{}}}
l3 = {'4': {'1': {}}, '5': {'1': {'2': {}}}, '6': {'1': {'2':{'4':{}}}}, '7': {'1': {'2':{}}}}
match, unique, canonical = Data.compare_poly(l1, l3)
print(match, unique)
print(match/unique)


match2, unique2, canonical2 = Data.compare_poly_2(l1, l3)
print(match2, unique2)
print(match2/unique2)
print(canonical2)
'''

Data.compile_results("network_plots")