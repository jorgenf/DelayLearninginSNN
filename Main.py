import gzip

import Data
import Population
from Population import *
import numpy as np
import multiprocessing as mp
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img = [10]
layers = [2]
train_inst = [20]
train_digits = [[0,1]]
test_inst = [25]
test_digits = [[0,1]]
w = [6]
th = [0.7]
p = [0.1]
par = [True]
seed = list(range(500))

params = [img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, par, seed]
combos = list(itertools.product(*params))
combos = combos[:250]



def run_training_phase(img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, par, seed):
    save_delays = False
    interval = 200
    rng = np.random.default_rng(seed)
    name = f"TrainingPhase_img-{img}_layers-{layers}_train_inst-{train_inst}_traindigits-{train_digits}_test_inst-{test_inst}_testdigits-{test_digits}_w-{w}_th-{th}_p-{p}_par-{par}_seed-{seed}"

    test = [y for y in test_digits for x in range(test_inst)]
    train = [rng.choice(train_digits) for x in range(train_inst)]
    pattern = test + train + test
    counter = 0
    sequence = [test_inst for x in range(len(test_digits))] + [train_inst] + [test_inst for x in
                                                                              range(len(test_digits))]
    breaks = []
    for step in sequence:
        breaks.append(counter + step)
        counter += step
    input = Data.create_mnist_sequence_input(pattern, interval, breaks, img)

    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=False, seed=seed)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if rng.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[rng.integers(1,40) for x in range(len(ij))], trainable=False)


    training_change = [test_inst*len(test_digits), test_inst*len(test_digits) + train_inst]
    durations = []

    for i, l in enumerate(list(map(list, zip(*input)))):
        if i in training_change:
            durations.append(min(l)-1)

    pop.run(durations[0], path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False,
            save_neuron_data=True)
    for syn in pop.synapses:
        syn.trainable = True
    pop.run(durations[1] - durations[0], path='network_plots/', name=name, record_PG=True,
            save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False,
            save_neuron_data=True)
    for syn in pop.synapses:
        syn.trainable = False
    pop.run(max([max(x) for x in input]) + interval - durations[1], path='network_plots/', name=name, record_PG=True, save_post_model=True, PG_duration=100, PG_match_th=th, save_delays=save_delays, save_synapse_data=False, save_neuron_data=True)



def run_0_8_0(img, layers, num, inst, w, th, p, par, train, seed):
    name = f"MNIST_img-{img}_layers-{layers}_num-{num}_inst-{inst}_w-{w}_th-{th}_p-{p}_par-{par}_train-{train}"
    input = Data.create_mnist_input(inst, num, 200, image_size=img)
    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=train)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if np.random.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[np.random.randint(1,40) for x in range(len(ij))], trainable=train)
    pop.run(max([max(inp) for inp in input]) + 100, path='network_plots/', name=name, record_PG=True, save_post_model=False, PG_duration=100, PG_match_th=th, save_delays=False, save_synapse_data=False, save_neuron_data=True)


if __name__ == '__main__':
    with mp.Pool(30) as p:
        p.starmap(run_training_phase, combos)





#file1 =  json.load(open(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-5_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-1\neuron_data.json"))#

#file2 =  json.load(open(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-5_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-1_1\neuron_data.json"))

#for k1, k2 in zip(list(file1.keys())[-25:], list(file2.keys())[-25:]):
#    if file1[k1] != file2[k2]:
#        for kk1, kk2 in zip(file1[k1], file2[k2]):
#            if file1[k1][kk1] != file2[k2][kk2]:
#                print(k1, k2)
#                print(file1[k1][kk1])
#                print(file2[k2][kk2])


#Data.compile_results("G:/introducing unseen digit in testing", 'unseen_digit_test', 3, 25, 20)

#run_training_phase(img=4, layers=3, train_inst=20, train_digits=[0,1], test_inst=25, test_digits=[0,1,2], w=6, th=0.7, p=0.3, par=True, seed=1)
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


#import re
#ddir = "G:/multiple runs of 12by12 th0.9"

#Data.compile_results(r"G:\multiple runs of 10by10 th0.9", '10by10th0.9')
#Data.compile_results(r"G:\multiple runs of 12by12 th0.8", '12by12th0.8')
#Data.compile_results(r"G:\multiple runs of 12by12 th0.7", '12by12th0.7')
#Data.compile_results(r"G:\multiple runs of 10by10 th0.9", '10by10th0.9')
#Data.compile_results(r"G:\multiple runs of 10by10 th0.8", '10by10th0.8')
#Data.compile_results(r"G:\multiple runs of 10by10 th0.7", '10by10th0.7')

'''
ddir = "G:/introducing unseen digit in testing th0.5"
param = []
for dir in os.listdir(ddir):
    param.append((ddir, dir))
def change_threshold(ddir, dir):
    print(dir)
    m = Data.load_model(os.path.join(os.path.join(ddir,dir),"post_sim_model.pkl"))
    m.dir = os.path.join(ddir, dir)
    m.build_pgs(min_threshold=0.5)
    m.save_PG_data()
    m.plot_raster()
    Data.save_model(m, os.path.join(ddir,os.path.join(dir, "post_sim_model.pkl")))
    os.rename(os.path.join(ddir,dir), os.path.join(ddir, dir.replace('th-0.6', 'th-0.5')))
'''
#if __name__ == '__main__':
#
#   with mp.Pool(30) as p:
#        p.starmap(change_threshold, param)


#Data.compile_results("G:/introducing unseen digit in testing th0.5", 'unseen_digit_test_th0.5', 3, 25, 20)


#Data.compile_results(r"G:\multiple runs of 10by10 th0.8")