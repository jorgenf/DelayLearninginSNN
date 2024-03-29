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
test_digits = [[0,1,2]]
w = [6]
th = [0.7]
p = [0.1]
par = [True]
seed = list(range(100))

params = [img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, par, seed]
combos = list(itertools.product(*params))
#combos = combos[:250]



def run_training_phase(img, layers, train_inst, train_digits, test_inst, test_digits, w, th, p, par, seed):
    save_delays = False
    interval = 200
    rng = np.random.default_rng(seed)
    static_rng = np.random.default_rng(1)
    name = f"TrainingPhase_img-{img}_layers-{layers}_train_inst-{train_inst}_traindigits-{train_digits}_test_inst-{test_inst}_testdigits-{test_digits}_w-{w}_th-{th}_p-{p}_par-{par}_seed-{seed}"
    test = [y for y in test_digits for x in range(test_inst)]
    train = [static_rng.choice(train_digits) for x in range(train_inst)]
    counter = 0
    sequence = [test_inst for x in range(len(test_digits))] + [train_inst] + [test_inst for x in
                                                                              range(len(test_digits))]
    breaks = []
    for step in sequence:
        breaks.append(counter + step)
        counter += step
    input = Data.create_mnist_sequence_input(train, test, interval, breaks, img)

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


#if __name__ == '__main__':
#    with mp.Pool(30) as p:
#        p.starmap(run_training_phase, combos)


#m2 = json.load(open(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-5_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-1\neuron_data.json"))
#d = Data.load_model(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-5_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-0\post_sim_model.pkl")

#d.plot_raster(duration=(950,1050),legend=False, plot_pg=False)
#d.plot_raster(duration=(20750,20850),legend=False, plot_pg=False)

#m1 = json.load(open(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-10_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-99\neuron_data.json"))
'''
for n in list(m1.keys())[-100:]:
    spike = m1[n]["spikes"]
    for i in range(75):
        #print(spike[i], spike[i+95])
        if spike[i] + 19800 != spike[i+95]:
            print(n)
'''

#Data.compile_results("network_plots", "unseen_digit_th0.7", 3, 25, 20)
#file1 =  json.load(open(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-5_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-1\neuron_data.json"))#

#file2 =  json.load(open(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-5_layers-2_train_inst-20_traindigits-[0, 1]_test_inst-25_testdigits-[0, 1, 2]_w-6_th-0.7_p-0.1_par-True_seed-1_1\neuron_data.json"))

#for k1, k2 in zip(list(file1.keys())[-25:], list(file2.keys())[-25:]):
#    if file1[k1] != file2[k2]:
#        for kk1, kk2 in zip(file1[k1], file2[k2]):
#            if file1[k1][kk1] != file2[k2][kk2]:
#                print(k1, k2)
#                print(file1[k1][kk1])
#                print(file2[k2][kk2])




#run_training_phase(img=4, layers=3, train_inst=20, train_digits=[0,1], test_inst=25, test_digits=[0,1,2], w=6, th=0.7, p=0.3, par=True, seed=1)
#m = Data.load_model(r"C:\Users\jorge\PycharmProjects\MasterThesis\network_plots\TrainingPhase_img-12_layers-2_num-[0, 1]_train_inst-20_test_inst-10_w-6_th-0.7_p-0.05_par-True\post_sim_model.pkl")
#m.build_pgs()
#m.plot_raster()
#m.save_PG_data()




Data.compile_results("G:/USABLE RESULTS/unseen digit 0.8", "unseen_digit_th0.8", 3, 25, 20)
Data.compile_results("G:/USABLE RESULTS/unseen digit 0.9", "unseen_digit_th0.9", 3, 25, 20)

#ddir = "G:/USABLE RESULTS/unseen digit 0.9"
#param = []
#for dir in os.listdir(ddir):
#    param.append((ddir, dir))
def change_threshold(ddir, dir):
    print(dir)
    m = Data.load_model(os.path.join(os.path.join(ddir,dir),"post_sim_model.pkl"))
    m.dir = os.path.join(ddir, dir)
    m.build_pgs(min_threshold=0.9)
    m.save_PG_data()
    m.plot_raster()
    Data.save_model(m, os.path.join(ddir,os.path.join(dir, "post_sim_model.pkl")))
    os.rename(os.path.join(ddir,dir), os.path.join(ddir, dir.replace('th-0.7', 'th-0.9')))

#if __name__ == '__main__':
#
#   with mp.Pool(30) as p:
#        p.starmap(change_threshold, param)


