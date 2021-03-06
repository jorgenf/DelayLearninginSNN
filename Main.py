import gzip

import Population
from Population import *
import numpy as np
import multiprocessing as mp
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img = [12]
layers = [2]
num = [[0, 8, 0]]
inst = [20]
w = [4, 6]
th = [0.7]
p = [0.1, 0.2]
par = [True]
train = [False, True]
seed = [1]

params = [img, layers, num, inst, w, th, p, par, train, seed]
combos = list(itertools.product(*params))


def do_it(img, layers, num, inst, w, th, p, par, train, seed):
    input = Data.create_mnist_input(inst, num, 200, image_size=img)
    pop = Population((img**2*layers, Population.RS))
    pop.create_feed_forward_connections(d=list(range(1,40)), n_layers=layers, w=w, p=p, partial=par, trainable=True)
    for id, i in enumerate(input):
        ij = [ij for ij in range(img**2) if np.random.random() < p]
        pop.create_input(i, j=ij, wj=w, dj=[np.random.randint(1,40) for x in range(len(ij))], trainable=train)
    pop.run(max([max(inp) for inp in input]) + 100, path='network_plots/', name=f"MNIST_img-{img}_layers-{layers}_num-{num}_inst-{inst}_w-{w}_th-{th}_p-{p}_par-{par}_train-{train}", record_PG=True, save_post_model=False, PG_duration=50, PG_match_th=th, save_data=False)
    #for syn in pop.synapses:
        #syn.trainable = False
    #pop.run(400, record_PG=True, save_post_model=False, PG_duration=50, PG_match_th=th)

#do_it(15, 2, [0, 0, 0], 20, 4, 0.7, 0.05, True, True, 1)

if __name__ == '__main__':
    with mp.Pool(1) as p:
        p.starmap(do_it, combos)


