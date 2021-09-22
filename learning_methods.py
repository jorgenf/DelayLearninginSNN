from brian2.units import stdunits as units

D_WND = 6 * ms

def STDDP(synapse, t, d_intvl):
    syn_data = synapse.get_states()
    for ind in range(len(syn_data["lastspike"])):
        pre = syn_data["lastspike"][ind]
        post = syn_data["lastspike_post"][ind]
        delays[ind].append(synapse.delay[ind])
        if post == t * units.ms:
            print(post - pre + synapse.delay[ind])
            if 0 < (post - pre + synapse.delay[ind]) <= D_WND and synapse.delay[ind] > d_intvl[0]:
                synapse.delay[ind] -= 1 * units.ms
            if -D_WND <= (post - pre + synapse.delay[ind]) < 0 and synapse.delay[ind] < d_intvl[1]:
                synapse.delay[ind] += 1 * units.ms