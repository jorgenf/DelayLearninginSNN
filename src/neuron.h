#include "nodes.h"
#include "synapse.h"


class Neuron: public Node{
    private:
    static int global_id;
    static const int threshold = 30;
    double u;
    double v;
    std::list<float> v_hist;
    std::list<float> u_hist;
    std::list<float> spikes;
    std::list<Synapse> up;
    std::list<Synapse> down;
    std::list<int> inputs;
    const int id = global_id;
    global_id++;
    const float a = neuron_type.a;
    const float b = neuron_type.b;
    const float c = neuron_type.c;
    const float d = neuron_type.d;
    double u = neuron_type.u_init;
    double v = neuron_type.v_init;
}