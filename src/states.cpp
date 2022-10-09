/*
#include <pybind11/pybind11.h>


namespace py = pybind11;
/*
float update_v(float v, float u, float i, float dt){
    float first_v = v + 0.5 *(0.04 * pow(v, 2) + 5*v +140 - u + i) * dt;
    float second_v = first_v + 0.5 *(0.04 * pow(v, 2) + 5*v +140 - u + i) * dt;
    return second_v;
};
*/
/*
float update_u(float u, float v, float a, float b, float dt){
    float new_u = u + a * (b * v - u) * dt;
    return new_u;
};

PYBIND11_MODULE(update_states, handle){
    handle.doc() = "Module to update neuron states";
    //handle.def("update_v", &update_v);
    handle.def("update_u", &update_u);
}
*/