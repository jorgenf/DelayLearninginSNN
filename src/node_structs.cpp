const struct neuron{
    const float a;
    const float b;
    const float c;
    const float d;
    const double u_init;
    const double v_init;
};

const struct FS_neuron : neuron
{
    const float a = 0.1;
    const float b = 0.2;
    const float c = -65;
    const float d = 2;
    const double u_init = 14;
    const double v_init = c;
};

const struct RS_neuron : neuron
{
    const float a = 0.02;
    const float b = 0.2;
    const float c = -65;
    const float d = 8;
    const double u_init = -14;
    const double v_init = -70;
};

const struct RZ_neuron : neuron
{
    const float a = 0.1;
    const float b = 0.26;
    const float c = -65;
    const float d = 2;
    const double u_init = -16;
    const double v_init = c;
};

const struct LTS_neuron : neuron
{
    const float a = 0.02;
    const float b = 0.25;
    const float c = -65;
    const float d = 2;
    const double u_init = -16;
    const double v_init = c;
};

const struct TC_neuron : neuron
{
    const float a = 0.02;
    const float b = 0.25;
    const float c = -65;
    const float d = 0.05;
    const double u_init = -16;
    const double v_init = c;
};

const struct IB_neuron : neuron
{
    const float a = 0.02;
    const float b = 0.2;
    const float c = -55;
    const float d = 4;
    const double u_init = -14;
    const double v_init = c;
};

const struct CH_neuron : neuron
{
    const float a = 0.02;
    const float b = 0.2;
    const float c = -50;
    const float d = 2;
    const double u_init = -14;
    const double v_init = c;
};

const struct POLY_neuron : neuron
{
    const float a = 0.02;
    const float b = 0.2;
    const float c = -65;
    const float d = 2;
    const double u_init = -14;
    const double v_init = c;
};