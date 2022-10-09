


class Neuron:Node{
    
    static int global_id;
    //std::map<std::map> pg;


    Neuron(const neuron &neuron_type){
        const int id = global_id;
        global_id++;
        const float a = neuron_type.a;
        const float b = neuron_type.b;
        const float c = neuron_type.c;
        const float d = neuron_type.d;
        double u = neuron_type.u_init;
        double v = neuron_type.v_init;
        
        
    };

    void update(){
        for (const Synapse &synapse : up){
            
        };
    };
};