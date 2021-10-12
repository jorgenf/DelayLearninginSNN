standard = {"eqs": '''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u : volt
du/dt = a*(b*v-u)                              : volt/second
a = 0.02/ms : 1/second
b = 0.2/ms : 1/second
c = -65*mV : volt
d = 2*mV/ms : volt/second
v_max = 30*mV : volt
I = 20*mV : volt
ref_t = 2*ms : second
v_th = 30*mV : volt
u = -12 mV/second : volt/second
''',
            "reset": '''
v = c
u = u + d
''',
            "threshold":'''v>v_th'''
            }

no_units = {"eqs": '''
dv/dt = (0.04)*v**2+(5)*v+140-u : 1
du/dt = a*(b*v-u)                              : 1
a = 0.02 : 1
b = 0.2 : 1
c = -65 : 1
d = 2 : 1
v_max = 30 : 1
I = 15 : 1
ref_t = 2 : 1
''',
            "reset": '''
v = c
u = u + d
'''
            }