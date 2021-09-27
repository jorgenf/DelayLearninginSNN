from brian2.units import stdunits  as units

eqs = '''
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u : volt
du/dt = a*(b*v-u)                                : volt/second
'''
reset = '''
v = c
u = u + d
'''

a = 0.02/units.ms
b = 0.2/units.ms
c = -65*units.mV
d = 2*units.mV/units.ms
v_max = 30 * units.mV
I = 15*units.mV
ref_t = 2 * units.ms
