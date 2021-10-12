import random
from matplotlib import pyplot as plt
import seaborn as sns
import time
import numpy as np
sns.set()
start = time.time()


dt = 0.00001
t0 = 0
tn = 100
v0 = -65
a = 0.02
b = 0.2
c = -55
d = 4


def integrate(v, u, I):
    v += I
    v = min(v+(0.04*v**2+5*v+140-u)*dt, 30)
    u += (a*(b*v-u))*dt
    return v, u


v_hist = []
u_hist = []
I_hist = []
v = v0
u = -14
t = 0
ref_t = 0
input = [2.0,10.0,11.0, 22.0,21.0, 34.0,35.0, 66.0, 78.0]
for i in range(round((tn-t0)/dt)):
    t += dt
    t = round(t, 5)
    I = 20 if t in input else 0
    #I = 20 if (random.random() < 0.5 * dt and ref_t == 0) else 0
    if I != 0:
        I_hist.append(t)
    v,u = integrate(v, u, I)
    if v >= 30:
        v = c
        u = u + d
        ref_t = 3 / dt
    else:
        ref_t = max(ref_t - 1, 0)
    v_hist.append(v)
    u_hist.append(u)

end = time.time()
print(end-start)

fig, (sub1,sub2, sub3) = plt.subplots(3,1)
sub1.plot(v_hist)
sub1.set_xlim([0,tn/dt])
sub1.set_title("Membrane potential")
sub1.set_ylabel("mV")
sub2.plot(u_hist)
sub2.set_xlim([0,tn/dt])
sub2.set_title("U-variable")
sub2.set_ylabel("U")
sub3.eventplot(I_hist)
sub3.set_xlim([0,tn])
sub3.set_title("Input")
sub3.set_xlabel("Time")
plt.show()

#plot_args = {"v":{"data":v_hist, "mthd":plt.plot, "title": "Membrane potential", "y_label": "Volt", "x_label": "Time"},"u":{"data":u_hist, "mthd":plt.plot, "title":"U-variable","x_label":"Time", "y_label":"U"}, "input":{"data":I_hist, "mthd":plt.eventplot, "title":"Input", "x_label":"Time","y_label":"NeuronID"}}
#plots.plot_data(plot_args, tn)
