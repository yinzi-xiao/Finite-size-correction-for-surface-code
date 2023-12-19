import sys 
sys.path.append("..") 
import json5
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
import toolbox

ETA = 300

p_min = 0.31
p_max = 0.51
d_list = [31,35,39,43,47]
special_point = (ETA+1)/(2*ETA+1)

# ---------------------- plot p vs pf ------------------------
data = toolbox.load_simdata('zbias{}.json'.format(ETA))
data.extend(toolbox.load_simdata('zbias{}_sp.json'.format(ETA)))
run_case = data[0]['n_run']

# prepare code to x,y map and print
d_to_xys = {}
for run in data:
    if run['n_k_d'][2] in d_list and run['error_probability']>p_min and run['error_probability']<p_max:
        xys = d_to_xys.setdefault(run['n_k_d'][2], [])
        xys.append((run['error_probability'], run['logical_failure_rate']))
# print('\n'.join('{}: {}'.format(k, v) for k, v in code_to_xys.items()))

# format plot
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
plt.title('Z-bias $\eta = $'+'{}'.format(ETA))
plt.xlabel('Physical error rate')
ax.set_ylabel('Logical failure rate')
ax.set_xlim(p_min, p_max+0.01)
ax.set_ylim(-0.05, 0.6)
# add data
for d, xys in d_to_xys.items():
    x,y = zip(*xys)
    y_error = []
    for element in y:
        y_error.append(np.sqrt(element*(1-element)/run_case))
    ax.errorbar(x,y,y_error,ms=3,fmt='d-',capsize=3,label='d = {} '.format(d) + '$P_f$')
ax.legend(loc=(0,0.65),columnspacing=0.2,frameon=False)

# logical z
# prepare code to x,y map and print
d_to_xys_z = {}
for run in data:
    if run['n_k_d'][2] in d_list and run['error_probability']>p_min and run['error_probability']<p_max:
        xys = d_to_xys_z.setdefault(run['n_k_d'][2], [])
        xys.append((run['error_probability'], run['logicalz_failure_rate']))
# print('\n'.join('{}: {}'.format(k, v) for k, v in code_to_xys.items()))

ax2 = ax.twinx()
ax2.set_ylabel('Logical Z failure rate')
ax2.set_ylim(-0.02, 0.3)
for d, xys in d_to_xys_z.items():
    x,y = zip(*xys)
    y_error = []
    for element in y:
        y_error.append(np.sqrt(element*(1-element)/run_case))
    ax2.errorbar(x,y,y_error,ms=3,fmt='v--',capsize=3,label='d = {} '.format(d) + '$P_f$')

# analytical results for special point
def cal_special_point(eta,d):
    Pf = 3/4-1/4*np.exp(-d/eta)
    Pfz = 1/2-1/2*np.exp(-d/eta)
    return [Pf,Pfz]

# plot the special point 
for d in d_list:
    ax.scatter(special_point,cal_special_point(ETA,d)[0],s=20,marker='o',c='black')
    ax2.scatter(special_point,cal_special_point(ETA,d)[1],s=20,marker='o',c='black')

# pinpoint the special point
plt.vlines([special_point],-0.05,0.75,linestyles='dashed',colors='black')
# custom x-axis ticks
x_value = [0.3,0.35,0.4,0.45,special_point]
x_index = ['0.3','0.35','0.4','0.45','$p_s$']
plt.xticks(x_value,x_index)
ax2.legend(loc=(0.2,0.65),columnspacing=0.2,frameon=False)
plt.show()