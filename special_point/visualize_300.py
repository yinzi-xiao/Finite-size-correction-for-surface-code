import sys 
sys.path.append("..") 
import json5
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
import toolbox

ETA = 300
d1_list = [5,25,45,65,85]
d2_list = [3,5,7,9,11]

# ---------------------- plot p vs pf ------------------------
data1 = toolbox.load_simdata('zbias{}_xzzx.json'.format(ETA))
data2 = toolbox.load_simdata('zbias{}_xy.json'.format(ETA))
run_case = data1[0]['n_run']

# format plot
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
plt.title('Z-bias $\eta = $'+'{} noise'.format(ETA))
plt.xlabel('$d_Z$')
ax.set_ylabel('$P_f$')
ax.set_ylim(0.48, 0.6)

# add analytical results
distance_z = np.arange(1,125,0.5)
failure_rate = 3/4 - 1/4*np.exp(-distance_z/ETA)
ax.plot(distance_z,failure_rate,'r-',label='Exact')

# add simulation results
dz1 = []
p_f1 = []
for run in data1:
    if run['n_k_d'][2] in d1_list:
        dz1.append(run['n_k_d'][2])
        p_f1.append(run['logical_failure_rate'])
# add error bar
p_f_error1 = []
for element in p_f1:
    p_f_error1.append(np.sqrt(element*(1-element)/run_case))
ax.errorbar(dz1,p_f1,p_f_error1,fmt='ko',capsize=3,markersize=5,label='XZZX code')

dz2 = []
p_f2 = []
for run in data2:
    if run['n_k_d'][2] in d2_list:
        dz2.append((run['n_k_d'][2])**2)
        p_f2.append(run['logical_failure_rate'])
# add error bar
p_f_error2 = []
for element in p_f2:
    p_f_error2.append(np.sqrt(element*(1-element)/run_case))
ax.errorbar(dz2,p_f2,p_f_error2,fmt='go',capsize=3,markersize=5,label='XY code')
ax.legend(loc='best',columnspacing=0.2,frameon=True)


# ---------------------- plot p vs pfz ------------------------
fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111)
plt.title('Z-bias $\eta = $'+'{} noise'.format(ETA))
plt.xlabel('$d_Z$')
ax2.set_ylabel('$P_{f,Z}$')
ax2.set_ylim(-0.025, 0.2)

# add analytical results
distance_z = np.arange(1,125,0.5)
failure_ratez = 1/2 - 1/2*np.exp(-distance_z/ETA)
ax2.plot(distance_z,failure_ratez,'r-',label='Exact')

# add simulation results
dz1 = []
p_fz1 = []
for run in data1:
    if run['n_k_d'][2] in d1_list:
        dz1.append(run['n_k_d'][2])
        p_fz1.append(run['logicalz_failure_rate'])
# add error bar
p_fz_error1 = []
for element in p_fz1:
    p_fz_error1.append(np.sqrt(element*(1-element)/run_case))
ax2.errorbar(dz1,p_fz1,p_fz_error1,fmt='ko',capsize=3,markersize=5,label='XZZX code')

dz2 = []
p_fz2 = []
for run in data2:
    if run['n_k_d'][2] in d2_list:
        dz2.append((run['n_k_d'][2])**2)
        p_fz2.append(run['logicaly_failure_rate'])
# add error bar
p_fz_error2 = []
for element in p_fz2:
    p_fz_error2.append(np.sqrt(element*(1-element)/run_case))
ax2.errorbar(dz2,p_fz2,p_fz_error2,fmt='go',capsize=3,markersize=5,label='XY code')
ax2.legend(loc='best',columnspacing=0.2,frameon=True)
plt.show()