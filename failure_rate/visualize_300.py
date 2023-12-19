import sys 
sys.path.append("..") 
import json5
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
import toolbox

ETA = 300
P0_1 = [0.44,0.46,0.49]
P0_2 = [0.31,0.38,0.42]

# ---------------------- plot p vs pf ------------------------
data = toolbox.load_simdata('zbias{}_2.json'.format(ETA))
run_case = data[0]['n_run']

# format plot
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
plt.title('XZZX code, z-bias $\eta = $'+'{} noise'.format(ETA))
plt.xlabel('d')
ax.set_ylabel('$P_f$')
ax.set_ylim(0.25, 0.55)
for p in P0_1:
    d = []
    p_f = []
    for run in data:
        if run['error_probability']>p-0.00001 and run['error_probability']<p+0.00001:
            d.append(run['n_k_d'][2])
            p_f.append(run['logical_failure_rate'])
    # add error bar
    p_f_error = []
    for element in p_f:
        p_f_error.append(np.sqrt(element*(1-element)/run_case))
    ax.errorbar(d,p_f,p_f_error,fmt='o-',capsize=3,markersize=5,label='$p={}$'.format(p))
ax.legend(loc='best',columnspacing=0.2,frameon=True)

fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111)
plt.title('XZZX code, z-bias $\eta = $'+'{} noise'.format(ETA))
plt.xlabel('d')
ax2.set_ylabel('$P_{f,Z}$')
ax2.set_ylim(-0.005, 0.075)
for p in P0_2:
    d = []
    p_fz = []
    for run in data:
        if run['error_probability']>p-0.00001 and run['error_probability']<p+0.00001:
            d.append(run['n_k_d'][2])
            p_fz.append(run['logicalz_failure_rate'])
    # add error bar
    p_fz_error = []
    for element in p_fz:
        p_fz_error.append(np.sqrt(element*(1-element)/run_case))
    ax2.errorbar(d,p_fz,p_fz_error,fmt='d-',capsize=3,markersize=5,label='$p={}$'.format(p))
ax2.legend(loc='best',columnspacing=0.2,frameon=True)
plt.show()