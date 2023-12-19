import sys 
sys.path.append("..") 
import json5
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
import toolbox

ETA = 300

# load data
code_distance = []
threshold = []
xthreshold = []
ythreshold = []
zthreshold = []
thre_err, xthre_err, ythre_err, zthre_err = [], [], [], []
nu,nux,nuy,nuz = [],[],[],[]
nu_err,nux_err,nuy_err,nuz_err = [],[],[],[]
with open('zbias{}_pc.txt'.format(ETA),"r", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data = line.split(',')
        data[-1] = data[-1].replace('\n','')
        code_distance.append(int(data[0]))
        threshold.append(float(data[1]))
        xthreshold.append(float(data[2]))
        ythreshold.append(float(data[3]))
        zthreshold.append(float(data[4]))
        thre_err.append(float(data[5]))
        xthre_err.append(float(data[6]))
        ythre_err.append(float(data[7]))
        zthre_err.append(float(data[8]))

plt.figure()
plt.errorbar(code_distance,threshold,thre_err,ms=4,fmt='k-D',capsize=3,label='Total')
plt.errorbar(code_distance,xthreshold,xthre_err,ms=4,fmt='r-D',capsize=3,label='Logical X')
plt.errorbar(code_distance,ythreshold,ythre_err,ms=4,fmt='g-D',capsize=3,label='Logical Y')
plt.errorbar(code_distance,zthreshold,zthre_err,ms=4,fmt='b-D',capsize=3,label='Logical Z')
plt.title('XZZX code, z-bias $\eta$ = {} error model'.format(ETA))
plt.xlabel('$d$')
plt.ylabel('$p_c$')
plt.legend()

with open('zbias{}_nu.txt'.format(ETA),"r", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        data = line.split(',')
        data[-1] = data[-1].replace('\n','')
        nu.append(float(data[1]))
        nux.append(float(data[2]))
        nuy.append(float(data[3]))
        nuz.append(float(data[4]))
        nu_err.append(float(data[5]))
        nux_err.append(float(data[6]))
        nuy_err.append(float(data[7]))
        nuz_err.append(float(data[8]))

plt.figure()
plt.errorbar(code_distance,nu,nu_err,ms=4,fmt='k-D',capsize=3,label='Total')
plt.errorbar(code_distance,nux,nux_err,ms=4,fmt='r-D',capsize=3,label='Logical X')
plt.errorbar(code_distance,nuy,nuy_err,ms=4,fmt='g-D',capsize=3,label='Logical Y')
plt.errorbar(code_distance,nuz,nuz_err,ms=4,fmt='b-D',capsize=3,label='Logical Z')
plt.title('XZZX code, z-bias $\eta$ = {} error model'.format(ETA))
plt.xlabel('$d$')
plt.ylabel(r'$\nu$')
plt.legend()

plt.show()