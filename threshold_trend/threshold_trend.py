import sys 
sys.path.append("..") 
import json5
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
import toolbox

ETA = 300

# def get_data(eta,axis,num,d):
#     data = toolbox.load_simdata('zbias{}_{}_logical{}.json'.format(eta,num,axis))
#     # load data for code distance d and d+2
#     x, y1, y2 = [], [], []
#     for run in data:
#         if run['n_k_d'][2] == d:
#             x.append(run['error_probability'])
#             y1.append(run['logical{}_failure_rate'.format(axis)])
#         elif run['n_k_d'][2] == d+2:
#             y2.append(run['logical{}_failure_rate'.format(axis)])
#     return np.array(x),np.array(y1),np.array(y2)

# def get_crosspoint(eta,axis,num,d):
#     """
#     get the cross point probability p of two different code distances.
#     Note: d is integer; axis = 'x','y','z'
#     """
#     x, y1, y2 = get_data(eta,axis,num,d)
#     k1,b1 = np.polyfit(x,y1,1)
#     k2,b2 = np.polyfit(x,y2,1)
#     return (b2-b1)/(k1-k2)

# # write crosspoints into a txt file
# with open('zbias{}.txt'.format(ETA),"w", encoding='utf-8') as f:
#     f.writelines('code_distance, crosspoint_x, crosspoint_y, crosspoint_z')
#     f.write('\n')
#     for d in range(15,21,2):
#         f.writelines(['{},{},{},{}'.format(d,get_crosspoint(ETA,'x',1,d),get_crosspoint(ETA,'y',1,d),get_crosspoint(ETA,'z',1,d))])
#         f.write('\n')
#     for d in range(35,41,2):
#         f.writelines(['{},{},{},{}'.format(d,get_crosspoint(ETA,'x',2,d),get_crosspoint(ETA,'y',2,d),get_crosspoint(ETA,'z',2,d))])
#         f.write('\n')
#     for d in range(55,61,2):
#         f.writelines(['{},{},{},{}'.format(d,get_crosspoint(ETA,'x',3,d),get_crosspoint(ETA,'y',3,d),get_crosspoint(ETA,'z',3,d))])
#         f.write('\n')
#     for d in range(75,79,2):
#         f.writelines(['{},{},{},{}'.format(d,get_crosspoint(ETA,'x',4,d),get_crosspoint(ETA,'y',4,d),get_crosspoint(ETA,'z',4,d))])
#         f.write('\n')
#     for d in range(95,99,2):
#         f.writelines(['{},{},{},{}'.format(d,get_crosspoint(ETA,'x',5,d),get_crosspoint(ETA,'y',5,d),get_crosspoint(ETA,'z',5,d))])
#         f.write('\n')
#     f.close()

def get_threshold(eta,num):
    data_total = toolbox.load_simdata('zbias{}_{}_total.json'.format(eta,num))
    data_x = toolbox.load_simdata('zbias{}_{}_logicalx.json'.format(eta,num))
    data_y = toolbox.load_simdata('zbias{}_{}_logicaly.json'.format(eta,num))
    data_z = toolbox.load_simdata('zbias{}_{}_logicalz.json'.format(eta,num))
    # put data from different code size together
    p_ds, p_dxs, p_dys, p_dzs, pfs, pfxs, pfys, pfzs = [], [], [], [], [], [], [], []
    for run in data_total:
        p_ds.append([run['physical_error_rate'], run['n_k_d'][2]])
        pfs.append(run['logical_failure_rate'])
    for run in data_x:
        p_dxs.append([run['physical_error_rate'], run['n_k_d'][2]])
        pfxs.append(run['logicalx_failure_rate'])
    for run in data_y:
        p_dys.append([run['physical_error_rate'], run['n_k_d'][2]])
        pfys.append(run['logicaly_failure_rate'])
    for run in data_z:
        p_dzs.append([run['physical_error_rate'], run['n_k_d'][2]])
        pfzs.append(run['logicalz_failure_rate'])
    # transform to array
    p_ds = np.array(p_ds)
    p_dxs = np.array(p_dxs)
    p_dys = np.array(p_dys)
    p_dzs = np.array(p_dzs)
    pfs = np.array(pfs)
    pfxs = np.array(pfxs)
    pfys = np.array(pfys)
    pfzs = np.array(pfzs)

    # define fitting function
    def fit_fun(p_d,pc,v,A,B,C):
        """
        global fitting function of p vs pf.
        p_d: data form as (p,d), function input.
        pc,v,A,B,C: function coefficient.
        return: fitting value of function output.
        """
        x = (p_d[0]-pc)*p_d[1]**(1/v)
        return (A + B*x + C*x**2)

    # get fitting coefficients
    popt, pcov = op.curve_fit(fit_fun,p_ds.T,pfs,maxfev=50000)
    pc,nu = popt[0],popt[1]
    # standard deviation of pc and nu
    pc_err, nu_err = pcov[0,0]**0.5, pcov[1,1]**0.5
    poptx, pcovx = op.curve_fit(fit_fun,p_dxs.T,pfxs,maxfev=50000)
    pcx,nux = poptx[0],poptx[1]
    # standard deviation of pc and nu
    pcx_err, nux_err = pcovx[0,0]**0.5, pcovx[1,1]**0.5
    popty, pcovy = op.curve_fit(fit_fun,p_dys.T,pfys,maxfev=50000)
    pcy,nuy = popty[0],popty[1]
    # standard deviation of pc and nu
    pcy_err, nuy_err = pcovy[0,0]**0.5, pcovy[1,1]**0.5
    poptz, pcovz = op.curve_fit(fit_fun,p_dzs.T,pfzs,maxfev=50000)
    pcz,nuz = poptz[0],poptz[1]
    # standard deviation of pc and nu
    pcz_err, nuz_err = pcovz[0,0]**0.5, pcovz[1,1]**0.5
    with open('zbias{}_pc.txt'.format(ETA),"a", encoding='utf-8') as f:
        f.writelines(['{},{},{},{},{},{},{},{},{}'.format(10*num-3,pc,pcx,pcy,pcz,pc_err,pcx_err,pcy_err,pcz_err)])
        f.write('\n')
    f.close()
    with open('zbias{}_nu.txt'.format(ETA),"a", encoding='utf-8') as f:
        f.writelines(['{},{},{},{},{},{},{},{},{}'.format(10*num-3,nu,nux,nuy,nuz,nu_err,nux_err,nuy_err,nuz_err)])
        f.write('\n')
    f.close()

# get thresholds and write them into the file
for num in range(2,11):
    get_threshold(ETA,num)
