import json5
import time
import logging
import json
import statistics
import multiprocessing
import numpy as np
from qecsim import paulitools as pt

logger = logging.getLogger(__name__)

def run(code,decoder,error_model,error_probability,max_runs):
    """
    This is a packaged code sequence for running simulation max_runs times 
    with recording the failure rate for each logical operator, and returns to a 
    dictionary of running data.
    code: XZZX code
    decoder: XZZX MPS decoder
    error_model: error model used in simulation
    error_probability: error probability parameter used in simulation
    max_runs: number of simulation running times
    return: a dictionary with running data
    """
    wall_time_start = time.perf_counter()
    # initialize runs_data
    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': 1, # 1 for ideal simulation
        'decoder': decoder.label,
        'error_model' : error_model.label,
        'error_probability': error_probability,
        'measurement_error_probability': 0.0, # 0 for ideal simulation
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'n_xfail': 0,
        'n_yfail': 0,
        'n_zfail': 0,
        'n_logical_commutations': None,
        'custom_totals': None,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }
    error_weights = []  # list of error_weight from current run

    # initialize rng
    rng = np.random.default_rng()

    # each error probability is simulated max_run times
    for run in range(max_runs):
        # generate a random error
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=error_model.probability_distribution(error_probability)
        ))
        error = pt.pauli_to_bsf(error_pauli)
        # transform error to syndrome
        syndrome = pt.bsp(error, code.stabilizers.T)
        # decode to find recovery
        recovery = decoder.decode(code, syndrome, error_model, error_probability)
        # check if recovery is success or not
        # check if recovery communicate with stabilizers
        commutes_with_stabilizers = np.all(pt.bsp(recovery^error, code.stabilizers.T) == 0)
        if not commutes_with_stabilizers:
            log_data = {  # enough data to recreate issue
                # models
                'code': repr(code), 'decoder': repr(decoder),
                # variables
                'error': pt.pack(error), 'recovery': pt.pack(recovery),
            }
            logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
        # check if recovery communicate with logical operations
        commutes_with_logicals = np.all(pt.bsp(recovery^error, code.logicals.T) == 0)
        # respectively check if recovery communicate with logical X, Y and Z
        # define logical y 
        logical_ys = code.logical_xs^code.logical_zs
        commutes_with_logicalx = np.all(pt.bsp(recovery^error, code.logical_xs.T) == 0)
        commutes_with_logicaly = np.all(pt.bsp(recovery^error, logical_ys.T) == 0)
        commutes_with_logicalz = np.all(pt.bsp(recovery^error, code.logical_zs.T) == 0)
        # success if recovery communicate with both stabilizers and logical operations
        success = commutes_with_stabilizers and commutes_with_logicals
        # record the logical x, y and z failures seperately
        failure_x = commutes_with_stabilizers and not commutes_with_logicalx
        failure_y = commutes_with_stabilizers and not commutes_with_logicaly
        failure_z = commutes_with_stabilizers and not commutes_with_logicalz
        # increment run counts
        runs_data['n_run'] += 1
        if success:
            runs_data['n_success'] += 1
        else:
            runs_data['n_fail'] += 1
        if failure_x:
            runs_data['n_xfail'] += 1
        if failure_y:
            runs_data['n_yfail'] += 1
        if failure_z:
            runs_data['n_zfail'] += 1
        # append error weight
        error_weights.append(pt.bsf_wt(np.array(error)))

    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    # add rate statistics
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    n_xfail = runs_data['n_xfail']
    n_yfail = runs_data['n_yfail']
    n_zfail = runs_data['n_zfail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0]

    runs_data['logical_failure_rate'] = n_fail / n_run
    runs_data['logicalx_failure_rate'] = n_xfail / n_run
    runs_data['logicaly_failure_rate'] = n_yfail / n_run
    runs_data['logicalz_failure_rate'] = n_zfail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run
    
    return runs_data

def run_once(index,code,decoder,error_model,error_probability,error_paulis):
    """
    This is a sub-function for 'run_multicore' function.
    """
    # index parameter is used for Pool
    id = index
    error_pauli = error_paulis[id]
    error = pt.pauli_to_bsf(error_pauli)
    # transform error to syndrome
    syndrome = pt.bsp(error, code.stabilizers.T)
    # decode to find recovery
    recovery = decoder.decode(code, syndrome, error_model,error_probability)
    # check if recovery is success or not
    # check if recovery communicate with stabilizers
    commutes_with_stabilizers = np.all(pt.bsp(recovery^error, code.stabilizers.T) == 0)
    if not commutes_with_stabilizers:
        log_data = {  # enough data to recreate issue
            # models
            'code': repr(code), 'decoder': repr(decoder),
            # variables
            'error': pt.pack(error), 'recovery': pt.pack(recovery),
        }
        logger.warning('RECOVERY DOES NOT RETURN TO CODESPACE: {}'.format(json.dumps(log_data, sort_keys=True)))
    # check if recovery communicates with logical operations
    commutes_with_logicals = np.all(pt.bsp(recovery^error, code.logicals.T) == 0)
    # respectively check if recovery communicates with logical X, Y and Z
    # define logical y 
    logical_ys = code.logical_xs^code.logical_zs
    commutes_with_logicalx = np.all(pt.bsp(recovery^error, code.logical_xs.T) == 0)
    commutes_with_logicaly = np.all(pt.bsp(recovery^error, logical_ys.T) == 0)
    commutes_with_logicalz = np.all(pt.bsp(recovery^error, code.logical_zs.T) == 0)
    # success if recovery communicate with both stabilizers and logical operations
    success = commutes_with_stabilizers and commutes_with_logicals
    # record the logical x, y and z failures seperately
    failure_x = commutes_with_stabilizers and not commutes_with_logicalx
    failure_y = commutes_with_stabilizers and not commutes_with_logicaly
    failure_z = commutes_with_stabilizers and not commutes_with_logicalz
    error_weight = pt.bsf_wt(np.array(error))
    # return to a list containing success, error_weight and different failure types
    return [success,error_weight,failure_x,failure_y,failure_z]

def run_multicore(code,decoder,error_model,error_probability,max_runs):
    """
    This is a packaged code sequence for running simulation max_runs times 
    with recording the failure rate for each logical operator in multi-cores parallely, 
    and returns to a dictionary of running data.
    code: XZZX code
    decoder: XZZX MPS decoder
    error_model: error model used in simulation
    error_probability: error probability parameter used in simulation
    max_runs: number of simulation running times
    return: a dictionary with running data
    """
    wall_time_start = time.perf_counter()
    # initialize runs_data
    runs_data = {
        'code': code.label,
        'n_k_d': code.n_k_d,
        'time_steps': 1, # 1 for ideal simulation
        'decoder': decoder.label,
        'error_model' : error_model.label,
        'error_probability': error_probability,
        'measurement_error_probability': 0.0, # 0 for ideal simulation
        'n_run': 0,
        'n_success': 0,
        'n_fail': 0,
        'n_xfail' : 0,
        'n_yfail' : 0,
        'n_zfail' : 0,
        'n_logical_commutations': None,
        'custom_totals': None,
        'error_weight_total': 0,
        'error_weight_pvar': 0.0,
        'logical_failure_rate': 0.0,
        'logicalx_failure_rate': 0.0,
        'logicaly_failure_rate': 0.0,
        'logicalz_failure_rate': 0.0,
        'physical_error_rate': 0.0,
        'wall_time': 0.0,
    }
    # count cpu cores
    num_cores = multiprocessing.cpu_count()
    # initialize rng
    rng = np.random.default_rng()
    # generate errors in advance to make sure all the errors are different (otherwise if directly generate error in
    # sub function run_once, the errors are all identical)
    error_paulis = []
    for _ in range(max_runs):
        # generate a random error
        n_qubits = code.n_k_d[0]
        error_pauli = ''.join(rng.choice(
            ('I', 'X', 'Y', 'Z'),
            size=n_qubits,
            p=error_model.probability_distribution(error_probability)
        ))
        error_paulis.append(error_pauli)
    
    # create a Pool and run the sub function in parallel.
    pool = multiprocessing.Pool(processes=num_cores)
    results = [pool.apply_async(run_once,args=(index,code,decoder,
                error_model,error_probability,error_paulis)) for index in range(max_runs)]
    pool.close()
    pool.join()
    
    # get the results from ApplyResult object
    true_results = []
    for result in results:
        true_results.append(result.get()) 
    
    # update run counts
    runs_data['n_run'] = int(len(true_results))
    success, failure_x, failure_y, failure_z = 0, 0, 0, 0
    error_weights = []  # list of error_weight from current run
    for result in true_results:
        success += int(result[0])
        error_weights.append(int(result[1]))
        failure_x += int(result[2])
        failure_y += int(result[3])
        failure_z += int(result[4])
    runs_data['n_success'] = success
    runs_data['n_fail'] = runs_data['n_run'] - runs_data['n_success']
    runs_data['n_xfail'] = failure_x
    runs_data['n_yfail'] = failure_y
    runs_data['n_zfail'] = failure_z
    # error weight statistics
    runs_data['error_weight_total'] = sum(error_weights)
    runs_data['error_weight_pvar'] = statistics.pvariance(error_weights)

    # record wall_time
    runs_data['wall_time'] = time.perf_counter() - wall_time_start

    # add rate statistics
    time_steps = runs_data['time_steps']
    n_run = runs_data['n_run']
    n_fail = runs_data['n_fail']
    n_xfail = runs_data['n_xfail']
    n_yfail = runs_data['n_yfail']
    n_zfail = runs_data['n_zfail']
    error_weight_total = runs_data['error_weight_total']
    code_n_qubits = runs_data['n_k_d'][0]

    runs_data['logical_failure_rate'] = n_fail / n_run
    runs_data['logicalx_failure_rate'] = n_xfail / n_run
    runs_data['logicaly_failure_rate'] = n_yfail / n_run
    runs_data['logicalz_failure_rate'] = n_zfail / n_run
    runs_data['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run
    
    return runs_data

def save_simdata(filename,data):
    """
    Save the simulation result(s) of qecsim to local pc.
    filename: the name or relative path of saved file (format:"xxx.json").
    data: the data needed saving. It is a list of dictionaries.
    return: none
    """
    with open(filename,"w", encoding='utf-8') as f:
        for dict in data:
            f.write(json5.dumps(dict))
            f.write("\n")

def load_simdata(filename):
    """
    Load the file containing simulation result(s) of qecsim
    filename: the name or relative path of saved file (format:"xxx.json").
    return: data of simulation result(s). It is a list of dictionaries.
    """
    data = []
    with open(filename,"r", encoding='utf-8') as f:
        for line in f:
            data.append(json5.loads(line.rstrip(';\n')))
    return data

def reshape_data(data_name,data_num):
    """
    Compose a series of data files into one data file. Each data file name should be like: data_name + data_num + '.json'.
    data_name: the name of data file.
    data_num: number of the data file series.
    return: one data file which contains all simulation information of the data series.
    """
    # make a list store all data together
    data_list = []
    for i in range(data_num):
        data_list.append(load_simdata(data_name+'{}'.format(i)+'.json'))
    # example data file to extract simulation information
    data_eg = load_simdata(data_name+"0.json")
    # collect all data in one list
    data = []
    # compose simulation results of each run 
    for i_run in range(len(data_eg)):
        # initialize result dict
        dict = {
            'code': data_eg[i_run]['code'],
            'n_k_d': data_eg[i_run]['n_k_d'],
            'time_steps': 1, # 1 for ideal simulation
            'decoder': data_eg[i_run]['decoder'],
            'error_probability': data_eg[i_run]['error_probability'],
            'measurement_error_probability': 0.0, # 0 for ideal simulation
            'n_run': 0,
            'n_success': 0,
            'n_fail': 0,
            'n_xfail' : 0,
            'n_yfail' : 0,
            'n_zfail' : 0,
            'n_logical_commutations': None,
            'custom_totals': None,
            'error_weight_total': 0,
            'error_weight_pvar': 0.0,
            'logical_failure_rate': 0.0,
            'physical_error_rate': 0.0,
            'wall_time': 0.0,
        }
        # if there's 'error_model' key in data, add it into dict
        if 'error_model' in data_eg[i_run].keys():
            dict['error_model'] = data_eg[i_run]['error_model']
        # compose simulation results
        for i in range(data_num):
            dict['n_run'] += (data_list[i][i_run])['n_run']
            dict['n_success'] += (data_list[i][i_run])['n_success']
            dict['n_fail'] += (data_list[i][i_run])['n_fail']
            dict['n_xfail'] += (data_list[i][i_run])['n_xfail']
            dict['n_yfail'] += (data_list[i][i_run])['n_yfail']
            dict['n_zfail'] += (data_list[i][i_run])['n_zfail']
            dict['error_weight_total'] += (data_list[i][i_run])['error_weight_total']
            dict['error_weight_pvar'] += (data_list[i][i_run])['error_weight_pvar']
            dict['wall_time'] += (data_list[i][i_run])['wall_time']
        # add rate statistics
        dict['error_weight_pvar'] = dict['error_weight_pvar']/data_num
        time_steps = dict['time_steps']
        n_run = dict['n_run']
        n_fail = dict['n_fail']
        n_xfail = dict['n_xfail']
        n_yfail = dict['n_yfail']
        n_zfail = dict['n_zfail']
        error_weight_total = dict['error_weight_total']
        code_n_qubits = dict['n_k_d'][0]

        dict['logical_failure_rate'] = n_fail / n_run
        dict['logicalx_failure_rate'] = n_xfail / n_run
        dict['logicaly_failure_rate'] = n_yfail / n_run
        dict['logicalz_failure_rate'] = n_zfail / n_run
        dict['physical_error_rate'] = error_weight_total / code_n_qubits / time_steps / n_run

        # add composed dict to list
        data.append(dict)
    return data