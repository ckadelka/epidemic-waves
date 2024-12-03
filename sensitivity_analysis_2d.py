# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:35:20 2024

@author: shahriar, ckadelka
"""

import numpy as np
from scipy.signal import find_peaks

from func_file_utils import save_data, load_data
from func_draw_heatmap import draw_heatmap
import model

def find_peak_nums(results,min_prominence=0.002):
    peaks, _ = find_peaks(results[:, 1], prominence=min_prominence)
    return len(peaks)

def x_y_sensitivity(x_range, y_range, x_param_name, y_param_name, dt=1, t_end=500, case='Hill', k=16):
    number_of_waves = np.zeros((len(y_range), len(x_range)))
    final_epidemic_sizes = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            if 'k' not in [x_param_name,y_param_name]:
                parameters = {x_param_name: x, y_param_name: y, 'dt': dt, 't_end': t_end, 'case': case, 'k':k}
            else:
                parameters = {x_param_name: x, y_param_name: y, 'dt': dt, 't_end': t_end, 'case': case}
            ts, results, _, _ = model.simulate(**parameters)
            wave_num = find_peak_nums(results)
            number_of_waves[i, j] = wave_num
            final_epidemic_sizes[i,j] = 100-results[-1,0]
    return number_of_waves,final_epidemic_sizes

def calculate_sensitivity_matrices(dt = 1,case='Hill',k=16,tau_min=0,tau_max=20,t_end = 500):
    assert tau_min >=0, 'tau_min >= 0 required'
    assert tau_min < tau_max, 'tau_max > tau_max required'
    n_steps = round((tau_max-tau_min) / dt + 1)
    tau_range = np.linspace(tau_min, tau_max, n_steps)
    c_range = np.linspace(1, 9, n_steps)
    if case=='Hill':
        k_range = np.linspace(6, 36, n_steps)
    elif case=='sigmoid':
        k_range = np.linspace(200, 440, n_steps)
    beta_range = np.linspace(0.2, 0.8, n_steps)
    gamma_range = np.linspace(0.1, 0.4, n_steps)
        
    matrix_tau_c = x_y_sensitivity(tau_range, c_range, 'tau', 'c', dt,case=case,k=k,t_end=t_end)
    matrix_tau_k = x_y_sensitivity(tau_range, k_range, 'tau', 'k', dt,case=case,t_end=t_end)
    matrix_tau_beta = x_y_sensitivity(tau_range, beta_range, 'tau', 'beta', dt,case=case,k=k,t_end=t_end)
    matrix_tau_gamma = x_y_sensitivity(tau_range, gamma_range, 'tau', 'gamma', dt,case=case,k=k,t_end=t_end)
    
    data_dict = {
        'beta_range': beta_range,
        'gamma_range': gamma_range,
        'c_range': c_range,
        'k_range': k_range,
        'tau_range': tau_range,
        'matrix_tau_c': matrix_tau_c,
        'matrix_tau_k': matrix_tau_k,
        'matrix_tau_beta': matrix_tau_beta,
        'matrix_tau_gamma': matrix_tau_gamma,
    }
    
    save_data(data_dict,'sensitivity_matrices_for_%s_with_%i_mesh.pkl' % (case,len(tau_range)))


if __name__ == "__main__":
    dt = 0.125 #days
    tau_max = 20 #days
    tau_min = 0 #days
    t_end = 1000 #days
    case = 'sigmoid'
    k = 250
    
    len_tau_range = round((tau_max-tau_min) / dt + 1)    
    
    try:
        data = load_data('sensitivity_matrices_for_%s_with_%i_mesh.pkl' % (case,len_tau_range))
        print("Data loaded successfully")
    except FileNotFoundError:
        calculate_sensitivity_matrices(dt=dt,case=case,k=k,t_end=t_end,tau_min=tau_min,tau_max=tau_max)
        try:
            data = load_data('sensitivity_matrices_for_%s_with_%i_mesh.pkl' % (case,len_tau_range))
            print("Data loaded successfully")
        except FileNotFoundError as e:
            print(e)
        
    figsize = (4,3)
    
    draw_heatmap(data['matrix_tau_c'][0], data['tau_range'], data['c_range'], 'tau', 'c', case+'_figs',figsize=figsize,global_max=10,global_min=1,SHOWARGMAX=True)
    draw_heatmap(data['matrix_tau_k'][0], data['tau_range'], data['k_range'], 'tau', 'k', case+'_figs',figsize=figsize,global_max=10,global_min=1,SHOWARGMAX=True)
    draw_heatmap(data['matrix_tau_beta'][0], data['tau_range'], data['beta_range'], 'tau', 'beta', case+'_figs',figsize=figsize,global_max=10,global_min=1,SHOWARGMAX=True)
    draw_heatmap(data['matrix_tau_gamma'][0], data['tau_range'], data['gamma_range'], 'tau', 'gamma', case+'_figs',figsize=figsize,global_max=10,global_min=1,SHOWARGMAX=True)
    
    draw_heatmap(data['matrix_tau_c'][1], data['tau_range'], data['c_range'], 'tau', 'c', case+'_figs_FES',figsize=figsize,cmap='inferno_r',FES=True,countour_matrix=data['matrix_tau_c'][0])
    draw_heatmap(data['matrix_tau_k'][1], data['tau_range'], data['k_range'], 'tau', 'k', case+'_figs_FES',figsize=figsize,cmap='inferno_r',FES=True,countour_matrix=data['matrix_tau_k'][0])
    draw_heatmap(data['matrix_tau_beta'][1], data['tau_range'], data['beta_range'], 'tau', 'beta', case+'_figs_FES',figsize=figsize,cmap='inferno_r',FES=True,countour_matrix=data['matrix_tau_beta'][0])
    draw_heatmap(data['matrix_tau_gamma'][1], data['tau_range'], data['gamma_range'], 'tau', 'gamma', case+'_figs_FES',figsize=figsize,cmap='inferno_r',FES=True,countour_matrix=data['matrix_tau_gamma'][0])
        
    figsize = (4,3)
    
    draw_heatmap(data['matrix_tau_beta'][1], data['tau_range'], data['beta_range'], 'tau', 'beta', case+'_figs_FES_reduction_classical',figsize=figsize,cmap='inferno',FES=True,countour_matrix=data['matrix_tau_beta'][0],NORMALIZE_BY_Y=True,clines_color='white',COMPARE_TO_CLASSICAL=True)
    draw_heatmap(data['matrix_tau_gamma'][1], data['tau_range'], data['gamma_range'], 'tau', 'gamma', case+'_figs_FES_reduction_classical',figsize=figsize,cmap='inferno',FES=True,countour_matrix=data['matrix_tau_gamma'][0],NORMALIZE_BY_Y=True,clines_color='white',COMPARE_TO_CLASSICAL=True)
        
    draw_heatmap(data['matrix_tau_beta'][1], data['tau_range'], data['beta_range'], 'tau', 'beta', case+'_figs_FES_reduction',figsize=figsize,cmap='inferno',FES=True,countour_matrix=data['matrix_tau_beta'][0],NORMALIZE_BY_Y=True,clines_color='white',COMPARE_TO_CLASSICAL=False)
    draw_heatmap(data['matrix_tau_gamma'][1], data['tau_range'], data['gamma_range'], 'tau', 'gamma', case+'_figs_FES_reduction',figsize=figsize,cmap='inferno',FES=True,countour_matrix=data['matrix_tau_gamma'][0],NORMALIZE_BY_Y=True,clines_color='white',COMPARE_TO_CLASSICAL=False)
    
    
    