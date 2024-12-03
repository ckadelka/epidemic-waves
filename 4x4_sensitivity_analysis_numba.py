# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 00:42:13 2024

@author: shahriar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from func_file_utils import save_data, load_data
from sensitivity_analysis_2d import find_peak_nums
from model import simulate


def max_tau_matrix_with_optimized_runtime():
    r0_range = np.linspace(1.2, 3, 10)
    beta_range = np.linspace(0.1, 0.8, 10)
    c_range = [2, 4]
    k_range = [16, 32]
    tau_range = np.linspace(0, 21, 211)
    
    data_dict = {
        'parameters': {
            'r0_range': r0_range,
            'beta_range': beta_range,
            'c_range': c_range,
            'k_range': k_range,
            'tau_range': tau_range,
        },
        'matrices': []
    }
    
    for k in k_range:
        for c in c_range:
            tau_matrix = np.zeros((len(r0_range), len(beta_range)))
            wave_num_matrix = np.zeros((len(r0_range), len(beta_range)))
            for i, beta in enumerate(beta_range):
                related_gamma_range = beta / r0_range
                for j, gamma in enumerate(related_gamma_range):
                    #print(f"Progress report: (c,k,beta,gamma) = ({c},{k},{beta},{gamma})")
                    wave_num = []
                    for tau in tau_range:
                        ts, results, _, _ = simulate(c=c, k=k, beta=beta, gamma=gamma, tau=tau)
                        current_wave_num = find_peak_nums(results)
                        if wave_num and current_wave_num < wave_num[-1]:
                            break
                        wave_num.append(current_wave_num)
                    index = np.argmax(wave_num)
                    tau_matrix[j, i] = tau_range[index]
                    wave_num_matrix[j, i] = wave_num[-1]
            data_dict['matrices'].append({
                'c': c,
                'k': k,
                'tau_matrix': tau_matrix,
                'wave_num_matrix': wave_num_matrix,
            })
    save_data(data_dict, '4X4_data_hill_func_hill.pkl')


def max_tau_matrix_with_optimized_runtime_mod():
    r0_range = np.linspace(1.5, 4, 6)
    beta_range = np.logspace(-1,0,50)
    c_range = [2]
    k_range = [16]
    tau_range = np.linspace(0, 40, 401)
    
    data_dict = {
        'parameters': {
            'r0_range': r0_range,
            'beta_range': beta_range,
            'c_range': c_range,
            'k_range': k_range,
            'tau_range': tau_range,
        },
        'matrices': []
    }
    
    for k in k_range:
        for c in c_range:
            tau_matrix = np.zeros((len(r0_range), len(beta_range)))
            wave_num_matrix = np.zeros((len(r0_range), len(beta_range)))
            for i, beta in enumerate(beta_range):
                related_gamma_range = beta / r0_range
                for j, gamma in enumerate(related_gamma_range):
                    #print(f"Progress report: (c,k,beta,gamma) = ({c},{k},{beta},{gamma})")
                    wave_num = []
                    for tau in tau_range:
                        ts, results, _, _ = simulate(c=c, k=k, beta=beta, gamma=gamma, tau=tau)
                        current_wave_num = find_peak_nums(results)
                        if wave_num and current_wave_num < wave_num[-1]:
                            break
                        wave_num.append(current_wave_num)
                    index = np.argmax(wave_num)
                    tau_matrix[j, i] = tau_range[index]
                    wave_num_matrix[j, i] = wave_num[-1]
            data_dict['matrices'].append({
                'c': c,
                'k': k,
                'tau_matrix': tau_matrix,
                'wave_num_matrix': wave_num_matrix,
            })
    save_data(data_dict, '4X4_data_hill_func_hill_mod.pkl')

def max_tau_matrix_with_optimized_runtime_mod2():
    r0_range = np.linspace(1.5, 4, 6)
    gamma_range = np.logspace(-1.5,0,50)
    c_range = [2]
    k_range = [16]
    tau_range = np.linspace(0, 40, 401)
    
    data_dict = {
        'parameters': {
            'r0_range': r0_range,
            'gamma_range': gamma_range,
            'c_range': c_range,
            'k_range': k_range,
            'tau_range': tau_range,
        },
        'matrices': []
    }
    
    for k in k_range:
        for c in c_range:
            tau_matrix = np.zeros((len(r0_range), len(beta_range)))
            wave_num_matrix = np.zeros((len(r0_range), len(beta_range)))
            for i, gamma in enumerate(gamma_range):
                related_beta_range = gamma * r0_range
                for j, beta in enumerate(related_beta_range):
                    #print(f"Progress report: (c,k,beta,gamma) = ({c},{k},{beta},{gamma})")
                    wave_num = []
                    for tau in tau_range:
                        ts, results, _, _ = simulate(c=c, k=k, beta=beta, gamma=gamma, tau=tau)
                        current_wave_num = find_peak_nums(results)
                        if wave_num and current_wave_num < wave_num[-1]:
                            break
                        wave_num.append(current_wave_num)
                    index = np.argmax(wave_num)
                    tau_matrix[j, i] = tau_range[index]
                    wave_num_matrix[j, i] = wave_num[-1]
            data_dict['matrices'].append({
                'c': c,
                'k': k,
                'tau_matrix': tau_matrix,
                'wave_num_matrix': wave_num_matrix,
            })
    save_data(data_dict, '4X4_data_hill_func_hill_mod2.pkl')


def find_matrix_for_ck_from_optimized_runtime_data(data,c,k):
    for entry in data['matrices']:
        if entry['c'] == c and entry['k'] == k:
            return entry['tau_matrix'], entry['wave_num_matrix']
    return None


def infer_index_given_min_max_number(value,min_value,max_value,number):
    dx = (number-1)/(max_value-min_value)
    return (value-min_value) * dx
 
    
def infer_ticks(ticks,parameter_values):
    min_value = min(parameter_values)
    max_value = max(parameter_values)
    number = len(parameter_values)
    return np.array([infer_index_given_min_max_number(el,min_value,max_value,number) for el in ticks])


def draw_4X4_plot(data):
    c_range = sorted(set(entry['c'] for entry in data['matrices']))
    k_range = sorted(set(entry['k'] for entry in data['matrices']))
    
    global_max = -np.inf
    global_min = np.inf
    for entry in data['matrices']:
        c = entry['c']
        k = entry['k']
        tau_matrix = entry['tau_matrix']
        print(f"Tau matrix for c = {c}, k = {k}:\n{tau_matrix}")
        local_max = np.max(tau_matrix)
        local_min = np.min(tau_matrix)
        if local_max > global_max:
            global_max = local_max
        if local_min < global_min:
            global_min = local_min
            
    f, ax = plt.subplots(2, 2, figsize=(5, 5), sharex='col', sharey='row')
    n_bins = 5*int(global_max - global_min + 1)
    for j, k in enumerate(k_range):
        for i, c in enumerate(c_range):
            tau_matrix, wave_matrix = find_matrix_for_ck_from_optimized_runtime_data(data,c,k)
            tau_matrix = tau_matrix.T
            wave_matrix = wave_matrix.T
            masked_array = np.ma.masked_where(wave_matrix == 1, tau_matrix)
            cmap = plt.get_cmap('jet', n_bins)#rainbow
            bounds = np.linspace(global_min, global_max + 1, n_bins + 1)
            norm = BoundaryNorm(bounds, cmap.N)
            cmap = ListedColormap(cmap(np.linspace(0, 1, n_bins)))
            cmap.set_bad(color='gray')
            images=[]
            img = ax[j, i].imshow(masked_array, cmap=cmap, origin='lower', norm=norm)
            images.append(img)
            
            for y in range(wave_matrix.shape[0]):
                for x in range(wave_matrix.shape[1]):
                    if not np.ma.is_masked(masked_array[y, x]):  
                        text_color = 'white' if tau_matrix[y, x] < 6 or tau_matrix[y, x] > 16 else 'black'
                        ax[j, i].text(x, y, f'{wave_matrix[y, x]:.0f}', 
                                          ha='center', va='center', color=text_color,
                                          fontsize=8)
            
            y_axis_ticks = [0.1,0.4,0.7,1.0]
            y_axis_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            ax[j,i].set_yticks(infer_ticks(y_axis_ticks,y_axis_vals))
            if i==0:
                ax[j,i].set_yticklabels(list(map(str,y_axis_ticks)))
                ax[j,i].set_ylabel(r'transmission rate ($\beta$)')
           
            xticks = [0,3,6,9]
            xlabels = ['1.2','1.8','2.4','3.0']
            ax[j,i].set_xticks(xticks)
            if j==1:
                ax[j,i].set_xticklabels(xlabels)
    
    pos1 = ax[0, 0].get_position()
    pos2 = ax[1, 1].get_position()
    cbar_ax = f.add_axes([pos2.x1 + 0.02, pos2.y0, 0.02, pos1.y1 - pos2.y0])
    
    cbar = f.colorbar(images[0], cax=cbar_ax)
    exten = 2 if global_max%2==0 else 1
    tick_locs = np.arange(global_min + 0.5, global_max + exten + 0.5)
    cbar.set_ticks(tick_locs[(tick_locs-0.5) % 2 == 0])
    tick_labs = np.arange(global_min, global_max + exten, dtype=int)
    cbar.set_ticklabels(tick_labs[tick_labs % 2 == 0])
    cbar.set_label(r'delay causing the maximal number of waves')
    cbar.ax.tick_params(length=0)
    cbar.ax.yaxis.set_tick_params(which='both',length=0)
    #cbar_ax.text(5,global_min+0.5,'low delay',va='center',ha='left')
    #cbar_ax.text(5,global_max+0.5,'high delay',va='center',ha='left')
    
    ax_left = f.add_axes([pos1.x0 - 0.14, pos2.y0, 0.02, pos1.y1 - pos2.y0])
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.plot([0,0],[0,1],'k-',lw=0.5)
    ax_left.set_ylim([0,1])
    ax_left.text(-0.3,0.5,'sensitivity of the contact reduction function',ha='center',va='center',rotation=90)    
    
    ax_left.text(-0.1,0.25,'high ('+r'$k_h =$ '+str(k_range[1])+')',ha='center',va='center',rotation=90)    
    ax_left.text(-0.1,0.75,'low ('+r'$k_h =$ '+str(k_range[0])+')',ha='center',va='center',rotation=90)    
    
    ax_top = f.add_axes([pos1.x0, pos1.y1 + 0.01, pos2.x1 - pos1.x0, 0.02])
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    ax_top.plot([0,1],[0,0],'k-',lw=0.5)
    ax_top.set_xlim([0,1])
    ax_top.text(0.5,0.3,'half-maximal reduction point',ha='center',va='center')    
    ax_top.text(0.25,0.1,'low ('+r'$c =$ '+str(c_range[0])+'%)',ha='center',va='center',rotation=0)    
    ax_top.text(0.75,0.1,'high ('+r'$c =$ '+str(c_range[1])+'%)',ha='center',va='center',rotation=0)    
    
    ax_top.text(0.5,-4.65,r'basic reproduction number ($R_{0}$)',ha='center',va='center')    
    
    f.subplots_adjust(wspace=0.1, hspace=0.1)
    folder_name = os.path.join('data', 'hill_figs')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, '4x4_sensitivity.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    
    max_tau_matrix_with_optimized_runtime()
    
    try:
        data = load_data('4X4_data_hill_func_hill.pkl')
        print("Data loaded successfully")
    except FileNotFoundError as e:
        print(e)
    
    draw_4X4_plot(data)
    

    figsize=(3,2.5)

    #max_tau_matrix_with_optimized_runtime_mod()
    try:
        data = load_data('4X4_data_hill_func_hill_mod.pkl')
        print("Data loaded successfully")
    except FileNotFoundError as e:
        print(e)
        
    ro_range = data['parameters']['r0_range']
    beta_range = data['parameters']['beta_range']
    
    tau_matrix = data['matrices'][0]['tau_matrix']
    wave_num_matrix = data['matrices'][0]['wave_num_matrix']
    f,ax = plt.subplots(figsize=figsize)
    for i in range(tau_matrix.shape[0]):
        argmax_waves = np.argmax(np.bincount(np.array(wave_num_matrix[i],dtype=int)))
        which = wave_num_matrix[i] == argmax_waves
        ax.plot(beta_range[which],tau_matrix[i][which],label=r'$R_0 =$ '+str(ro_range[i]) + ' ('+str(argmax_waves)+' waves)')
    ax.set_xlabel(r'transmission rate $\beta$')
    ax.set_ylabel(r'wave-maximizing delay $\tau$')
    #ax.legend(loc='best',frameon=False)#,title='basic reproduction number')
    ax.spines[['top','right']].set_visible(False) 
    folder_name = os.path.join('data', 'hill_figs')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, 'relationship_beta.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()    
    

    f,ax = plt.subplots(figsize=figsize)
    for i in range(tau_matrix.shape[0]):
        argmax_waves = np.argmax(np.bincount(np.array(wave_num_matrix[i],dtype=int)))
        which = wave_num_matrix[i] == argmax_waves
        ax.plot(1/(beta_range[which]/ro_range[i]),tau_matrix[i][which],label=r'$R_0 =$ '+str(ro_range[i]) + ' ('+str(argmax_waves)+' waves)')
    ax.set_xlabel(r'disease generation time')
    ax.set_ylabel(r'wave-maximizing delay $\tau$')
    #ax.legend(loc='best',frameon=False)#,title='basic reproduction number')
    ax.spines[['top','right']].set_visible(False) 
    folder_name = os.path.join('data', 'hill_figs')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, 'relationship_beta_DGT.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()   
    
    
    #max_tau_matrix_with_optimized_runtime_mod2()
    try:
        data = load_data('4X4_data_hill_func_hill_mod2.pkl')
        print("Data loaded successfully")
    except FileNotFoundError as e:
        print(e)
        
    ro_range = data['parameters']['r0_range']
    gamma_range = data['parameters']['gamma_range']
    
    tau_matrix = data['matrices'][0]['tau_matrix']
    wave_num_matrix = data['matrices'][0]['wave_num_matrix']
    f,ax = plt.subplots(figsize=figsize)
    for i in range(tau_matrix.shape[0]):
        argmax_waves = np.argmax(np.bincount(np.array(wave_num_matrix[i],dtype=int)))
        which = wave_num_matrix[i] == argmax_waves
        ax.plot(gamma_range[which],tau_matrix[i][which],label=r'$R_0 =$ '+str(ro_range[i]) + ' ('+str(argmax_waves)+' waves)')
    ax.set_xlabel(r'recovery rate $\gamma$')
    ax.set_ylabel(r'wave-maximizing delay $\tau$')
    ax.legend(loc='best',frameon=False)#,title='basic reproduction number')
    ax.spines[['top','right']].set_visible(False) 
    folder_name = os.path.join('data', 'hill_figs')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, 'relationship_gamma.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()   
    
    f,ax = plt.subplots(figsize=figsize)
    for i in range(tau_matrix.shape[0]):
        argmax_waves = np.argmax(np.bincount(np.array(wave_num_matrix[i],dtype=int)))
        which = wave_num_matrix[i] == argmax_waves
        ax.plot(1/(gamma_range[which]),tau_matrix[i][which],label=r'$R_0 =$ '+str(ro_range[i]) + ' ('+str(argmax_waves)+' waves)')
    ax.set_xlabel(r'disease generation time')
    ax.set_ylabel(r'wave-maximizing delay $\tau$')
    #ax.legend(loc='best',frameon=False)#,title='basic reproduction number')
    ax.spines[['top','right']].set_visible(False) 
    folder_name = os.path.join('data', 'hill_figs')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_path = os.path.join(folder_name, 'relationship_gamma_DGT.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()  