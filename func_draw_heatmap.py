# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:36:27 2024

@author: shahriar
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.optimize import fsolve

def infer_index_given_min_max_number(value,min_value,max_value,number):
    dx = (number-1)/(max_value-min_value)
    return (value-min_value) * dx


def infer_ticks(ticks,parameter_values):
    min_value = min(parameter_values)
    max_value = max(parameter_values)
    number = len(parameter_values)
    return np.array([infer_index_given_min_max_number(el,min_value,max_value,number) for el in ticks])


def draw_heatmap(matrix,x_range,y_range,x_param_name,y_param_name,subfolder_name,global_max=None,global_min=None,figsize=(6,4.5),cmap='jet',FES=False,SHOWARGMAX=False,countour_matrix=None,NORMALIZE_BY_Y=False,COMPARE_TO_CLASSICAL=True,clines_color='white'):
    folder_name = os.path.join('data', subfolder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if NORMALIZE_BY_Y==False:
        df = pd.DataFrame(matrix, index=np.round(y_range, 2), columns=np.round(x_range, 2))
        local_min = global_min if global_min is not None else np.array(matrix).min()
        local_max = global_max if global_max is not None else np.array(matrix).max()
        n_bins = int(local_max - local_min + 1)
    else:
        if COMPARE_TO_CLASSICAL:
            if y_param_name=='beta':
                gamma=0.2
                theoretic_FES_values = [fsolve(lambda x: 1-np.exp(-R0*x)-x,[0.5])[0] for R0 in y_range/gamma]
            else:
                beta = 0.4
                theoretic_FES_values = [fsolve(lambda x: 1-np.exp(-R0*x)-x,[0.5])[0] for R0 in beta/y_range]
                
            mod_matrix = np.array([list(map(lambda el: max(0,el),100* (max_value-matrix[i,:])/max_value)) for i,max_value in enumerate(100*np.array(theoretic_FES_values))]) #only reason we see negative values is because we start with a non-zero number of infected
            if y_param_name=='beta':
                mod_matrix[0,:] = 0 #manual mod if R0<=1
            else:
                mod_matrix[-1,:] = 0 #manual mod if R0<=1
                
        else:
            max_per_row = np.max(matrix,1)
            mod_matrix = np.array([max_value-row for row,max_value in zip(matrix,max_per_row)])
        df = pd.DataFrame(mod_matrix, index=np.round(y_range, 2), columns=np.round(x_range, 2))
        local_min = global_min if global_min is not None else np.array(mod_matrix).min()
        local_max = global_max if global_max is not None else np.array(mod_matrix).max()
        n_bins = int(local_max - local_min + 1)        
        

    if FES==False:
        cmap = plt.get_cmap(cmap, n_bins)
        bounds = np.linspace(local_min, local_max + 1, n_bins + 1)
        norm = BoundaryNorm(bounds, cmap.N)
        
        plt.figure(figsize=figsize)
        ax = sns.heatmap(df, cmap=ListedColormap(cmap(np.linspace(0, 1, n_bins))), norm=norm, annot=False, cbar_kws={'label': 'Number of Waves'},rasterized=True)
    else:
        cmap = plt.get_cmap(cmap)
        
        plt.figure(figsize=figsize)
        if COMPARE_TO_CLASSICAL:
            ax = sns.heatmap(df, cmap=cmap, annot=False, cbar_kws={'label': 'relative reduction in final epidemic size\n compared to the classical SIR model\nfor a fixed ' + (r'$\beta$' if y_param_name=='beta' else r'$\gamma$') if NORMALIZE_BY_Y else 'final epidemic size'},rasterized=True)        
        else:
            ax = sns.heatmap(df, cmap=cmap, annot=False, cbar_kws={'label': 'absolute reduction in final epidemic size\n compared to maximal value for fixed ' + (r'$\beta$' if y_param_name=='beta' else r'$\gamma$') if NORMALIZE_BY_Y else 'final epidemic size'},rasterized=True)        
            ax.figure.axes[-1].yaxis.label.set_size(9)
            
    cbar = ax.collections[0].colorbar
    if FES==False:
        exten = 2 if (local_max-local_min)%2==0 else 1
        if local_max > 20:
            tick_locs = np.arange(local_min + 0.5, local_max + exten + 0.5)
            # print(tick_locs)
            cbar.set_ticks(tick_locs[(tick_locs-0.5) % 2 == 0] - 1)
            tick_labs = np.arange(local_min, local_max + exten, dtype=int)
            # print(tick_labs)
            cbar.set_ticklabels(tick_labs[tick_labs % 2 == 0] - 1)
            #cbar.ax.tick_params()
            cbar.set_label('number of waves')
            cbar.ax.yaxis.set_tick_params(which='both',length=0)
        else:            
            tick_locs = np.arange(local_min + 0.5, local_max + 1.5)
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels(np.arange(local_min, local_max + 1, dtype=int))
            #cbar.ax.tick_params()
            cbar.set_label('number of waves')
            cbar.ax.yaxis.set_tick_params(which='both',length=0)
    else:
            tick_locs = np.array([round(np.ceil(local_min)), round(np.floor(local_max))])
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels([str(el)+'%' for el in tick_locs])
            cbar.ax.yaxis.set_tick_params(which='both',length=0)
            if not  NORMALIZE_BY_Y:
                cbar.ax.yaxis.set_label_coords(2,0.5)
        
    ax.tick_params(axis='both', which='major')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.invert_yaxis()
    
    a = y_range.min()
    b = y_range.max()
    if y_param_name == 'k':
        n_yticks = 4
        y_axis_ticks_vals = np.array([f"{val:.0f}" for val in np.linspace(a, b, n_yticks)])
        y_axis_ticks_pos = np.linspace(a, b, n_yticks) + (b-a)/((len(y_range)-1)*2)
        ax.set_ylabel("behavioral response sensitivity ($k_h$)")
    elif y_param_name == 'c':
        n_yticks = 5
        y_axis_ticks_vals = np.array([f"{val:.0f}" for val in np.linspace(a, b, n_yticks)])
        y_axis_ticks_pos = np.linspace(a, b, n_yticks) + (b-a)/((len(y_range)-1)*2)
        #ax.set_ylabel("half-maximal reduction point ($c$)\nin % of total population")
        ax.set_ylabel("behavioral response midpoint ($c$)")
    elif y_param_name == 'beta':
        n_yticks = 4
        y_axis_ticks_vals = np.array([f"{val:.1f}" for val in np.linspace(a, b, n_yticks)])
        y_axis_ticks_pos = np.linspace(a, b, n_yticks) + (b-a)/((len(y_range)-1)*2)
        ax.set_ylabel(r"transmission rate ($\beta$)")
    elif y_param_name == 'gamma':
        n_yticks = 4
        y_axis_ticks_vals = np.array([f"{val:.1f}" for val in np.linspace(a, b, n_yticks)])
        y_axis_ticks_pos = np.linspace(a, b, n_yticks) + (b-a)/((len(y_range)-1)*2)
        ax.set_ylabel(r"recovery rate ($\gamma$)")
    else:
        y_axis_ticks_vals = np.array([f"{val:.2f}" for val in y_axis_ticks_vals])      
    y_axis_vals = np.linspace(a,b,len(y_range))
    ax.set_yticks(infer_ticks(y_axis_ticks_pos,y_axis_vals))
    if y_param_name!='c':
        ax.set_yticklabels(list(map(str,y_axis_ticks_vals)))
    else:
        ax.set_yticklabels([el+'%' for el in list(map(str,y_axis_ticks_vals))])
    
    aa = x_range.min()
    bb = x_range.max()
    x_axis_ticks_pos = np.linspace(aa, bb, 5) + (bb-aa)/((len(x_range)-1)*2)
    x_axis_ticks_vals = np.linspace(aa, bb, 5)
    x_axis_ticks_vals = np.array([f"{val:.0f}" for val in x_axis_ticks_vals])
    x_axis_vals = np.linspace(aa,bb,len(x_range))
    ax.set_xticks(infer_ticks(x_axis_ticks_pos,x_axis_vals))
    ax.set_xticklabels(list(map(str,x_axis_ticks_vals)))
    ax.set_xlabel(r"delay time ($\tau$)")
    
    y_min, y_max = ax.get_ylim()    
    #ax.text(x=-(len(x_range)/5), y=y_max - 0.5, s=h_text, ha='center', va='center')
    #ax.text(x=-(len(x_range)/5), y=y_min + 0.5, s=l_text, ha='center', va='center')

    if SHOWARGMAX:
        max_values = np.max(matrix,1)
        argmax_values = [np.where(row==max_value)[0] for row,max_value in zip(matrix,max_values)]
        mean_argmax_values = np.array(list(map(np.mean,argmax_values)))
        unique_max_values = np.array(list(set(max_values)))
        if max_values[0] > max_values[-1]:
            which = []
            for unique_max_value in unique_max_values[unique_max_values>(1 if y_param_name!='c' else 2)]:
                which.append(max(loc for loc, val in enumerate(max_values) if val == unique_max_value))
            which.append(0)
        else:
            which = []
            for unique_max_value in unique_max_values[unique_max_values>1]:
                which.append(min(loc for loc, val in enumerate(max_values) if val == unique_max_value)) 
            which.append(len(max_values)-1)
        which = np.array(which)
        where_max_2_or_greater = max_values > 1
        ax.plot(mean_argmax_values[which],np.arange(matrix.shape[0])[which],lw=2,color='white')

    if countour_matrix is not None:
        countour_matrix[countour_matrix==0] = 1 #manual mod
        x = np.arange(0,countour_matrix.shape[0], 1)
        y = np.arange(0,countour_matrix.shape[1], 1)
        X,Y = np.meshgrid(x,y)
        X = X * matrix.shape[0]/countour_matrix.shape[0]
        Y = Y * matrix.shape[1]/countour_matrix.shape[1]
        CS = ax.contour(X,Y,countour_matrix,colors=clines_color,levels=min(10,int(countour_matrix.max())-1),linewidths=0.75)
        #ax.clabel(CS, inline=True, fontsize=6)

    file_path = os.path.join(folder_name, 'sen_'+str(x_param_name)+'_'+str(y_param_name)+'_with_'+str(len(x_range))+'_mesh.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    return cbar
    
    
     