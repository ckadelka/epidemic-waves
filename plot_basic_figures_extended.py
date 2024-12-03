# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:32:40 2024

@author: shahriar
"""

import os
import numpy as np
from model_extended import simulate
import matplotlib.pyplot as plt


def hill_vs_sigmoid(c = 0.02, k_s = 250, k_h = [16,24],prevalence_min=0.4,prevalence_max=10):
    fig, ax = plt.subplots(figsize=(3,2.5))
    xs  = np.linspace(prevalence_min,prevalence_max,1000)/100
    lw=2
    ax.semilogx(xs,1/(1 + np.exp(-k_s * (xs - c))),'-',color='g',lw=lw,label='logistic: '+r'$k_s = $'+str(k_s))
    colors= ['k','orange']
    for i,k in enumerate(k_h):
        ax.semilogx(xs,1 - 1/ (1 + (np.log10(c)/np.log10(xs))**k),':',color=colors[i],lw=lw,label='Hill: '+r'$k_h = $'+str(k))        
    [y1,y2] = ax.get_ylim()
    ax.plot([c,c],[y1,0.5],'k:',lw=1)
    ax.set_ylim([y1,y2])
    [x1,x2] = ax.get_xlim()
    ax.plot([x1,c],[0.5,0.5],'k:',lw=1)
    ax.set_xlim([x1,x2])
    yticks = [0,0.25,0.5,0.75,1]
    ax.set_yticks(yticks)
    ax.spines[['right','top']].set_visible(False)
    ax.set_xlabel("prevalence")
    ax.set_ylabel('contact reduction')
    ax.set_yticklabels([str((round(el*100)))+'%'  for el in ax.get_yticks()])  
    ax.set_xticks(np.array([0.4,2,10])/100)
    ax.set_xticklabels([('c = ' if np.isclose(el, 100*c) else '') + str((el))+'%' for el in [0.4,2,10]])  
    legend = ax.legend(loc='center', frameon=False,bbox_to_anchor=[0.38,0.9],ncol=1)
    #legend = ax.legend(loc='center', frameon=False,bbox_to_anchor=[0.84,0.2],ncol=1)
    file_path = os.path.join(folder_name, 'hill_vs_sigmoid_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    

def no_delay_dynamics():
    fig, ax = plt.subplots(figsize=(5,3))
    ax2 = ax.twinx()
    
    ts, results, reduction, Reff = simulate(case='noReduction')
    I = results[:, 1]
    ax.plot(ts, I, color=color[0], linestyle='-')
    ax2.plot(ts, Reff, color=color[0], linestyle='--')
    
    ts, results, reduction, Reff = simulate(case='Hill')
    I = results[:, 1]
    ax.plot(ts, I, color=color[1], linestyle='-')
    ax2.plot(ts, Reff, color=color[1], linestyle='--')
        
    ax.set_xlabel('time in days')
    ax.set_ylabel('prevalence\n(in % of total population)')
    ax2.set_ylabel('effective reproduction number ($R_{eff}$)')
    ax.set_xlim([0,300])
    ax.set_ylim(bottom=None, top=20)
    ax2.set_ylim([-0.05*Reff.max(),1.05*Reff.max()])
    ax2.plot(ts,np.ones(len(ts)), linestyle=':', color='gray', linewidth=1)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', linestyle='-', lw=2, label='prevalence'),
                        Line2D([0], [0], color='k', linestyle='--', lw=2, label='$R_{eff}$'),
                        Line2D([0], [0], color=color[0], linestyle='-', lw=7, label='no contact reduction (standard SIR model)'),
                        Line2D([0], [0], color=color[1], linestyle='-', lw=7, label='immediate fear-based contact reduction')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    ax.spines[['right','top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def no_delay_dynamics_mod(tau=10):
    color = ['k', 'r', 'b']
    fig, ax = plt.subplots(figsize=(5,3))
    ax2 = ax.twinx()
    
    ts, results, reduction, Reff = simulate(case='noReduction')
    I = results[:, 1]
    ax.plot(ts, I, color=color[0], linestyle='-')
    ax2.plot(ts, Reff, color=color[0], linestyle='--')
    
    ts, results, reduction, Reff = simulate(case='Hill')
    I = results[:, 1]
    ax.plot(ts, I, color=color[1], linestyle='-')
    ax2.plot(ts, Reff, color=color[1], linestyle='--')
        
    ts, results, reduction, Reff = simulate(case='Hill',tau=tau)
    I = results[:, 1]
    ax.plot(ts, I, color=color[2], linestyle='-')
    ax2.plot(ts, Reff, color=color[2], linestyle='--')
        
    ax.set_xlabel('time in days')
    ax.set_ylabel('prevalence\n(in % of total population)')
    ax2.set_ylabel('effective reproduction number ($R_{eff}$)')
    ax.set_xlim([0,300])
    ax.set_ylim(bottom=None, top=20)
    ax2.set_ylim([-0.05*Reff.max(),1.05*Reff.max()])
    ax2.plot(ts,np.ones(len(ts)), linestyle=':', color='gray', linewidth=1)
    
    from matplotlib.lines import Line2D
    legend_elements = [
                        Line2D([0], [0], color=color[0], linestyle='-', lw=7, label='no contact reduction (standard SIR model)'),
                        Line2D([0], [0], color=color[1], linestyle='-', lw=7, label='immediate fear-based contact reduction'),
                        Line2D([0], [0], color=color[2], linestyle='-', lw=7, label=str(tau)+'-day delayed contact reduction'),
                        Line2D([0], [0], color='k', linestyle='-', lw=2, label='prevalence'),
                        Line2D([0], [0], color='k', linestyle='--', lw=2, label='$R_{eff}$')
                        ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    ax.spines[['right','top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics_mod_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def no_delay_dynamics_mod2(delay = 5,t_end=500,ylim_top = [11.5,14.2],ylim_bottom = [-0.3,5],  color = ['k', 'r', 'b']):
    I = []
    r = []
    reffs = []
    #fig, ax = plt.subplots(figsize=(3, 2.5))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(3, 2.5),height_ratios=[1,(ylim_bottom[1]-ylim_bottom[0])/(ylim_top[1]-ylim_top[0])])
    fig.subplots_adjust(hspace=0.05)
    lss = ['--','-','-']
    cases = ['noReduction','Hill','Hill']
    delays = [0,0,delay]
    labels = ['no response','immediate response','delayed response']
    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ts, results, reduction, Reff = simulate(tau=delay,case=case,t_end=t_end)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax1.plot(ts, I[i], color=color[i], ls=lss[i], label=label)
        ax2.plot(ts, I[i], color=color[i], ls=lss[i], label=label)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(*ylim_top)  # outliers only
    ax2.set_ylim(*ylim_bottom)  # most of the data
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)
    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.xaxis.set_tick_params(which='both',length=0)
    
    
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    
    ax2.set_xlabel("time in days")
    ax2.set_ylabel('prevalence')
    ax2.yaxis.set_label_coords(-0.02, 0.5, transform=fig.transFigure)
    #ax.grid(False)
    ax1.set_xlim([0, 150])
    ax2.set_xlim([0, 150])
    #ax.set_ylim(bottom=None, top=6.2)
    legend = ax1.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.7],ncol=1)
    ax1.set_yticklabels([str(int(el)) + '%' for el in ax1.get_yticks()])
    ax2.set_yticklabels([str(int(el)) + '%' for el in ax2.get_yticks()])
    #legend = ax1.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    #ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics_mod2_dynamics_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 1.7))
    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ax.plot(ts, r[i], color=color[i], ls=lss[i], label=label)
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('contact reduction')
    ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title(r"delay $(\tau)$")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics_mod2_reduction_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(3, 1.7))
    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ax.plot(ts, reffs[i], color=color[i], ls=lss[i], label=label)
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics_mod2_Reff_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def no_delay_dynamics_mod3(delay = 5,t_end=500,ylim_top = [11.5,14.2],ylim_bottom = [-0.3,5],  color = ['k', 'r', 'b']):
    I = []
    r = []
    reffs = []
    #fig, ax = plt.subplots(figsize=(3, 2.5))
    fig, (ax1, ax2,ax4,ax3) = plt.subplots(4, 1, sharex=True,figsize=(3, 6),height_ratios=[0.3,0.7,0.8,0.8])
    fig.subplots_adjust(hspace=0.05)
    lss = ['--','-','-']
    cases = ['noReduction','Hill','Hill']
    delays = [0,0,delay]
    labels = ['none','immediate','delayed']
    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ts, results, reduction, Reff = simulate(tau=delay,case=case,t_end=t_end)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax1.plot(ts, I[i], color=color[i], ls=lss[i], label=label)
        ax2.plot(ts, I[i], color=color[i], ls=lss[i], label=label)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(*ylim_top)  # outliers only
    ax2.set_ylim(*ylim_bottom)  # most of the data
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)
    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.xaxis.set_tick_params(which='both',length=0)
    
    
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    
    ax2.set_xlabel("time in days")
    ax2.set_ylabel('prevalence')
    ax2.yaxis.set_label_coords(-0.02, 0.75, transform=fig.transFigure)
    #ax.grid(False)
    ax1.set_xlim([0, 150])
    ax2.set_xlim([0, 150])
    #ax.set_ylim(bottom=None, top=6.2)
    legend = ax1.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.7],ncol=1,title='behavioral response')
    ax1.set_yticklabels([str(int(el)) + '%' for el in ax1.get_yticks()])
    ax2.set_yticklabels([str(int(el)) + '%' for el in ax2.get_yticks()])
    #legend = ax1.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    #ax.spines[['right','top']].set_visible(False)

    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ax3.plot(ts, r[i], color=color[i], ls=lss[i], label=label)
    ax3.set_xlabel(f"time in days (delay = {delay} days)")
    ax3.set_xlabel("time in days")
    ax3.set_ylabel('contact reduction')
    ax3.set_yticklabels([str(int(el*100)) + '%' for el in ax3.get_yticks()])
    ax3.grid(False)
    ax3.set_xlim([0, 150])
    ax3.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title(r"delay $(\tau)$")
    ax3.spines[['right','top']].set_visible(False)

    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ax4.plot(ts, reffs[i], color=color[i], ls=lss[i], label=label)
    ax4.set_xlabel(f"time in days (delay = {delay} days)")
    ax4.set_xlabel("time in days")
    ax4.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax4.grid(False)
    ax4.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax4.set_xlim([0, 150])
    ax4.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    ax4.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'no_delay_dynamics_mod3_all_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()

def no_delay_dynamics_mod4(delay = 5,t_end=500,ylim_top = [13.5,16.5],ylim_bottom = [-0.3,5],  color = ['k', 'r', 'b']):
    I = []
    r = []
    reffs = []
    height_ratios = [0.35,0.7,0.8,0.8]
    # We'll use two separate gridspecs to have different margins, hspace, etc
    gs_top = plt.GridSpec(4, 1,hspace=0.05,height_ratios=height_ratios)
    gs_base = plt.GridSpec(4, 1,hspace=0.2,height_ratios=height_ratios)
    fig = plt.figure(figsize=(3, 6))

    # Top (unshared) axes
    ax1 = fig.add_subplot(gs_top[0,:])
    ax2 = fig.add_subplot(gs_top[1,:])
    ax3 = fig.add_subplot(gs_base[3,:])
    ax4 = fig.add_subplot(gs_base[2,:])
    
    # Hide shared x-tick labels
    for ax in [ax1,ax2,ax4]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    # # Plot variable amounts of data to demonstrate shared axes
    # for ax in bottom_axes:
    #     data = np.random.normal(0, 1, np.random.randint(10, 500)).cumsum()
    #     ax.plot(data)
    #     ax.margins(0.05)
        
    
    
    
    #fig, (ax1, ax2,ax4,ax3) = plt.subplots(4, 1, sharex=True,figsize=(3, 6),height_ratios=[0.3,0.7,0.8,0.8])
    lss = ['--','-','-']
    cases = ['noReduction','Hill','Hill']
    delays = [0,0,delay]
    labels = ['none','immediate','delayed']
    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ts, results, reduction, Reff = simulate(tau=delay,case=case,t_end=t_end)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax1.plot(ts, I[i], color=color[i], ls=lss[i], label=label)
        ax2.plot(ts, I[i], color=color[i], ls=lss[i], label=label)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(*ylim_top)  # outliers only
    ax2.set_ylim(*ylim_bottom)  # most of the data
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)
    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.xaxis.set_tick_params(which='both',length=0)
    
    
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    
    #ax2.set_xlabel("time in days")
    ax2.set_ylabel('prevalence')
    ax2.yaxis.set_label_coords(-0.02, 0.75, transform=fig.transFigure)
    #ax.grid(False)
    ax1.set_xlim([0, 150])
    ax2.set_xlim([0, 150])
    #ax.set_ylim(bottom=None, top=6.2)
    legend = ax1.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.5],ncol=1,title='behavioral response')
    ax1.set_yticklabels([str(int(el)) + '%' for el in ax1.get_yticks()])
    ax2.set_yticklabels([str(int(el)) + '%' for el in ax2.get_yticks()])
    #legend = ax1.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    #ax.spines[['right','top']].set_visible(False)
    ax2.set_xticklabels(['' for _ in ax2.get_xticks()])

    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ax3.plot(ts, r[i], color=color[i], ls=lss[i], label=label)
    ax3.set_xlabel(f"time in days (delay = {delay} days)")
    ax3.set_xlabel("time in days")
    ax3.set_ylabel('contact reduction')
    ax3.set_yticklabels([str(int(el*100)) + '%' for el in ax3.get_yticks()])
    ax3.grid(False)
    ax3.set_xlim([0, 150])
    ax3.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title(r"delay $(\tau)$")
    ax3.spines[['right','top']].set_visible(False)

    for i, (case,delay,label) in enumerate(zip(cases,delays,labels)):
        ax4.plot(ts, reffs[i], color=color[i], ls=lss[i], label=label)
    #ax4.set_xlabel(f"time in days (delay = {delay} days)")
    #ax4.set_xlabel("time in days")
    ax4.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax4.grid(False)
    ax4.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax4.set_xlim([0, 150])
    ax4.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    ax4.spines[['right','top']].set_visible(False)
    ax4.set_xticklabels(['' for _ in ax4.get_xticks()])

    file_path = os.path.join(folder_name, 'no_delay_dynamics_mod3_all_extended.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def prevalence_v_reduction_plot(case='Hill',delay=0,e=0.2,beta_e=0):
    fig, ax = plt.subplots(figsize=(2,2.5))
    color = ['k', 'r', 'b']
    # ax2 = ax.twinx()
    ts, results, reduction, Reff = simulate(case=case,tau=delay,e=e,beta_e=beta_e)
    I = results[:, 2]
    ax.plot(I, reduction, color=color[2], linestyle='-',label='delay = %g days' % delay)
    ax.axvline(x=2, color='gray', linestyle=':',linewidth=1)
    indices = np.array([0,200,250,300,500,700,918,1700])#np.linspace(0,1500,10,dtype=int)
    for i in indices:
        if i < len(I) - 1:  # Ensure the index is valid
            ax.annotate('', xy=(I[i + 1], reduction[i + 1]), xytext=(I[i], reduction[i]),
                        arrowprops=dict(arrowstyle='-|>', color=color[2], linewidth=1.5, mutation_scale=15))
    
    
    ts, results, reduction, Reff = simulate(case=case,e=e,beta_e=beta_e)
    I = results[:, 2]
    ax.plot(I, reduction, color=color[1], linestyle='-',label='no delay')
    indices = np.array([0,150,250,600,2000])
    for i in indices:
        if i < len(I) - 1:
            ax.annotate('', xy=(I[i + 1], reduction[i + 1]), xytext=(I[i], reduction[i]),
                        arrowprops=dict(arrowstyle='-|>', color='r', linewidth=1.5, mutation_scale=15))
    ax.set_xlabel('prevalence')
    ax.set_ylabel('contact reduction')
    # ax.set_ylabel('contact reduction\nfor delay = '+str(delay)+' days', color=color[2])
    # ax2.set_ylabel('contact reduction\nfor no delay', color=color[1])
    ax.set_ylim(bottom=-0.05, top=1)
    yticks = [0,0.25,0.5,0.75,1]
    ax.set_yticks(yticks)
    # ax2.set_ylim(bottom=-0.05, top=1)
    # ax2.plot(ts,np.ones(len(ts)), linestyle=':', color='gray', linewidth=1)
    ax.set_xticklabels([(str(int(el)) + '%') if np.isclose(int(el),el) else '' for el in ax.get_xticks()])
    ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    # ax2.set_yticklabels([str(int(el*100)) + '%' for el in ax2.get_yticks()])
    #ax.legend(loc='best',ncol=1, frameon=False)
    ax.spines[['right','top']].set_visible(False)
    # ax2.spines[['top']].set_visible(False)
    file_path = os.path.join(folder_name, 'prevalence_v_reduction_extended_delay%s_e%s_betae%s.pdf' % (str(delay),str(e),str(beta_e)))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()

def delay_dynamics(delay_range=[5]):
    for delay in delay_range:
        ts, results, reduction, Reff = simulate(tau=delay)
        I = results[:, 1]
        fig, ax = plt.subplots(figsize=(5,3))
        ax2 = ax.twinx()
        
        ax.plot(ts, I, color=color[0], linestyle='-')
        ax.set_xlabel('time in days')# (delay = 5 days)')
        ax.set_ylabel('prevalence\n(in % of total population)', color=color[0])
        ax.set_xlim([0,140])
        ax.set_ylim(bottom=None, top=4)
        ax.spines[['top']].set_visible(False)
        
        ax2.plot(ts, Reff, color=color[1], linestyle='--')
        ax2.plot(ts,np.ones(len(ts)), linestyle=':', color='gray', linewidth=1)
        ax2.set_ylabel('effective reproduction number', color=color[1])
        ax2.set_ylim([-0.05*Reff.max(),1.05*Reff.max()])
        ax2.spines[['top']].set_visible(False)
        
        file_path = os.path.join(folder_name, 'delay_dynamics_with_delay_of_'+str(delay)+'_days_extended.pdf')
        plt.savefig(file_path, format='pdf', bbox_inches='tight')
        plt.show()


def dynamics_for_varying_k(delay=5, k_range=[8, 16, 24],case='Hill'):
    I = []
    r = []
    reffs = []
    color = ['r','k','b']
    fig, ax = plt.subplots(figsize=(3, 2.5))
    lss = ['--','-',':']
    for i, k in enumerate(k_range):
        ts, results, reduction, Reff = simulate(k=k, tau=delay,case=case)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax.plot(ts, I[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(k)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('prevalence')
    ax.set_yticklabels([str(int(el)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim(bottom=None, top=4.2)
    legend = ax.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.1],ncol=3)
    legend.set_title("behavioral response sensitivity ($k_h$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'dynamics_for_varying_k_delay%s_kvalues%s_extended.pdf' % (str(delay),'_'.join(list(map(str,k_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, k in enumerate(k_range):
        ax.plot(ts, r[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(k)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('contact reduction')
    ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("sensitivity of\nresponse ($k_h$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'reduction_for_varying_k_delay%s_kvalues%s_extended.pdf' % (str(delay),'_'.join(list(map(str,k_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, k in enumerate(k_range):
        ax.plot(ts, reffs[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(k)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("sensitivity of\nresponse ($k_h$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'Reff_for_varying_k_delay%s_kvalues%s_extended.pdf' % (str(delay),'_'.join(list(map(str,k_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    
def dynamics_for_varying_c(delay=5, c_range=[1,2,4],ymax = 6.2,case='Hill',k=16):
    I = []
    r = []
    reffs = []
    color = ['g','k','purple']
    fig, ax = plt.subplots(figsize=(3, 2.5))
    lss = ['--','-',':']
    for i, c in enumerate(c_range):
        ts, results, reduction, Reff = simulate(c=c, tau=delay,case=case,k=k)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax.plot(ts, I[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('prevalence')
    ax.set_yticklabels([str(int(el)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim(bottom=None, top=ymax)
    legend = ax.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.1],ncol=3)
    legend.set_title("behavioral response midpoint ($c$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'dynamics_for_varying_c_delay%s_cvalues%s_%s_k%i_extended.pdf' % (str(delay),'_'.join(list(map(str,c_range))),case,k))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, c in enumerate(c_range):
        ax.plot(ts, r[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('contact reduction')
    ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("half-maximal reduction point ($c$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'reduction_for_varying_c_delay%s_cvalues%s_%s_k%i_extended.pdf' % (str(delay),'_'.join(list(map(str,c_range))),case,k))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, c in enumerate(c_range):
        ax.plot(ts, reffs[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("half-maximal reduction point ($c$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'Reff_for_varying_c_delay%s_cvalues%s_%s_k%i_extended.pdf' % (str(delay),'_'.join(list(map(str,c_range))),case,k))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def dynamics_for_varying_delay(delay_range=[2,5,18],t_end=150,ylim_top = [11.5,14.2],ylim_bottom = [-0.3,5], color = ['cyan','k','orange']):
    I = []
    r = []
    reffs = []
    #fig, ax = plt.subplots(figsize=(3, 2.5))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(3, 2.5),height_ratios=[1,(ylim_bottom[1]-ylim_bottom[0])/(ylim_top[1]-ylim_top[0])])
    fig.subplots_adjust(hspace=0.05)
    lss = ['--','-',':']
    for i, delay in enumerate(delay_range):
        ts, results, reduction, Reff = simulate(tau=delay,t_end=t_end)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax1.plot(ts, I[i], color=color[i], ls=lss[i], label=r'$%s$' % (str(delay)))
        ax2.plot(ts, I[i], color=color[i], ls=lss[i], label=r'$%s$' % (str(delay)))
    
    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(*ylim_top)  # outliers only
    ax2.set_ylim(*ylim_bottom)  # most of the data
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)
    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.xaxis.set_tick_params(which='both',length=0)
    
    
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    
    ax2.set_xlabel("time in days")
    ax2.set_ylabel('prevalence')
    ax1.set_yticklabels([str(int(el)) + '%' for el in ax1.get_yticks()])
    ax2.set_yticklabels([str(int(el)) + '%' for el in ax2.get_yticks()])
    ax2.yaxis.set_label_coords(-0.02, 0.5, transform=fig.transFigure)
    #ax.grid(False)
    ax1.set_xlim([0, t_end])
    ax2.set_xlim([0, t_end])
    #ax.set_ylim(bottom=None, top=6.2)
    legend = ax1.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.3],ncol=3)

    #legend = ax1.legend(loc='best', frameon=False)
    legend.set_title(r"delay $(\tau)$")
    #ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'dynamics_for_varying_delay_values%s_extended.pdf' % ('_'.join(list(map(str,delay_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, delay in enumerate(delay_range):
        ax.plot(ts, r[i], color=color[i], ls=lss[i], label=r'$%s$' % (str(delay)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('contact reduction')
    ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, t_end])
    ax.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title(r"delay $(\tau)$")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'reduction_for_varying_delay_values%s_extended.pdf' % ('_'.join(list(map(str,delay_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, delay in enumerate(delay_range):
        ax.plot(ts, reffs[i], color=color[i], ls=lss[i], label=r'$%s$' % (str(delay)))
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax.set_xlim([0, t_end])
    ax.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("half-maximal reduction point ($c$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'Reff_for_varying_delay_values%s_extended.pdf' % ('_'.join(list(map(str,delay_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
def dynamics_for_varying_c_broken_axis(delay = 5, c_range=[2,5,18],t_end=500,ylim_top = [11.5,14.2],ylim_bottom = [-0.3,5],  color = ['g','k','purple']):
    I = []
    r = []
    reffs = []
    #fig, ax = plt.subplots(figsize=(3, 2.5))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(3, 2.5),height_ratios=[1,(ylim_bottom[1]-ylim_bottom[0])/(ylim_top[1]-ylim_top[0])])
    fig.subplots_adjust(hspace=0.05)
    lss = ['--','-',':']
    for i, c in enumerate(c_range):
        ts, results, reduction, Reff = simulate(tau=delay,c=c,t_end=t_end)
        I.append(results[:, 1])
        r.append(reduction)
        reffs.append(Reff)
        ax1.plot(ts, I[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')
        ax2.plot(ts, I[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(*ylim_top)  # outliers only
    ax2.set_ylim(*ylim_bottom)  # most of the data
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)
    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    ax1.xaxis.set_tick_params(which='both',length=0)
    
    
    
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
    
    ax2.set_xlabel("time in days")
    ax2.set_ylabel('prevalence')
    ax1.set_yticklabels([str(int(el)) + '%' for el in ax1.get_yticks()])
    ax2.set_yticklabels([str(int(el)) + '%' for el in ax2.get_yticks()])
    ax2.yaxis.set_label_coords(-0.02, 0.5, transform=fig.transFigure)
    #ax.grid(False)
    ax1.set_xlim([0, 150])
    ax2.set_xlim([0, 150])
    #ax.set_ylim(bottom=None, top=6.2)
    legend = ax1.legend(loc='center', frameon=False,bbox_to_anchor=[0.5,1.3],ncol=3)

    #legend = ax1.legend(loc='best', frameon=False)
    legend.set_title("behavioral response midpoint ($c$)")
    #ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'dynamics_for_varying_c_delay%s_cvalues%s_extended.pdf' % (str(delay),'_'.join(list(map(str,c_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, c in enumerate(c_range):
        ax.plot(ts, r[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('contact reduction')
    ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 1.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title(r"delay $(\tau)$")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'reduction_for_varying_c_delay%s_cvalues%s_extended.pdf' % (str(delay),'_'.join(list(map(str,c_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(3, 2.5))
    for i, c in enumerate(c_range):
        ax.plot(ts, reffs[i], color=color[i], ls=lss[i], label=r'$%g$' % (int(c))+'%')
    ax.set_xlabel(f"time in days (delay = {delay} days)")
    ax.set_xlabel("time in days")
    ax.set_ylabel('effective\nreproduction number')
    #ax.set_yticklabels([str(int(el*100)) + '%' for el in ax.get_yticks()])
    ax.grid(False)
    ax.plot([-1,10000],[1,1],linestyle=':', color='gray', linewidth=1)
    ax.set_xlim([0, 150])
    ax.set_ylim([-0.05, 2.05])
    #legend = ax.legend(loc='best', frameon=False)
    #legend.set_title("behavioral response midpoint ($c$)")
    ax.spines[['right','top']].set_visible(False)
    file_path = os.path.join(folder_name, 'Reff_for_varying_c_delay%s_cvalues%s_extended.pdf' % (str(delay),'_'.join(list(map(str,c_range)))))
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.show()
    
    

if __name__ == "__main__":
    
    folder_name = 'basic_figures'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    color = ['k','r','b']
    
    hill_vs_sigmoid()
    no_delay_dynamics_mod4()
    
    dynamics_for_varying_k(delay=5, k_range=[8, 16, 32])
    dynamics_for_varying_c(delay=5, c_range=[1,2,4])
    dynamics_for_varying_delay(delay_range=[2,5,18])
    
    dynamics_for_varying_c(delay=15, c_range=[2,4,6])
    dynamics_for_varying_k(delay=5, k_range=[2,4,6])

    dynamics_for_varying_c_broken_axis(delay=15, c_range=[2,4,6],ylim_top=[10.5,15.5],color = ['k','g','purple'])
    dynamics_for_varying_delay(delay_range=[15,19,23],ylim_top=[10.5,15.5],color = ['k','cyan','orange'])


    