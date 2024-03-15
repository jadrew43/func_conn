# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 09:54:12 2022

@author: jad91
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_At
import networkx as nx
import glob
import os.path as op
import yaml
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

bs = 100
ROIs = ['ACC','lDLPFC','rDLPFC','lAUD','rAUD','lFEF','rFEF','Vis','lIPS','LIPSP','rIPS','RTPJ']
# index   0       1        2       3      4      5     6      7      8      9      10     11
d = len(ROIs)
experiment_dir = './analysis output'

with open('labels.pkl','rb') as f:
    label_names = pickle.load(f)
    
def set_plot(ax,times):
    diag_lims = [0, 1]
    off_lims = [-0.25, 0.25]
    for ri, row in enumerate(ax):
        for ci, a in enumerate(row):
            ylim = diag_lims if ri == ci else off_lims
            a.set(ylim=ylim, xlim=times[[0, -1]])
            for line in a.lines:
                line.set_clip_on(False)
                line.set(lw=1.)
            if ci != 0:
                a.yaxis.set_major_formatter(plt.NullFormatter())
            if ri != d - 1: #len(label_names) - 1:
                a.xaxis.set_major_formatter(plt.NullFormatter())
            if ri == ci:
                for spine in a.spines.values():
                    spine.set(lw=2)
            else:
                a.axhline(0, color='k', ls=':', lw=1.)                
    # ax[0, 0].legend()
    plt.tight_layout()
    plt.show()
                    
def drawNet(G, bsNo=None, colormap=False):
    w0 = np.linspace(85, 100, 100)
    for i in range(len(w0)):
        w0[i] = round(w0[i])
    w1 = np.linspace(.01,1,100)
    fig, ax = plt.subplots()
    edges = G.edges()
    if colormap: 
        pos = {'ACC':(0,-7),'lFEF':(-1,-8),'rFEF':(1,-8),'lDLPFC':(-1.5,-6),'rDLPFC':(1.5,-6),'lIPS':(-1,-4.5),'rIPS':(1,-4.5),'LIPSP':(-1.5,-3),'RTPJ':(1.5,-3),'lAUD':(-1,-1),'rAUD':(1,-1),'Vis':(0,0)}
        cmap = mpl.cm.get_cmap('YlOrRd')
        weights = 2#[G[u][v]['weight']/100 for u,v in edges]
        G1 = G.copy(); G2 = G.copy()
        sig = 0
        # print(f'{G.graph["thresh"]}-------------------------')
        for r in ROIs:
            for rr in ROIs:
                if r != rr:
                    try:
                        wt = G[r][rr]['weight']
                        # print(wt)
                        w = w1[np.where(w0==wt)[0][3]]
                        color = cmap(w) 
                        # print(w)
                        if wt >= 95:
                            sig += 1
                            G1[r][rr]['color'] = color
                            G2.remove_edge(r, rr)
                        else:
                            G1.remove_edge(r,rr)
                            G2[r][rr]['color'] = color
                    except:
                        continue
        colors1 = [G1[u][v]['color'] for u,v in G1.edges()]  
        colors2 = [G2[u][v]['color'] for u,v in G2.edges()]
        img = plt.imshow(np.array([[.85,1]]), cmap=cmap, aspect='auto')
        img.set_visible(False)
        plt.colorbar(orientation="vertical")
    else: 
        pos = {'ACC':(0,7),'lFEF':(-1,8),'rFEF':(1,8),'lDLPFC':(-1.5,6),'rDLPFC':(1.5,6),'lIPS':(-1,5),'rIPS':(1,5),'LIPSP':(-1.5,3),'RTPJ':(1.5,3),'lAUD':(-1,1),'rAUD':(1,1),'Vis':(0,0)}
        weights = [G[u][v]['weight'] for u,v in edges]
        colors = None
    nx.draw(G1, pos=pos, with_labels=True, width=weights, edge_color=colors1, 
            connectionstyle='arc3, rad=0.1')
    nx.draw(G2, pos=pos, with_labels=True, width=weights, edge_color=colors2, 
            connectionstyle='arc3, rad=0.1', style='dashed')
    cond = G.graph['cond']
    thr = G.graph['thresh']
    if bsNo==None: 
        fig.suptitle(f'{cond}_{thr:.2f}_{sig}')
    else:
        fig.suptitle(f'{cond}_bs{bsNo}')
    plt.show()

def drawSumNet(G, experiment, colormap=False, maxWeight=0, thresh=None):
    fig, ax = plt.subplots()
    edges = G.edges()
    if colormap: 
        x = 0.5
        pos = {'ACC':(0,-7),'lFEF':(-1+x,-6),'rFEF':(1-x,-6),'lDLPFC':(-1.45+x,-8),
                'rDLPFC':(1.5-x,-8),'lIPS':(-1+x,-4.5),'rIPS':(1-x,-4.5),
                'LIPSP':(-1.5+x,-3),'RTPJ':(1.5-x,-3),'lAUD':(-1+x,-1),
                'rAUD':(1-x,-1),'Vis':(0,0)} #flips with colorbar
        # pos = {'ACC':(0,7),'lFEF':(-1+x,6),'rFEF':(1-x,6),'lDLPFC':(-1.25+x,8),
        #        'rDLPFC':(1.25-x,8),'lIPS':(-1+x,4.5),'rIPS':(1-x,4.5),
        #        'LIPSP':(-1.5+x,3),'RTPJ':(1.5-x,3),'lAUD':(-1+x,1),
        #        'rAUD':(1-x,1),'Vis':(0,0)}
        cmap = mpl.cm.get_cmap('YlOrRd')
        weights1 = 4
        weights2 = 2#[G[u][v]['weight']/100 for u,v in edges]
        G1 = G.copy(); G2 = G.copy()
        for r in ROIs:
            for rr in ROIs:
                if r != rr:
                    try:
                        w = G[r][rr]['weight']
                        w = w / maxWeight #normalized weight, not actual sigLvl
                        # print(f'{r} -> {rr}: w={w}')
                        color = cmap(w) 
                        
                        #thresh_array[ind] = np.linspace(0,1,size(thresh))[ind]
                        #h    =  0    25    50    75    95    100
                        #        a    b     c     d     e
                        #p    = [55    75    90    95    99   100]
                        #exp1 = [.04  .07   .11   .13   .18   .21]
                        #exp2 = [.04  .07   .11   .14   .19   .21]
                        
                        w1 = [0.0, 0.18, 0.41, 0.58, 0.88, 1.0] 
                        
                        if w < w1[1]:
                            h = 0
                        elif w1[1] <= w <= w1[2]: #[25:50]
                            h = 25 + (w-w1[1])/(w1[2] - w1[1])*25
                        elif w1[2] <= w <= w1[3]: #[50:75]
                            h = 50 + (w-w1[2])/(w1[3] - w1[2])*25
                        elif w1[3] <= w <= w1[4]: #[75:95]
                            h = 75 + (w-w1[3])/(w1[4] - w1[3])*20
                        elif w1[4] <= w <= w1[5]: #[95:100]
                            h = 95 + (w-w1[4])/(w1[5] - w1[4])*5
                        h = h/100; color = cmap(h)
                        # cond = G.graph['cond']; print(f'{cond} {w} -> {h}')
                        if h >= 0.5: #thresh = 0.11 map 0:1 to sweep .04:.12
                           G1[r][rr]['color'] = color
                           G2.remove_edge(r, rr)
                        elif h > .25:
                            G1.remove_edge(r,rr)
                            G2[r][rr]['color'] = color
                        else:
                           G1.remove_edge(r,rr)
                           G2.remove_edge(r,rr)
                    except:
                        continue
        colors1 = [G1[u][v]['color'] for u,v in G1.edges()]  
        colors2 = [G2[u][v]['color'] for u,v in G2.edges()]  
        img = plt.imshow(np.array([[0,100]]), cmap=cmap, aspect='auto')
        img.set_visible(False)
        cbar = plt.colorbar(img, ticks=[25, 50, 75, 95], orientation="vertical")
        cbar.ax.set_yticklabels(['75th','90th','95th','99th'], fontsize=10)
    else: 
        pos = {'ACC':(0,7),'lFEF':(-1,8),'rFEF':(1,8),'lDLPFC':(-1.5,6),'rDLPFC':(1.5,6),'lIPS':(-1,5),'rIPS':(1,5),'LIPSP':(-1.5,3),'RTPJ':(1.5,3),'lAUD':(-1,1),'rAUD':(1,1),'Vis':(0,0)}
        weights = [G[u][v]['weight'] for u,v in edges]
        colors = None
    nx.draw(G1, pos=pos, with_labels=True, width=weights1, edge_color=colors1, 
            connectionstyle='arc3, rad=0.1', font_size=16)
    nx.draw(G2, pos=pos, with_labels=True, width=weights2, edge_color=colors2, 
            connectionstyle='arc3, rad=0.1', style='dashed', font_size=16, node_color='lightblue')
    cond = G.graph['cond']
    thr = G.graph['thresh_bs']
    
    group = experiment
    if cond == 'UD' or cond[:2] == 'SP':
        title = f'{group} - Switch Pitch' 
    if cond == 'LR' or cond[:2] == 'SS':
        title = f'{group} - Switch Space'  
    if cond == 'UU' or cond[:2] == 'MP':
        title = f'{group} - Maintain Pitch' 
    if cond == 'LL' or cond[:2] == 'MS':
        title = f'{group} - Maintain Space' 
    if cond == 'LX' :
        title = f'{group} - Space to Pitch' 
    if cond == 'UX':
        title = f'{group} - Pitch to Space' 
    if cond[:2] == 'MB':
        title = f'{group} - Maintain Both' 
    if cond[:2] == 'SB':
        title = f'{group} - Switch Both' 
    # print(cond)
    fig.suptitle(title, size='x-large')
    plt.show()
    
    
def drawDiffNet0(G, cond1, cond2):
    fig, ax = plt.subplots()
    B, G1G2, G2G1, G1, G2, b = G
    x=.0
    y=0
    pos = {'ACC':(0,8),'lFEF':(-1+x,6),'rFEF':(1-x,6),'lDLPFC':(-1.5+y,8),
            'rDLPFC':(1.5-y,8),'lIPS':(-1+x,4+y),'rIPS':(1-x,4+y), 
            'LIPSP':(-1.5+y,3),'RTPJ':(1.5-x,3),'lAUD':(-1+x,0),
            'rAUD':(1-x,0),'Vis':(0,2)}
    
    #ONE CUE ASD
    color_map = []
    for node in G1.nodes:
        if node == 'LIPSP' or node == 'RTPJ':
            print(node)
            color_map.append('lightblue')
        else:
            color_map.append('violet')
            
    nx.draw(G1G2, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16)#
    nx.draw(B, pos=pos, with_labels=True, width=3, edge_color='green',  font_size=16)
    # nx.draw(G1, pos=pos, with_labels=True, width=3, edge_color='darkviolet',  font_size=16) #deeppink #darkviolet
    nx.draw(G2, pos=pos, with_labels=True, width=3, edge_color='gold',  font_size=16, node_color=color_map) #gold #darkorange
    # print(color_map)
    
    
    #FOR STATS
    # nx.draw(G1, pos=pos, with_labels=True, width=3, edge_color='red', connectionstyle='arc3, rad=0.1', font_size=16)
    # nx.draw(G1G2, pos=pos, with_labels=True, width=3, edge_color='red', connectionstyle='arc3, rad=0.1', style='dashed', font_size=16)
    # nx.draw(B, pos=pos, with_labels=True, width=3, edge_color='limegreen', connectionstyle='arc3, rad=0.1', font_size=16)
    # nx.draw(b, pos=pos, with_labels=True, width=3, edge_color='limegreen', connectionstyle='arc3, rad=0.1', style='dashed', font_size=16)
    # nx.draw(G2, pos=pos, with_labels=True, width=3, edge_color='darkblue', connectionstyle='arc3, rad=0.1', font_size=16)
    # nx.draw(G2G1, pos=pos, with_labels=True, width=3, edge_color='darkblue', connectionstyle='arc3, rad=0.1', style='dashed', font_size=16, node_color='lightblue')
   
    #APRIORI FIG
    # nx.draw(G1G2, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=-0.09', font_size=16)#
    # nx.draw(G1, pos=pos, with_labels=True, width=3, edge_color='darkblue', connectionstyle='arc3, rad=-0.09', font_size=16)#
    # nx.draw(B, pos=pos, with_labels=True, width=3, edge_color='green',  font_size=16)
    # nx.draw(G2, pos=pos, with_labels=True, width=3, edge_color='green', connectionstyle='arc3, rad=-0.09', font_size=16, node_color='lightblue')
    
    #CAAN
    # nx.draw(B, pos=pos, with_labels=True, width=3, edge_color='gold', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(G2, pos=pos, with_labels=True, width=3, edge_color='gold', connectionstyle='arc3, rad=0.1',  font_size=16)#, style='dashed')
    # nx.draw(G2_, pos=pos, with_labels=True, width=3, edge_color='gold', connectionstyle='arc3, rad=0.1',  font_size=16, style='dashed')
    # nx.draw(G1, pos=pos, with_labels=True, width=3, edge_color='blueviolet', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(G1G2, pos=pos, with_labels=True, width=3, edge_color='darkturquoise', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(b, pos=pos, with_labels=True, width=3, edge_color='green', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(b_, pos=pos, with_labels=True, width=3, edge_color='green', connectionstyle='arc3, rad=0.1',  font_size=16, style='dashed')
    # nx.draw(G2G1, pos=pos, with_labels=True, width=3, edge_color='deeppink', connectionstyle='arc3, rad=0.1', font_size=16)
    # nx.draw(G2G1_, pos=pos, with_labels=True, width=3, edge_color='deeppink', connectionstyle='arc3, rad=0.1', font_size=16, style='dashed', node_color='lightblue')
    
    # nx.draw(B, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(G2, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16)#, style='dashed')
    # nx.draw(G2_, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16, style='dashed')
    # nx.draw(G1, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(G1G2, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(b, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16)
    # nx.draw(b_, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1',  font_size=16, style='dashed')
    # nx.draw(G2G1, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1', font_size=16)
    # nx.draw(G2G1_, pos=pos, with_labels=True, width=3, edge_color='white', connectionstyle='arc3, rad=0.1', font_size=16, style='dashed', node_color='lightblue')
    
    #FOR LEGEND FOR STATS
    # h1 = nx.draw_networkx_edges(G1, pos=pos, width=3, edge_color='red', connectionstyle='arc3, rad=0.1', label='O.E. significant')
    # h2 = nx.draw_networkx_edges(G1G2, pos=pos, width=3, edge_color='red', connectionstyle='arc3, rad=0.1', style='dashed', label='O.E. sig C.V. n.s.')
    # h3 = nx.draw_networkx_edges(B, pos=pos, width=3, edge_color='limegreen', connectionstyle='arc3, rad=0.1', label='Both significant')
    # h4 = nx.draw_networkx_edges(b, pos=pos, width=3, edge_color='limegreen', connectionstyle='arc3, rad=0.1', style='dashed', label='Both n.s.' )
    # h5 = nx.draw_networkx_edges(G2, pos=pos, width=3, edge_color='darkblue', connectionstyle='arc3, rad=0.1', label='C.V. significant')
    # h6 = nx.draw_networkx_edges(G2G1, pos=pos, width=3, edge_color='darkblue', connectionstyle='arc3, rad=0.1', style='dashed', label='C.V. sig O.E. n.s.')
    # H = [h1,h2,h3,h4,h5,h6]
    
    clrs = ['red','red','limegreen','limegreen','darkblue','darkblue']
    styles = [None,'dashed',None,'dashed',None,'dashed']
    # clrs = ['red','orange']
    # styles = [None, 'dashed']
    
    group = ' (Validation)'
    conds = [cond1, cond2]; titles = []; cos = [];
    for cond in conds:
        if len(cond) > 2: cond = cond[:2]
        if cond == 'UD' or cond == 'SP':
            title = f'Switch Pitch{group}'; co = 'SP'
        elif cond == 'LR' or cond == 'SS':
            title = f'Switch Space{group}'; co = 'SS' 
        elif cond == 'UU' or cond == 'MP':
            title = f'Maintain Pitch{group}' ; co = 'MP'
        elif cond == 'LL' or cond == 'MS':
            title = f'Maintain Space{group}' ; co = 'MS'
        elif cond == 'LX':
            title = f'Space to Pitch{group}' ; co = 'S->P'
        elif cond == 'UX':
            title = f'Pitch to Space{group}' ; co = 'P->S'
        elif cond == 'MB':
            title = f'Maintain Both{group}' ; co = 'MB'
        elif cond == 'SB':
            title = f'Switch Both{group}' ; co = 'SB'
        else:
            title = cond
        titles.append(title); cos.append(cond)#co)
    
    # co1 = cos[0]; co2 = cos[1]
    co1 = 'Exp 1'; co2 = 'Exp 2'
    labels = [f'{co1} significant',f'{co1} sig {co2} near-sig','Both significant',
              'Both near-significant',f'{co2} significant',f'{co2} sig {co1} near-sig'] #SPS
    # labels = [f'{co1} significant',f'{co1} sig {co2} near-sig','Both significant',
    #           'Both near-sig',f'{co2} significant',f'{co2} sig {co1} near-sig'] #SPS
    # labels = ['significant', 'near-significant']
    print(f'cond={cond1}vs{cond2},title={title}')
    
    def make_proxy(clr, mappable, style, **kwargs):
        return Line2D([0,1],[1,0], color=clr, linestyle=style, **kwargs)        
    # proxies = [make_proxy(clr, h, style, lw=3) for clr,h,style in zip(clrs,H,styles)]
    # cond = G.graph['cond']
    # fig.suptitle(f'{titles[0]}', size='x-large')#f'{cond}')
    fig.suptitle(f'One Cue - NT and ASD', size='x-large')#f'{cond}')
    # plt.legend(proxies,labels,loc=10, framealpha=.1, fontsize=10)#.5
    # fig.savefig(f'/Users/jordandrew/Dropbox/UW/presentations/MS figs/one cue.eps')
    plt.show()
    
    
def plot(At_bs, times, cond): 
    _,_,d,dd = At_bs.shape
    assert d == dd                   
    skip = False
    fig, ax = plt.subplots(d,d, constrained_layout=True, squeeze=False,
                            figsize=(12,10))
    # _ = plot_At(truLR, times=times, label='ground truth', skipdiag=skip, ax=ax, width=10)
    _ = plot_At(At_bs, times=times, labels=label_names, line='dashed', skipdiag=skip, ax=ax)
    # _ = plot_At(meanLL, times=timesLL, labels=label_names, skipdiag=skip, ax=ax)
    set_plot(ax,times)
    fig.suptitle(cond)

class FC_stats(object):
    
    def __init__(self, cond):
        self.d = len(label_names) # number of ROIs
        self.net = None
        self.top5 = None
        self.cond = cond
        self.thresh = 0
        self.Abar = []
            
    # def loadSyn(self):
    #     self.n = bs
    #     path = experiment_dir + '/synthetic/std_0.03'
    #     A_n = [] 
    #     for i in range(bs):
    #         fname = glob.glob(op.join(
    #             path, f'bs_{i}.npz')) 
    #         assert len(fname) == 1, f'Need exactly one filename, got {fname}'
    #         fname = fname[0]
    #         data = np.load(fname)
    #         # A, times = data['A'], data['times']
    #         A = data['A']
    #         A_n.append(A)
    #     self.A_n = np.array(A_n)
    #     _, T, d, _ = self.A_n.shape
    #     self.data = data
    #     self.times = range(T)
    #     self.d = d
        
    #     self.t1 = 85 # t = 1.95 in SPS
    #     self.t2 = 165 # t = 2.75 in SPS
    #     # self.times = times
        
    def loadEXP1(self):
        self.n = bs
        path = experiment_dir + '/exp1'
        A_n = [] 
        for i in range(bs):
            fname = glob.glob(op.join(
                path,
                f'cond_{self.cond}-seed_8675309-lam0_*-lam1_*-bs_{i}.npz')) 
            assert len(fname) == 1, f'Need exactly one filename, got {fname}'
            fname = fname[0]
            data = np.load(fname)
            A, times = data['A'], data['times']
            A_n.append(A)
        self.A_n = np.array(A_n)
        self.data = data
        self.t1 = np.where(times==1.95)[0][0]
        self.t2 = np.where(times==2.75)[0][0]
        self.times = times
            
    def loadCLOUDS(self, group, heatmap=False):
        if heatmap:
            path = experiment_dir + f'/clouds/{group}/bootstraps'
            A_n = []
            for i in range(bs):
                fname = glob.glob(op.join(
                    path,
                    f'cond_{self.cond}-seed_8675309-lam0_*-lam1_*-bs_{i}.npz')) 
                assert len(fname) == 1, f'Need exactly one filename, got {fname}'
                fname = fname[0]
                data = np.load(fname)
                A, times = data['A'], data['times']
                A_n.append(A)
            self.A_n = np.array(A_n)
        else: 
            path = experiment_dir + f'/clouds/{group}/veridical'
            fname = glob.glob(op.join(
                path,
                f'cond_{self.cond}-seed_1337-lam0_*-lam1_*.npz')) 
            assert len(fname) == 1, f'Need exactly one filename, got {fname}'
            fname = fname[0]
            data = np.load(fname)
            A, times = data['A'], data['times']
            self.A = A
        self.t1 = np.where(times==2.4)[0][0]
        self.t2 = np.where(times==3.)[0][0]
        self.times = times
        
    def loadEXP2(self):
        path = experiment_dir + f'/exp2'
        A_n = []
        for i in range(bs):
            fname = glob.glob(op.join(
                path,
                f'cond_{self.cond}-seed_8675309-lam0_*-lam1_*-bs_{i}.npz')) 
            assert len(fname) == 1, f'Need exactly one filename, got {fname}'
            fname = fname[0]
            data = np.load(fname)
            A, times = data['A'], data['times']
            A_n.append(A)
        self.A_n = np.array(A_n)
        self.t1 = np.where(times==2.4)[0][0]
        self.t2 = np.where(times==3.)[0][0]
        self.times = times
    
           
    def buildNet(self, A, bsNo=None, avg=False, zscore=False,
                 clouds=None, thresh=0.11, thresh_90th=0.11): # build graph
        d = self.d
        if clouds is not None: self.cond = self.cond + clouds
        else: thresh = thresh
        if self.net==None: self.net = nx.DiGraph(cond=self.cond, thresh=thresh)
        net = self.net
        for roi in ROIs:
            net.add_node(roi) # add node for each ROI (d)
        A = A.T
        for i in range(d):
            for j in range(d):
                if i!=j:
                    edge = A[j][i] # transposed ie i = row j = col 
                    edge = edge[self.t1 : self.t2] # remove values outside time of interest
                    w = np.abs(np.mean(edge)) # weight = average of AR coef
                    if thresh == thresh_90th:
                        self.Abar.append(round(w,3))  
                    if w >= thresh: # if weight > thresh, create edge
                        if avg: w = 100*w**2
                        if bsNo != None: w=1 #binary graphs for nonparametric
                        # print(f'{ROIs[i]} -> {ROIs[j]} w={w}')
                        try:
                            net[ROIs[i]][ROIs[j]]['weight'] += w
                        except:
                            net.add_edge(ROIs[i], ROIs[j], weight=w)
        # drawNet(net, bsNo)
        self.net = net
    
            
def plot_cond(nt, thresh_90th, thresh_bs, experiment, sig_edges=None, line=None):
    times = nt.times
    with open('labels.pkl','rb') as f:
        label_names = pickle.load(f)
    plt.rcParams.update(
        {'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'})
    n_row = len(label_names)
    fig, ax = plt.subplots(n_row, n_row, constrained_layout=True, squeeze=False,
                            figsize=(12, 10))
    if nt.cond[:2] == 'LL' or nt.cond[:2] == 'MS':
      color = 'paleturquoise'; title = 'Maintain Space'
    elif nt.cond[:2] == 'LR' or nt.cond[:2] == 'SS':
      color = 'limegreen'; title = 'Switch Space'
    elif nt.cond[:2] == 'UU' or nt.cond[:2] == 'MP':
      color = 'gold'; title = 'Maintain Pitch'
    elif nt.cond[:2] == 'UD' or nt.cond[:2] == 'SP':
      color = 'palevioletred'; title = 'Switch Pitch'
    elif nt.cond[:2] == 'UX':
      color = 'red'; title = 'Pitch -> Space'
    elif nt.cond[:2] == 'LX':
      color = 'chocolate'; title = 'Space -> Pitch'
    elif nt.cond[:2] == 'MB':
      color = 'palevioletred'; title = 'Maintain Both'
    elif nt.cond[:2] == 'SB':
      color = 'teal'; title = 'Switch Both'
      
    
    plot_At(nt.A_n, labels=label_names, times=times, ax=ax,color=color, line=line, vline=True)
    An = np.asarray(nt.A_n)
    x = nt.times
        
    if sig_edges is not None:
        plot_At(sig_edges, labels=label_names, times=times, ax=ax, line=line, color='black')              
    
    diag_lims = [0, 1]
    off_lims = [-0.3, 0.3]
    fig.suptitle(f'{experiment} - {title}')
    for ri, row in enumerate(ax):
        for ci, a in enumerate(row):
            ylim = diag_lims if ri == ci else off_lims
            a.set(ylim=ylim, xlim=times[[0, -1]])
            if ri == 0:
                a.set_title(a.get_title(), fontsize='small')
            if ci == 0:
                a.set_ylabel(a.get_ylabel(), fontsize='small')
            for line in a.lines:
                line.set_clip_on(False)
                line.set(lw=1.)
            if ci != 0:
                a.yaxis.set_major_formatter(plt.NullFormatter())
            if ri != len(label_names) - 1:
                a.xaxis.set_major_formatter(plt.NullFormatter())
            if ri == ci:
                for spine in a.spines.values():
                    spine.set(lw=2)
            else:
                a.axhline(0, color='k', ls=':', lw=1.)
                
def plot_histo(Abar_all, thresh_90th, experiment):
    n0 = np.percentile(Abar_all, 90)
    n5 = np.percentile(Abar_all, 95)
    n9 = np.percentile(Abar_all, 99)
    s5 = np.percentile(Abar_all, 75)
    if thresh_90th != round(n0,2): #check that manually set threshold = 90th %ile
        print(f'set threshold [{thresh_90th}] != 90th percentile [{round(n0,2)}]')
    fig, ax = plt.subplots()
    hi = plt.hist(Abar_all, bins=50)
    ymax = round(max(hi[0]),-3)
    if ymax < max(hi[0]): ymax += 1000
    y_txt = 0.75*ymax
    y_num = 0.625*ymax                
    plt.axvline(x = n0, color='gray', linestyle='dashed')
    plt.axvline(x = n5, color='gray', linestyle='dashed')
    plt.axvline(x = n9, color='gray', linestyle='dashed')
    plt.axvline(x = s5, color='gray', linestyle='dashed') 
    plt.text(n0, y_txt, ' 90th')
    plt.text(n0, y_num, f' {round(n0,2)}')
    plt.text(n5, y_txt, ' 95th')
    plt.text(n5, y_num, f' {round(n5,2)}')
    plt.text(n9, y_txt, ' 99th')
    plt.text(n9, y_num, f' {round(n9,2)}')
    plt.text(s5, y_txt, ' 75th')
    plt.text(s5, y_num, f' {round(s5,2)}')
    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    plt.title(F'{experiment} - Averaged AR Coefficients per potential FC')
    plt.xlabel('Averaged AR Coefficients (a.u.)')
    plt.ylabel('Count')
    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    plt.ylim((0,ymax))
    plt.show()

    



