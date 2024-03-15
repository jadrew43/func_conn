#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:12:41 2022

@author: jordandrew
"""

from graphStats_ROIs import FC_stats, drawNet, drawSumNet, plot_cond, plot_histo, ROIs, bs
import numpy as np
import networkx as nx

thresh = np.linspace(0.04, 0.21, num=18) #[0.11]
thresh_90th = 0.11
thresh_bs = 95

MS = FC_stats('MS') #maintain space
SS = FC_stats('SS') #switch space
MP = FC_stats('MP') #maintain pitch
SP = FC_stats('SP') #switch pitch
MB = FC_stats('MB') #maintain both
SB = FC_stats('SB') #switch both
conds = [MS, SS, MP, SP, MB, SB]   
experiment = 'Experiment 2 (Validation)'

Abar_all = []
for C in conds:
    H = []
    C.loadEXP2()
    for th in thresh:
        th = round(th,2)
        if C.net != None: C.net = None
        c,T,_,_ = C.A_n.shape
        assert c == bs
        for i in range(bs):
            A = C.A_n[i,:,:,:]
            C.buildNet(A, bsNo=i, thresh=th)
            
        sig_edges = np.full_like(C.A_n, np.nan)
        G = C.net.copy()
        edges = C.net.edges
        weights = [C.net[u][v]['weight'] for u,v in edges]
        for e,w in zip(edges, weights):
            # print(f'{e} : {w}')
            # print(len(edges))
            if w >= thresh_bs: #significant edges 
                # print(f'{C.cond}: {e[0]} -> {e[1]}')
                sig_edges[:,:,ROIs.index(e[0]),ROIs.index(e[1])] = C.A_n[:,:,ROIs.index(e[0]),ROIs.index(e[1])]
            else:
                G.remove_edge(e[0],e[1])
        H.append(G)
        if th == thresh_90th:
            plot_cond(C, thresh_90th, thresh_bs, experiment, sig_edges)
    sumNet = nx.DiGraph(cond=C.cond, thresh_bs=thresh_bs)
    for r in ROIs:
        sumNet.add_node(r)
    for i in range(len(H)): #in loop per condition
        # print(f'{H[i].graph["thresh"]}-------------------------')
        for r in ROIs:
            for rr in ROIs:
                try:
                    w = H[i][r][rr]['weight']
                except:
                    continue
                try:
                    sumNet[r][rr]['weight'] += w
                    # print(sumNet[r][rr]['weight'])
                except:
                    sumNet.add_edge(r, rr, weight=w)
    maxW = 100*len(thresh)
    drawSumNet(sumNet, experiment, colormap=True, maxWeight=maxW, thresh=thresh_90th)
    C.sumNet = sumNet
    Abar_all = np.concatenate((Abar_all, C.Abar))
plot_histo(Abar_all, thresh_90th, experiment)
   
    
    
    
    
    
    
    
    
    
    
    

       