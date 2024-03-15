#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:57:11 2022

@author: jordandrew
"""

from graphStats_ROIs import FC_stats, drawSumNet, plot_cond, plot_histo, ROIs, bs
import numpy as np
import networkx as nx

thresh = np.linspace(0.04, 0.21, num=18) #90th percentile = 0.11
thresh_90th = 0.11
thresh_bs = 95 #95/100 bootstraps for significance
   
LL = FC_stats('LL') #maintain space
LR = FC_stats('LR') #switch space
LX = FC_stats('LX') #space -> pitch
UU = FC_stats('UU') #maintain pitch
UD = FC_stats('UD') #switch pitch
UX = FC_stats('UX') #pitch -> space
conds = [LL,LR, UU, UD, LX, UX]
experiment = 'Experiment 1 (Original Study)'

Abar_all = []
for C in conds:
    H = []
    C.loadEXP1()
    for th in thresh:
        th = round(th,2)
        if C.net != None: C.net = None
        c, T, _, d = C.A_n.shape  
        assert c == bs
        for i in range(bs):
            A = C.A_n[i,:,:,:]
            C.buildNet(A, bsNo=i, thresh=th, thresh_90th=thresh_90th)
        
        sig_edges = np.full_like(C.A_n, np.nan)
        G = C.net.copy() #pseudo graph for removal of edges
        edges = C.net.edges
        weights = [C.net[u][v]['weight'] for u,v in edges]
        for e,w in zip(edges, weights):
            # print(f'{e} : {w}')
            # # print(len(edges))
            if w >= thresh_bs: #significant edges - ≥95 bootstraps                       
                sig_edges[:,:,ROIs.index(e[0]),ROIs.index(e[1])] = C.A_n[:,:,ROIs.index(e[0]),ROIs.index(e[1])]        
            else: #remove edge
                G.remove_edge(e[0],e[1])
            # print(len(edges))
        if th == thresh_90th:
            plot_cond(C, thresh_90th, thresh_bs, experiment, sig_edges)
        H.append(G)
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
    C.sumNet = sumNet
    drawSumNet(sumNet, experiment, colormap=True, maxWeight=maxW, thresh=thresh_90th)
    Abar_all = np.concatenate((Abar_all,C.Abar))
plot_histo(Abar_all, thresh_90th, experiment)
    



























