""" plotting functions """

import numpy as np
import matplotlib.pyplot as plt

def plot_At(A, ci='2sd', times=None, ax=None, skipdiag=False, labels=None, 
    showticks=True, color=None, fold=False, line=None, cond=None, width=None, 
    vline=False, zbar=False, pairs=None, **kwargs):
    """ plot traces of each entry of dynamics A in square grid of subplots """
    if A.ndim == 3:
        T, d, _ = A.shape
    elif A.ndim == 4:
        _, T, d, _ = A.shape

    if times is None:
        times = np.arange(T)

    if ax is None or ax.shape != (d, d):
        fig, ax = plt.subplots(d, d, sharex=True, sharey=True, squeeze=True)#False)
        
    else:
        fig = ax[0, 0].figure
    
    
    pair=False
    COLOR = color
    for i in range(d):
        for j in range(d):
            
            if pairs:
                for n in range(len(pairs)):
                    if i==pairs[n][0]:
                        if j==pairs[n][1]:
                            pair=True; I=i; J=j
            
            # only print in lower triangle
            ii = i 
            jj = j
            if fold and j > i:
                jj = i
                ii = j
                color = 'black'
            else:
                color = COLOR
                
            # skip and hide subplots on diagonal
            if skipdiag and i == j:
                ax[i, j].set_visible(True)
                continue
            # vertical line params for gap duration
            if i == j:
                ymin = -0
                ymax = 1
                # continue
            else:
                ymin = -0.25
                ymax = 0.25
            if vline:
                ax[i,j].vlines(85,ymin,ymax,colors=color,#'lightgray',  #VLINES
                                    linestyles='dashed') #gap onset
                ax[i,j].vlines(165,ymin,ymax,colors=color,#'lightgray',
                                    linestyles='dashed') #gap offset
            if zbar:
                if i != j:
                    ax[i,j].text(1.75,-2.5, round(np.mean(A[:,i,j]),3))
            # plot A entry as trace with/without error band
            if A.ndim == 3:
                if pair:
                    # print(f'PAIRS={pairs}')
                    if i==I and j==J: color='k'
                    else: color=COLOR
                    pair=False
                ax[i, j].plot(times[:-1], A[:-1, i, j], color=color,#**kwargs)
                               linestyle=line, linewidth=width, **kwargs)
            elif A.ndim == 4:
                plot_fill(A[:, :-1, i, j], ci=ci, times=times[:-1],
                          ax=ax[ii, jj], color=color, cond=cond, line=line, **kwargs)

            # add labels above first row and to the left of the first column
            if labels is not None:
                if i == 0 or (skipdiag and (i, j) == (1, 0)):
                    ax[i, j].set_title(labels[j], fontsize=12)
                if j == 0 or (skipdiag and (i, j) == (0, 1)):
                    ax[i, j].set_ylabel(labels[i], fontsize=12)

            # remove x- and y-ticks on subplot
            if not showticks:
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

    return fig, ax

def plot_fill(X, times=None, ax=None, ci='sd', color=None, cond=None, line=None, **kwargs):
    """ plot mean and error band across first axis of X """
    N, T = X.shape

    if times is None:
        times = np.arange(T)
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    mu = np.mean(X, axis=0)

    # define lower and upper band limits based on ci
    if ci == 'sd':    # standard deviation
        sigma = np.std(X, axis=0)
        lower, upper = mu - sigma, mu + sigma
        # s = float('%.1g' % np.mean(sigma))
        # ax.title.set_text(f'std dev = {s}')
    elif ci == 'se':    # standard error
        stderr = np.std(X, axis=0) / np.sqrt(X.shape[0])
        lower, upper = mu - stderr, mu + stderr
    elif ci == '2sd':    # 2 standard deviations
        sigma = np.std(X, axis=0)
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
    elif ci == 'max':    # range (min to max)
        lower, upper = np.min(X, axis=0), np.max(X, axis=0)
    elif type(ci) is float and 0 < ci < 1:
        # quantile-based confidence interval
        a = 1 - ci
        lower, upper = np.quantile(X, [a / 2, 1 - a / 2], axis=0)
    else:
        raise ValueError("ci must be in ('sd', 'se', '2sd', 'max') "
                         "or float in (0, 1)")

    ax.fill_between(times, lower, upper, color=color, alpha=0.3, lw=0)#c
    lines = ax.plot(times, mu, color='cornflowerblue', linestyle=line, **kwargs)#'color'
    c = lines[0].get_color()
