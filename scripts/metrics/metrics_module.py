from astropy.io import fits
import numpy as np
import matplotlib
import pylab as plt
import scipy.stats as stats
from scipy.stats import norm
matplotlib.rcParams.update({'font.size': 18})



def metrics_XIDp(samples,truth):
    """ returns error percent, precision (IQR/median), and accuracy ((output-truth)/truth) from XIDp samples (no.samp,no. parameters)"""
    nsamp,nparam = samples.shape
    error_percent=np.empty((nparam))
    IQR_med=np.empty((nparam))
    accuracy=np.empty((nparam))
    for i in range(0,nparam):
        error_percent[i]=norm.ppf(stats.percentileofscore(samples[:,i],truth[i])/100.0,loc=0,scale=1)
        IQR_med[i]=np.subtract(*np.percentile(samples[:,i],[75.0,25.0]))/truth[i]
        accuracy[i]=(np.median(samples[:,i])-truth[i])/truth[i]
    return error_percent,IQR_med,accuracy


def metrics_plot(metric,truth,bins,labels,ylim,yscale='linear',cmap=None):
    def upper(x):
        return np.percentile(x,[84.1])
    def lower(x):
        return np.percentile(x,[15.9])
        
    fig,ax=plt.subplots(figsize=(8,7))
    ind_good=np.isfinite(metric)
    mean=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic='median',bins=bins)
    std_dev=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=np.std,bins=bins)
    sig_plus=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=upper,bins=bins)
    sig_neg=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=lower,bins=bins)
    if cmap is None:
        cmap=plt.get_cmap('Blues')

    ax.plot(bins[0:-1],mean[0],'ko',linestyle='-')
    ax.plot(bins[0:-1],sig_plus[0],'ko',linestyle='--')
    ax.plot(bins[0:-1],sig_neg[0],'ko',linestyle='--')
    #ax.plot(bins[0:-1],mean[0]-std_dev[0],'r--')
    #ax.plot(bins[0:-1],mean[0]+std_dev[0],'r--')
    ax.set_xlabel(labels[0])
    ax.set_xscale('log')
    ax.set_ylabel(labels[1])
    ax.set_xlim((3.0,np.max(bins)))
    ind_good_hex=(metric > np.min(mean[0])-2*np.max(std_dev[0])) & (metric < np.max(mean[0])+2*np.max(std_dev[0]))
    if yscale !='linear':
        ax.set_yscale('log')
        tmp = ax.hexbin(truth[ind_good_hex], metric[ind_good_hex], gridsize=40, cmap=cmap,xscale = 'log',yscale='log')#,extent=(np.min(truth),np.max(truth),np.min(mean[0])-2*np.max(std_dev[0]),np.max(mean[0])+2*np.max(std_dev[0])))
    else:
        try:
            tmp = ax.hexbin(truth[ind_good_hex], metric[ind_good_hex], gridsize=40, cmap=cmap,xscale = 'log')#,extent=(np.min(truth),np.max(truth),np.min(mean[0])-2*np.max(std_dev[0]),np.max(mean[0])+2*np.max(std_dev[0])))
            ax.axhline(linewidth=4, color='k',alpha=0.5)
        except ValueError:  #raised if `y` is empty.
            pass
    ax.set_ylim(ylim)
    clrbar=fig.colorbar(tmp, ax=ax)
    clrbar.set_label(r'$N_{Gal.}$')
    return fig

def metrics_plot_nodensity(metric,truth,metricDES,truthDES,bins,labels,ylim,yscale='linear',cmap=None):
    def upper(x):
        return np.percentile(x,[84.1])
    def lower(x):
        return np.percentile(x,[15.9])

    fig,ax=plt.subplots(figsize=(8,7))
    ind_good=np.isfinite(metric)
    mean=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic='median',bins=bins)
    std_dev=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=np.std,bins=bins)
    sig_plus=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=upper,bins=bins)
    sig_neg=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=lower,bins=bins)

    ind_goodDES=np.isfinite(metricDES)
    meanDES=stats.binned_statistic(truthDES[ind_goodDES],metricDES[ind_goodDES],statistic='median',bins=bins)
    std_dev=stats.binned_statistic(truthDES[ind_goodDES],metricDES[ind_goodDES],statistic=np.std,bins=bins)
    sig_plusDES=stats.binned_statistic(truthDES[ind_goodDES],metricDES[ind_goodDES],statistic=upper,bins=bins)
    sig_negDES=stats.binned_statistic(truthDES[ind_goodDES],metricDES[ind_goodDES],statistic=lower,bins=bins)

    if cmap is None:
        cmap=plt.get_cmap('Blues')

    ax.plot(bins[0:-1],mean[0],'o',linestyle='-',color=cmap(0.9)[0:3])
    #ax.plot(bins[0:-1],sig_plus[0],'o',linestyle='--',color=cmap(0.9)[0:3])
    #ax.plot(bins[0:-1],sig_neg[0],'o',linestyle='--',color=cmap)
    ax.fill_between(bins[0:-1],sig_plus[0],sig_neg[0],color=cmap(0.9)[0:3],alpha=0.5)
    #ax.plot(bins[0:-1],mean[0]-std_dev[0],'r--')
    #ax.plot(bins[0:-1],mean[0]+std_dev[0],'r--')
    print 'am i ok here'
    ax.plot(bins[0:-1],meanDES[0],'ko',linestyle='--')
    ax.plot(bins[0:-1],sig_plusDES[0],'ko',linestyle='--')
    ax.plot(bins[0:-1],sig_negDES[0],'ko',linestyle='--')
    ax.set_xlabel(labels[0])
    ax.set_xscale('log')
    ax.set_ylabel(labels[1])
    ax.set_xlim((3.0,np.max(bins)))
    if yscale !='linear':
        ax.set_yscale('log')
    else:
        try:
            ax.axhline(linewidth=4, color='k',alpha=0.5)
        except ValueError:  #raised if `y` is empty.
            pass
    ax.set_ylim(ylim)

    return fig

def metrics_plot_nodensity_XIDp(metric,truth,bins,labels,ylim,yscale='linear',cmap=None):
    def upper(x):
        return np.percentile(x,[84.1])
    def lower(x):
        return np.percentile(x,[15.9])

    fig,ax=plt.subplots(figsize=(8,7))
    ind_good=np.isfinite(metric)
    mean=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic='median',bins=bins)
    std_dev=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=np.std,bins=bins)
    sig_plus=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=upper,bins=bins)
    sig_neg=stats.binned_statistic(truth[ind_good],metric[ind_good],statistic=lower,bins=bins)

   
    if cmap is None:
        cmap=plt.get_cmap('Blues')

    ax.plot(bins[0:-1],mean[0],'o',linestyle='-',color=cmap(0.9)[0:3])
    #ax.plot(bins[0:-1],sig_plus[0],'o',linestyle='--',color=cmap(0.9)[0:3])
    #ax.plot(bins[0:-1],sig_neg[0],'o',linestyle='--',color=cmap)
    ax.fill_between(bins[0:-1],sig_plus[0],sig_neg[0],color=cmap(0.9)[0:3],alpha=0.5)
    #ax.plot(bins[0:-1],mean[0]-std_dev[0],'r--')
    #ax.plot(bins[0:-1],mean[0]+std_dev[0],'r--')
    print 'am i ok here'
    ax.set_xlabel(labels[0])
    ax.set_xscale('log')
    ax.set_ylabel(labels[1])
    ax.set_xlim((3.0,np.max(bins)))
    if yscale !='linear':
        ax.set_yscale('log')
    else:
        try:
            ax.axhline(linewidth=4, color='k',alpha=0.5)
        except ValueError:  #raised if `y` is empty.
            pass
    ax.set_ylim(ylim)

    return fig
