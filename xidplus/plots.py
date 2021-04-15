from tempfile import NamedTemporaryFile
import base64
from xidplus import posterior_maps as postmaps
import seaborn as sns
import aplpy
import pylab as plt
import numpy as np


import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore',UnicodeWarning)

VIDEO_TAG = """<video style="max-width:100%" controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):

    """

    :param anim: matplotlib animation
    :return: an html embedded animation
    """
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('utf-8')

    return VIDEO_TAG.format(anim._encoded_video)
from matplotlib import animation
from IPython.display import HTML


def display_animation(anim):

    """

    :param anim: matplotlib animation
    :return: displayed animation in notebook
    """
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


def plot_map(priors):

    """Plot of the fitted maps, with fitted objects overplotted

    :param priors: list of xidplus.prior classes
    :return: the default xidplus map plot
    """
    sns.set_style("white")

    cmap=sns.cubehelix_palette(8, start=.5, rot=-.75,as_cmap=True)
    hdulists=list(map(lambda prior:postmaps.make_fits_image(prior,prior.sim), priors))
    fig = plt.figure(figsize=(10*len(priors),10))
    figs=[]
    for i in range(0,len(priors)):
        figs.append(aplpy.FITSFigure(hdulists[i][1],figure=fig,subplot=(1,len(priors),i+1)))

    for i in range(0,len(priors)):
        vmin=np.min(priors[i].sim)
        vmax=np.max(priors[i].sim)
        figs[i].show_colorscale(vmin=vmin,vmax=vmax,cmap=cmap)
        figs[i].show_markers(priors[i].sra, priors[i].sdec, edgecolor='black', facecolor='black',
                marker='o', s=20, alpha=0.5)
        figs[i].tick_labels.set_xformat('dd.dd')
        figs[i].tick_labels.set_yformat('dd.dd')
        figs[i].add_colorbar()
        figs[i].colorbar.set_location('top')
    return figs,fig



def replicated_map_movie(priors,posterior, frames):
    """

    :param priors: list of xidplus.prior classes
    :param posterior: xidplus.posterior class
    :param frames: number of frames
    :return: Movie of replicated maps. Each frame is a sample from the posterior
    """
    mod_map_array=postmaps.replicated_maps(priors,posterior,frames)
    # call our new function to display the animation
    return make_map_animation(priors,mod_map_array,frames)


def make_map_animation(priors,mod_map_array, frames):
    """

    :param priors: list of xidplus.prior classes
    :param mod_map_array: list of model map arrays
    :param frames: number of frames
    :return: Movie of replicated maps. Each frame is a sample from the posterior
    """
    figs, fig = plot_map(priors)
    cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, as_cmap=True)

    def animate(i):
        for b in range(0, len(priors)):
            figs[b]._data[
                priors[b].sy_pix - np.min(priors[b].sy_pix) - 1, priors[b].sx_pix - np.min(priors[b].sx_pix) - 1] = \
            mod_map_array[b][:, i]
            figs[b].show_colorscale(vmin=np.min(priors[b].sim), vmax=np.max(priors[b].sim), cmap=cmap)

        return figs

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                   frames=frames, interval=1000)

    return display_animation(anim)

def plot_Bayes_pval_map(priors, posterior):

    """

    :param priors: list of xidplus.prior classes
    :param posterior: xidplus.posterior class
    :return: the default xidplus Bayesian P value map plot
    """
    sns.set_style("white")
    mod_map_array = postmaps.replicated_maps(priors, posterior, posterior.samples['src_f'].shape[0])
    Bayes_pvals = []

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    hdulists = list(map(lambda prior: postmaps.make_fits_image(prior, prior.sim), priors))
    fig = plt.figure(figsize=(10 * len(priors), 10))
    figs = []
    for i in range(0, len(priors)):
        figs.append(aplpy.FITSFigure(hdulists[i][1], figure=fig, subplot=(1, len(priors), i + 1)))
        Bayes_pvals.append(postmaps.make_Bayesian_pval_maps(priors[i], mod_map_array[i]))

    for i in range(0, len(priors)):
        figs[i].show_markers(priors[i].sra, priors[i].sdec, edgecolor='black', facecolor='black',
                             marker='o', s=20, alpha=0.5)
        figs[i].tick_labels.set_xformat('dd.dd')
        figs[i].tick_labels.set_yformat('dd.dd')
        figs[i]._data[
            priors[i].sy_pix - np.min(priors[i].sy_pix) - 1, priors[i].sx_pix - np.min(priors[i].sx_pix) - 1] = \
        Bayes_pvals[i]
        figs[i].show_colorscale(vmin=-6, vmax=6, cmap=cmap)
        figs[i].add_colorbar()
        figs[i].colorbar.set_location('top')
    return figs, fig