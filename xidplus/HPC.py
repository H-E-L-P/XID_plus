import sys
import os
import numpy as np
from xidplus import moc_routines
from builtins import input
import pickle
import xidplus



def hierarchical_tile(masterfile,tilefile):

    """
    Create Hierarchical tile from Master prior

    :param masterfile: Master prior file
    :param tilefile:  File containing Tiling scheme
    """
    try:
        taskid = np.int(os.environ['SGE_TASK_ID'])
        task_first=np.int(os.environ['SGE_TASK_FIRST'])
        task_last=np.int(os.environ['SGE_TASK_LAST'])

    except KeyError:
        print("Error: could not read SGE_TASK_ID from environment")
        taskid = int(input("Please enter task id: "))
        print("you entered", taskid)


    with open(tilefile, 'rb') as f:
        obj = pickle.load(f)

    tiles = obj['tiles']
    order = obj['order']
    tiles_large = obj['tiles_large']
    order_large = obj['order_large']

    obj=xidplus.io.pickle_load(masterfile)
    priors = obj['priors']

    moc = moc_routines.get_fitting_region(order_large, tiles_large[taskid - 1])
    for p in priors:
        p.moc = moc
        p.cut_down_prior()

    outfile = 'Tile_'+ str(tiles_large[taskid - 1]) + '_' + str(order_large) + '.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump({'priors':priors, 'version':xidplus.io.git_version()}, f)