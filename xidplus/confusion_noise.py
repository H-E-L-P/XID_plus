__author__ = 'pdh21'
import numpy as np

def pixel_distance(prior):
    distance=np.empty((prior.snpix,prior.snpix))
    for i in range(0,prior.snpix):
        distance[:,i]=((prior.sx_pix[i]-prior.sx_pix)**2+(prior.sy_pix[i]-prior.sy_pix)**2)**0.5
    return distance

def select_confusion_cov_max_pixels(max_dist,prior):
    distance=pixel_distance(prior)
    distances=np.unique(distance)
    Rows=np.array([])
    Cols=np.array([])
    Vals=np.array([])
    ii=1
    for dist in distances:
        if dist < max_dist:
            (row_tmp,col_tmp)=np.where(distance == dist)
            Rows=np.append(Rows,row_tmp)
            Cols=np.append(Cols,col_tmp)
            Vals=np.append(Vals,np.full((row_tmp.size), ii, dtype=np.int))
        ii+=1
    return np.array(Rows)+1,np.array(Cols)+1,Vals,ii
