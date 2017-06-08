# init file for stan fit functions
import xidplus.io as io

__author__ = 'pdh21'
import os

output_dir = os.getcwd()
full_path = os.path.realpath(__file__)
path, file = os.path.split(full_path)

stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'
stan_path=os.path.split(os.path.split(path)[0])[0]+'/stan_models/'

import pystan
import pickle
import inspect


def get_stancode(model_file):

    """
    Check if exisiting compiled stan code exists and matches current git version, otherwise compile and save.
    
    :param model_file: filename of stan model
    :return: compiled stan code 
    """
    try:
        with open(output_dir+model_file+'.pkl', 'rb') as f:
            # using the same model as before

            print("%s found. Reusing" % model_file)
            obj = pickle.load(f)
            sm=obj['sm']
            if obj['version'] != io.git_version(): raise IOError

    except IOError as e:
        print("%s not found or wrong version. Compiling" % model_file)
        sm = pystan.StanModel(file=stan_path+model_file+'.stan')
        # save it to the file 'model.pkl' for later use
        with open(output_dir+model_file+'.pkl', 'wb') as f:
            pickle.dump({'sm': sm, 'version': io.git_version()}, f)
    return sm
