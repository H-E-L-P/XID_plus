import subprocess
import os
import numpy as np
from astropy.table import Table



def generate_SEDs(parameter_names,parameters,path_to_cigale,path_to_ini_file,filename='tmp'):


    fin = open(path_to_cigale+path_to_ini_file)
    fout = open(path_to_cigale+"pcigale.ini", "wt")
    for line in fin:
        ind_line=[param + " =" in line for param in parameter_names]
        if any(ind_line):

            param=parameter_names[np.array(ind_line)]
            fout.write("   "+param[0]+" = " + ", ".join(['{:.13f}'.format(i) for i in parameters[param[0]]]) + ' \n')
        else:
            fout.write(line)
    fin.close()
    fout.close()
    from shutil import copyfile, move, rmtree
    copyfile(path_to_cigale+path_to_ini_file+".spec",path_to_cigale+"pcigale.ini.spec")



    p = subprocess.Popen(['pcigale', 'run'], cwd=path_to_cigale)
    p.wait()
    rmtree(path_to_cigale+'{}/'.format(filename))
    move('/Volumes/pdh_storage/cigale/out/', '/Volumes/pdh_storage/cigale/{}/'.format(filename))
    SEDs = Table.read('/Volumes/pdh_storage/cigale/{}//models-block-0.fits'.format(filename))

    return SEDs