import subprocess
import os
import numpy as np
from astropy.table import Table


def generate_SED(parameter_names,path_to_cigale,path_to_ini_file,filename='tmp',output_table_path='CIGALE_sed.fits'):
    # open the template cigale file
    fin = open(path_to_cigale + path_to_ini_file)
    # open the standard pcigale ini file to copy edits to
    fout = open(path_to_cigale + "pcigale.ini", "wt")
    # for each line
    for line in fin:
        # check if the line match any parameter names
        ind_line = [param + " =" in line for param in parameter_names]

        if any(ind_line):
            param = parameter_names[onp.array(ind_line)]
            # if parameter name is fracAGN check if this is the first
            if param[0] == 'fracAGN':
                if fracagn:
                    fout.write(line)
                    fracagn = False
                else:
                    # otherwise write out parameter values
                    fout.write("   " + param[0] + " = " + ", ".join(
                            ['{:.13f}'.format(i) for i in parameters_tmp[param[0]]]) + ' \n')
                    fracagn = True
            else:
                fout.write("   " + param[0] + " = " + ", ".join(
                        ['{:.13f}'.format(i) for i in parameters_tmp[param[0]]]) + ' \n')
        else:
            fout.write(line)

    # close files
    fin.close()
    fout.close()

    from shutil import copyfile, move, rmtree
    # copy corresponding ini.spec file to standard path
    copyfile(path_to_cigale + path_to_ini_file + ".spec", path_to_cigale + "pcigale.ini.spec")
    # run cigale
    p = subprocess.Popen(['pcigale', 'run'], cwd=path_to_cigale)
    p.wait()
    # check if folder already exists
    try:
        rmtree(path_to_cigale + '{}/'.format(filename))
    except(FileNotFoundError):
        print('---')
    # move cigale output to folder
    move(path_to_cigale + '/out/', path_to_cigale + '/{}/'.format(filename))
    # read in SEDs
    SEDs = Table.read(path_to_cigale + '/{}//models-block-0.fits'.format(filename))
    # change units
    SEDs['dust.luminosity'] = SEDs['dust.luminosity'] / L_sun.value
    # repeat the SED table by the number of scale steps
    dataset = vstack([SEDs for i in range(0, parameters_tmp['sfr'].size)])
    # repeat the scale range by the number of entries in table (so I can easily multiply each column)
    scale_table = onp.repeat(parameters_tmp['sfr'], len(SEDs)) / dataset['sfh.sfr']
    # scale each column that should be scaled as SFR is scaled
    for c in col_scale:
        dataset[c] = dataset[c] * scale_table
    # create log10 version of SFR
    dataset['log10_sfh.sfr'] = onp.log10(dataset['sfh.sfr'])
    # write out scaled file
    dataset.write(output_table_path, overwrite=True)