Large Fields
==========================

Running XID+ over large areas or fields is computationally challenging. One way to overcome this
issue is to split the field into tiles and run them independently. For HELP, we use a tiling scheme based on the `HEALPix <http://healpix.jpl.nasa.gov/>`_ scheme.

We carry out this tiling scheme as follows:

1. Create a prior class containing all raw data (e.g. map, catalogue and PSF)

2. Use that prior class catalogue to construct 2 lists of tiles ::

    from xidplus import moc_routines
    order=11
    tiles=moc_routines.get_HEALPix_pixels(order,priors[0].sra,priors[0].sdec,unique=True)
    order_large=7
    tiles_large=moc_routines.get_HEALPix_pixels(order_large,prior250.sra,prior250.sdec,unique=True)
    print('----- There are '+str(len(tiles))+' tiles required for input catalogue and '+str(len(tiles_large))+' large tiles')
    output_folder='./'
    outfile=output_folder+'Master_prior.pkl'
    with open(outfile, 'wb') as f:
    pickle.dump({'priors':,priors,'tiles':tiles,'order':order,'version':xidplus.io.git_version()},f)
    outfile=output_folder+'Tiles.pkl'
    with open(outfile, 'wb') as f:
    pickle.dump({'tiles':tiles,'order':order,'tiles_large':tiles_large,'order_large':order_large,'version':xidplus.io.git_version()},f)
    raise SystemExit()


.. note:: The larger tiles are used to split up the main prior class into slightly smaller regions. The smaller tiles use the larger
 tiles as an input. This reduces the amount of memory required compared to having to read in the original prior class each time.

3. We create the larger tiles by running::

    python -c 'from xidplus import HPC; HPC.hierarchical_tile("Master_prior.pkl","Tiles.pkl")'

By default, this assumes the command is being run as an array job on an HPC environment running SunGrid (as we do at Sussex), with the length of the array equal to the number of large tiles required.
If it cannot find the ``$SGE_TASK_ID``, then it will ask for task id number. It reads in the master prior, cuts it down to the tile being referenced by the task id number and saves the cut down prior.

4. The final stage is to actually run XID+ on each small tile. For HELP, we use an HPC to fit multiple tiles at the same time.

.. note:: Ideally, each tile being fitted should use four cores (so that each MCMC chain can be run independently).









.. toctree::
   :maxdepth: 2