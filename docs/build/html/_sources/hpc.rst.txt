Large Fields
==========================

Running XID+ over large areas or fields is computationally challenging. One way to overcome this
issue is to split the field into tiles and run them independently. For HELP, we use a tiling scheme based on the `HEALPix <http://healpix.jpl.nasa.gov/>`_ scheme.

We carry out this tiling scheme as follows:

1. Create a prior class containing all raw data (e.g. map, catalogue and PSF) and construct 2 lists of tiles. `Here is an example script used for ELAIS-N1 <https://github.com/H-E-L-P/dmu_products/blob/master/dmu26/dmu26_XID%2BSPIRE_ELAIS-N1/XID%2BSPIRE_prior_SWIRE.ipynb>`_:



.. note:: The tiling scheme produces hierarchical larger tiles that are used to split up the main prior class into slightly smaller regions. The smaller tiles on which the fitting is actually done, uses the larger
 tiles as an input. This reduces the amount of memory required compared to having to read in the original prior class each time.

2. We create the larger tiles by running for each hierarcical tile::

    python -c 'from xidplus import HPC; HPC.hierarchical_tile("Master_prior.pkl","Tiles.pkl")'

By default, this assumes the command is being run as an array job on an HPC environment running SunGrid (as we do at Sussex), with the length of the array equal to the number of large tiles required.
If it cannot find the ``$SGE_TASK_ID``, then it will ask for task id number. It reads in the master prior, cuts it down to the tile being referenced by the task id number and saves the cut down prior.

3. Having created the hierarchical tiles, we actually run XID+ on each small tile. For HELP, we use an HPC to fit multiple tiles at the same time.

.. note:: Ideally, each tile being fitted should use four cores (so that each MCMC chain can be run independently).

4. Having carried out the fit to all the tiles, we can combine the Bayesian maps into one. `Here is an example script. <https://github.com/H-E-L-P/dmu_products/blob/master/dmu26/dmu26_XID%2BSPIRE_ELAIS-N1/make_combined_map.py>`_:
This will also pick up any failed tiles and list them in a failed_tiles.pkl
file, which you can then go back and fit.



.. toctree::
   :maxdepth: 2