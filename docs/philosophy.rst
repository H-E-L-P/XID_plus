A Generative Probabilistic Framework
=========================

Astronomical photometry methods are typically designed to construct catalogues from astronomical images. Constructing catalogues from images is a form of data compression. For high resolution imagery, a catalogue is good form of data compression. For confusion dominated maps it is not as the catalogues do not store the information that galaxies that are close together have correlated photometry measurements. Conventional methods also lose information by direct and constrained modelling of the physical parameters of each galaxy and then summarising those properties with theoretically (or historically) prejudiced assumptions. To preserve this information it is vital to constrain as close to the ``raw'' data as practical and be statistically rigorous with full characterisation of assumptions, uncertainty and degeneracies to exploit this information.

Bayesian Probabilistic Generative modelling satisfies those criteria and is the modelling approach on which XID+ is based.

Probabilisitic modelling
-----------------

Probabilistic generative modelling is an inference framework in  which all the elements of the modelling required to generate the observed data are represented probabilistically, and parameters of that model are inferred from the observed data. One key distinction between this and conventional model fitting is that a probabilistic model of the observed data (the data model) is included within the modelling rather than fitting the data with errors to a model.  This methodology has a number of important advantages over conventional techniques.  A primary advantage is that the uncertainties in the parameters from all aspects of the modelling, and the correlations between these, can be properly tracked. It is also a readily adaptable framework, allowing for incremental inclusion of more complexities into the model. However, it also has challenges, notably in the shear computational scale of inferring vast numbers of parameters (in this context multiplexed by the number of galaxies) on large data sets.

By necessity, HELP has had to develop new ways of analysing multi-wavelength data. The Herschel beam is broad (18 arcseconds at 250 microns) and thus severely blended so many galaxies may contribute to a single detected source (e.g.Scudder et al. 2016). However, the Herschel instruments are sufficiently sensitive and well behaved that the fluctuations within the maps provide information to constrain star formation at faint levels.


XID+
---------
Our philosophy with XID+ is to build a generative model of the confusion dominated maps. The basic model describes the maps with parameters for the flux.

However, it has been built to be extended so that more intricate models can be bolted on top of this basic model, hence the +. For example, we could add an extension to the model so that the fluxes are constrained by certain types of spectral energy distributions, giving us XID+SEDs.

XID+ is built using both Python and Stan.

Python
^^^^^^^^
The general interface with XID+ is through Python. It is here we provide the maps, point spread functions, catalogues and any other additional prior information. We also use PyStan, the Python interface to Stan. This passes our prior data to Stan and carry out the fitting. We have written other useful functions including
cutting downdata to specfic regions, through to plotting functions for visualising the output.

Stan
^^^^^^^
Our probabilistic generative models are built using Stan, the probabilistic programming language. At its core, Stan has an advanced Hamiltonian Monte Carlo inference engine.
This allows us to carry out inference on models with thousands of parameters and is well behaved so we can diagnose when there are problems.

We provide Stan models for carrying out XID+ on SPIRE, PACS and MIPS maps. These models can be extended for specific science cases, however it is up to the user to determine whether the Stan model is capable of being inferred. This can be done using the `Bayesian robust workflows <http://mc-stan.org/users/documentation/case-studies/pystan_workflow.html>`_ described by Stan developers.


.. toctree::
   :maxdepth: 2

