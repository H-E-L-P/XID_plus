
##Synopsis

XID+ is the next generation deblender tool for Herschel SPIRE maps. Its uses a probabilistic framework which allows the use prior information about the sources.

##Code Example
XIDp_mod_beta.py is the main module file containing classes, definitions and functions.

stan_models/* contains the stan models called by XID_plus_mod_beta.py. 

A very simple example can be found in ./scripts/test_run/Lacey_COSMOS_test.py, with test maps and catalogues in ./test_files/*. Run as follows:
`python Lacey_COSMOS_test.py`

This will generate a compiled stan model (XID+SPIRE.pkl), which can be used for other runs. The main output is stored in Lacy_test_file.pkl

An example script using the tiling scheme on an hpc, can be found in:
./scripts/run_scripts/Lacy_COSMOS_XIDp_SPIRE_beta_test.py
NOTE: if not running on hpc, you can define your own simple tile, where a tile is simply a transposed numpy array, with ra and dec co-ordinates for the four corners:
 e.g. np.array([[ra,dec],[ra+tile_l,dec],[ra+tile_l,dec+tile_l],[ra,dec+tile_l]]).T
 

##Installation

Requires the following python modules (all can be installed via pip)
1. pystan
2. dill
3. pickle
4. numpy
5. scipy
6. astropy


##Tests

Describe and show how to run the tests with code examples.

##Contributors

This code is being developed by Dr Peter Hurley. 

##License

