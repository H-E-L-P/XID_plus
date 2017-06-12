Getting Started
===============

Installation
------------

XID+ requires a Python (2 or 3) installation. I recommend the `Anaconda distribution <https://www.continuum.io/downloads>`_.

Once you have a working Python installation, you can download the code either directly from `github <https://github.com/H-E-L-P/XID_plus>`_ or by cloning the git repository::

   git clone https://github.com/H-E-L-P/XID_plus.git
   cd XID_plus

Requirements
^^^^^^^^^^^^
If using an Anaconda distribution, then it is best to install some prerequisites with conda with the following command::

   while read requirement; do conda install --yes $requirement; done < req.txt

If you are using your own Python setup, install the requirements via the command ::

   pip install -r req.txt


Finally
^^^^^^^
Then, install the package by running the following command::

   pip install -e './'

.. note:: As part of the Herschel Extragalatic Legacy Project, we will be providing a `Docker <https://www.docker.com/>`_ image that will run all HELP tools and Jupyter notebooks, including XID+. Docker is an open source tool that allows developers to package up an application with all of the parts it needs, such as libraries and dependencies. The resulting Docker image can then be run on ANY machine, be it Windows, Linux, Mac or in the cloud without having to worry about installing numerous dependencies. This should be available from October 2017.


.. toctree::
   :maxdepth: 2

