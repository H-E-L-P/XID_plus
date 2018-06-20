Installation
===============

XID+ requires a Python (2 or 3) installation. I recommend the `Anaconda distribution <https://www.continuum.io/downloads>`_.

Once you have a working Python installation, you can download the code either directly from `github <https://github.com/H-E-L-P/XID_plus>`_ or by cloning the git repository::

   git clone https://github.com/H-E-L-P/XID_plus.git
   cd XID_plus

Requirements
^^^^^^^^^^^^
Anaconda Python
"""""""""""""""
If using an Anaconda distribution, then it is best to install some prerequisites with conda with the following command::

   conda config --add channels conda-forge && conda install healpy
   while read requirement; do conda install --yes $requirement; done < req.txt


Custom Python
"""""""""""""
If you are using your own Python setup, install the requirements via the command ::

   pip install -r req.txt


Finally
^^^^^^^
Then, install the package by running the following command::

   pip install -e './'


Docker
======
As an alternative to having to install XID_plus, along with all the Python dependencies, we provide a docker image of XID plus.

Docker is an open source tool that allows developers to package up an application with all of the parts it needs, such as libraries and dependencies.

The resulting Docker image can then be run on ANY machine, be it Windows, Linux, Mac or in the cloud without having to worry about installing numerous dependencies.

For Docker installation instructions, `see the Docker main page <https://www.docker.com/get-docker>`_.

Once Docker is installed, running a docker image on a Linux or Mac OS is very simple. Just open a new terminal and use the command::

   docker run -it --rm -v $(pwd):/home -p 8888:8888 pdh21/xidplus:latest


Once the docker image is downloaded, open the shown URL link in your browser and you are good to go. The URL will look something like:
http://localhost:8888/?token=0312c1ef3b61d7a44ff5346d3d150c23249a548850e13868

Our Docker image has been created to run Jupyter notebook at startup.

The different flags do the following:

* The -it flag tells docker that it should open an interactive container instance.
* The --rm flag tells docker that the container should automatically be removed after we close docker.
* The -p flag specifies which port we want to make available for docker.
* The -v flag tells docker which folders should be mount to make them accesible inside the container.
Here: $(pwd) is your local directory you are running the Docker image from. The second part of the -v flag (here: /home) specifies under which path the mounted folders can be found inside the container.

Alternatively, rather than starting with the default Jupyter notebook, you can also access the Docker container directly with bash using the following command::

    docker run -it --rm -v $(pwd):/home -p 8888:8888 pdh21/xidplus:latest /bin/bash



.. toctree::
   :maxdepth: 2

