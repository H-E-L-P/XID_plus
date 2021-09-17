FROM continuumio/miniconda3

LABEL maintainer "p.d.hurley@sussex.ac.uk"
RUN apt-get update && apt-get -y install libgl1-mesa-glx && apt-get -y install build-essential
RUN conda update conda && git clone https://github.com/H-E-L-P/XID_plus.git
WORKDIR XID_plus
RUN conda config --add channels conda-forge && conda install healpy
RUN while read requirement; do conda install --yes $requirement; done < req.txt
RUN pip install -r req.txt && pip install -e './'
RUN conda update

