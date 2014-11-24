#!/bin/bash
#$ -S /bin/bash
#$ -o /mnt/lustre/scratch/inf/pdh21/log/out
#$ -e /mnt/lustre/scratch/inf/pdh21/log/err
cd /research/astro/fir/HELP/XID_plus/
echo "this is from the run script"
#module load sl5/qlogic/qlc/intel/1.2.7
#module load sl5/qlogic/openmpi/intel/1.4.3
#module load mpiexec/0.84_432
#module load intel-mpi/mic/4.1.0/030
#module load sl5/python/2.6.6
module load python/current
module load gcc/4.8.1
python COSMOS_XIDp_SPIRE_beta_test.py