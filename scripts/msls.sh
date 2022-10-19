#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J testjob
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ../bsub_logs/gpu_%J.out
#BSUB -e ../bsub_logs/gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.3
module load gcc/9.2.0 
source activate metric_learning

python run.py --config ../configs/msls/deterministic.yaml
python run.py --config ../configs/msls/pfe.yaml
###python run.py --config ../configs/msls/laplace_posthoc_fix.yaml
###python run.py --config ../configs/msls/laplace_posthoc_full.yaml
###python run.py --config ../configs/msls/laplace_posthoc_pos.yaml
###python run.py --config ../configs/msls/laplace_online_fix.yaml
###python run.py --config ../configs/msls/laplace_online_full.yaml
###python run.py --config ../configs/msls/laplace_online_pos.yaml