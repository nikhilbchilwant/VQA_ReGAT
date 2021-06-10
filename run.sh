#!/bin/bash
#----------------------------------
# Specifying grid engine options
#----------------------------------
#$ -S /bin/bash
# the working directory where the commands below will
# be executed: (make sure to specify)
#$ -wd /data/users/nchilwant/projects/VQA_ReGAT
#
# logging files will go here: (make sure to specify)
#$ -e /data/users/nchilwant/log/ -o /data/users/nchilwant/log/
#
# Specify the node on which to run
#$ -l hostname=cl10lx
#----------------------------------
#  Running some bash commands
#----------------------------------
export PATH="/nethome/nchilwant/miniconda3/bin:$PATH"
source activate vqa
nvidia-smi
#----------------------------------
# Running your code (here we run some python script as an example)
#----------------------------------
pwd

echo "training ReGAT"
CUDA_VISIBLE_DEVICES=1,2,3 python main.py --config config/butd_vqa.json

echo "Finished execution."