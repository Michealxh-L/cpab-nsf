#!/bin/bash
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J image-generation
#BSUB -R "rusage[mem=6GB]"
#BSUB -W 24:00
#BSUB -u s202724@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

# output file
#mkdir -p ./output/
#rm -f ./output/loss_hist.text
#OUTFILE = "./output/loss_hist.text"

#load modules
# pip install torch
# pip install nflows
source ../my_nflows/my_flows-env/bin/activate
module load binutils/2.34
module load gcc/8.4.0
module load python3/3.9.11
module load cuda/11.6
module load ninja
#unset PYTHONPATH
#unset PYTHONHOME

# module avail
# python -m pip install --user torchvision

#python experiments/images.py with experiments/image_configs/cpab-cifar-10-8bit.json
python experiments/images.py eval_on_test with experiments/image_configs/cpab-cifar-10-8bit.json
python experiments/images.py sample with experiments/image_configs/cpab-cifar-10-8bit.json


nvidia-smi
#load the cuda module
#module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
