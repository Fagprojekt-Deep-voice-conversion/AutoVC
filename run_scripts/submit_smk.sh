#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J SMK
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- set span if number of cores is more than 1
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 10GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s183920@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o AutoVC_SMK_%J.out
#BSUB -e AutoVC_SMK_%J.err
# -- end of LSF options --


### Set directory

# BASE_DIRECTORY=$(echo "$PWD" | cut -d "/" -f2)
# if [ $BASE_DIRECTORY = "work1" ]
# then 
# 	run_dir=/work1/s183920/Deep_voice_conversion/AutoVC/run_scripts
# else
# 	run_dir=~/Desktop/Deep_voice_conversion/AutoVC/run_scripts
# fi

# run_dir=/work1/s183920/AutoVC/run_scripts
run_dir=/work1/s183920/AutoVC
# cd $run_dir


### Load modules
# module load python3
module load cuda

### Run setup
sh setup.sh $run_dir || exit 1
source ../AutoVC-env/bin/activate
echo $PWD

### Run python script
python3 ../Run_AutoVC.py --pretrained_model_path Models/AutoVC/autovc.ckpt --epochs 50 --test_size 24 --model_path_name AutoVC_SMK_20211020 --data_path data/SMK_train/ --loss_path_name SMK_20211020 --save_every 50000




