#!/bin/sh

#  assert correct run dir
run_dir="AutoVC"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

# create dir for logs
mkdir -p "logs/hpc"

### General options
### –- specify queue --
#BSUB -q queue_name
### -- set the job Name --
#BSUB -J TemplateJob
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now 
#BSUB -W 8:00 
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
##BSUB -u studentnumber@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion-- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o status_file_name_%J.out 
#BSUB -e status_file_name_%J.err 
### -- end of LSF options --

# activate env
source AutoVC-env/bin/activate

# load additional modules
module load cuda/11.4

# run scripts
