#!/bin/bash

echo "\e[33mINFO: create_env.sh will create a virtual environment to use on the DTUs HPC cluster"

# set env name
env_name="AutoVC3-env"

if [ $( basename $PWD ) != "AutoVC" ]
then 
    echo "\e[33mWARN: Virtual environment about to be created without AutoVC as basename, instead it will be created at $PWD\e[0m" 
fi


### Make python environment
module load python3/3.8.9 # only use if on HPC
python3 -m venv $env_name

# source AutoVC-env/bin/activate
source $env_name/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .


if [ $( basename $PWD ) = "AutoVC" ]
then 
    echo "Virtual environment created at $PWD"
else
    echo "\e[33mWARN: Virtual environment was not created with AutoVC as basename, instead it was created at $PWD\e[0m"
fi

deactivate


