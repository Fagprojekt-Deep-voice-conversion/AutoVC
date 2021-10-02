#!/bin/bash

###cd ~/*/Deep_voice_conversion/AutoVC/run_scripts
### Set directory

if [ "$1" = "" ]
then 
	echo -e "\e[31mMissing a directory to do the setup in\e[0m"
	exit 1
else
	echo "$1"	
	cd $1
fi

if [ ! -d "$1" ] 
then
	echo -e "\e[31mSetup directory does not exist!\e[0m"
	exit 1
fi



### Make python environment
module load python3 # only use if on HPC
# python3 -m venv AutoVC-env
python3 -m venv test-env

# source AutoVC-env/bin/activate
source test-env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

deactivate


