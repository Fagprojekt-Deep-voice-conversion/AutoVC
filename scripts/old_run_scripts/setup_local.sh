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
python -m venv test-env # assumed deafult python used is python 3.6

source test-env/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_local.txt
# python -m pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/cu92/torch_stable.html # torch with cuda 9.2
# python -m pip install sklearn tqdm librosa webrtcvad scipy matplotlib pandas seaborn wavenet_vocoder 
# python -m pip install numba==0.49.1 ###numba==0.43.0

deactivate


