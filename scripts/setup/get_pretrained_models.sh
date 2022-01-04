#!/bin/bash

# script for downloading the necesary models, only one AutoVC model is necesary
# script takes a model type and a model name as input
# type is specified with the -t (type) flag
# a specific model can be obtained with the -n (name) flag

# get input
while getopts t:n: flag
do
    case "${flag}" in
        t) model_type=${OPTARG};;
        n) model_name=${OPTARG};;
    esac
done

# WaveRNN - no model name is necesary as only one is available
WaveRNN_folder="models/WaveRNN/"
WaveRNN="1dtkRFL83Iya1wBt0ucBBb0Q4i5GlU6IF"

# Speaker Encoder - no model name is necesary as only one is available
SpeakerEncoder_folder="models/SpeakerEncoder/"
SpeakerEncoder="1j-M5KoqvJWJINJLXyhz403gTuZTm4kwV"

# AutoVC - model name is given after first underscore
AutoVC_folder="models/AutoVC/"
AutoVC_basic="1jKTxQUhBXNVi38C43YuDIg46QdAPgGrH"
AutoVC_origin="1Pjhk-lb9QW4EzsUSzlpns0NOIKrHoBlU"
AutoVC_SMK="1npx7nzdVapSbZg5wkHFFGbSvUPLstF_A"
AutoVC_seed40_200k="1ovdribZLkx1Wky5IHEt2_1Ibo9AI3jbt"


# onsider dict: https://www.xmodulo.com/key-value-dictionary-bash.html

if [ "$model_type" = "WaveRNN" ]; then
    gdown $WaveRNN --id -O $WaveRNN_folder
elif [ "$model_type" = "SpeakerEncoder" ]; then
    gdown $SpeakerEncoder --id -O $SpeakerEncoder_folder
elif [ "$model_type" = "AutoVC" ];then
    if [ "$model_name" = "basic" ]; then
        gdown $AutoVC_basic --id -O $AutoVC_folder
    elif [ "$model_name" = "origin" ]; then
        gdown $AutoVC_origin --id -O $AutoVC_folder
    elif [ "$model_name" = "SMK" ]; then
        gdown $AutoVC_SMK --id -O $AutoVC_folder
    elif [ "$model_name" = "seed40_200k" ]; then
        gdown $AutoVC_seed40_200k --id -O $AutoVC_folder
    elif [ "$model_name" = "" ]; then
        echo "\e[31mError: A model name has to be specified for AutoVC with the -n flag, eg. -n basic\e[0m"
    else
        echo "\e[31mError: $model_name is not a valid model name for AutoVC to download\e[0m"
    fi
elif [ "$model_type" = "" ]; then
    echo "\e[31mError: A model type has to be specified with the -t flag, eg. -t WaveRNN\e[0m"
else
    echo "\e[31mError: $model_type is not a valid model type to download\e[0m"
fi