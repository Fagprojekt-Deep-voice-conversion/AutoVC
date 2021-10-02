#!/bin/bash

WaveRNN="1dtkRFL83Iya1wBt0ucBBb0Q4i5GlU6IF"

SpeakerEncoder="1j-M5KoqvJWJINJLXyhz403gTuZTm4kwV"

AutoVC_basic="1jKTxQUhBXNVi38C43YuDIg46QdAPgGrH"
AutoVC_origin="1Pjhk-lb9QW4EzsUSzlpns0NOIKrHoBlU"
AutoVC_SMK="1npx7nzdVapSbZg5wkHFFGbSvUPLstF_A"
AutoVC_seed40_200k="1ovdribZLkx1Wky5IHEt2_1Ibo9AI3jbt"


# choice=$( echo "$1" )
# echo $choice
# onsider dict: https://www.xmodulo.com/key-value-dictionary-bash.html

if [ "$1" = "WaveRNN" ]; then
    gdown $WaveRNN --id -O "models/WaveRNN/"
    # gdown $WaveRNN --id
elif [ "$1" = "SpeakerEncoder" ]; then
    gdown $SpeakerEncoder --id -O "models/SpeakerEncoder/"
else
    echo "\e[31mError: $1 is not a valid model name to download\e[0m"
fi