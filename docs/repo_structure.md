# Data structure

This repo has the msin structure seen below (some folders and files are deprecated). A short description is also provided.


```
AutoVC
    ├── autovc/
    │   ├── auto_encoder/
    │   ├── vocoder/
    │   ├── utils/
    │   └── ...
    ├── data/
    │   ├── VCTK-data/ 
    │   │   ├── VCTK-data/
    │   │   └── ...
    │   ├── samples/ 
    │   └── ...
    ├── docs/
    │   ├── repo_structure.md
    │   ├── clean_up_tasks.md
    │   └── ...
    ├── models/
    │   ├── AutoVC/
    │   │   ├── autovc.ckpt
    │   │   └── ...
    │   ├── SpeakerEncoder/
    │   │   ├── SpeakerEncoder.pt
    │   │   └── ...
    │   ├── WaveRNN/
    │   │   ├── WaveRNN_Pretrained.pyt
    │   │   └── ...
    ├── results/
    ├── scripts/
    │   ├── setup
    │   │   ├── create_env.sh
    │   │   └── ...
    │   └── ...
    ├── .gitignore
    ├── requirements.txt
    ├── README.md
    └── ...
```

## autovc

This is the package folder and contains all the code files which are used to convert voices.

## data

This folder contains all the data used for conversion. This is where you should put the sound files that you want to convert, e.g. the VCTK corpus. Only some small samples used for examples are provided.

## docs

Contains various markdown files regarding how to use the package and repository and more.

## models

The folder where pretrained models should be put and trained models are stored.

## results

The folder where the resulting conversions and loss files are saved.

## scripts

Various scripts for different purposes, e.g. the `create_env.sh` script, which creates a virtual environment.