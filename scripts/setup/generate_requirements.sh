#!/bin/sh

# activate env
source AutoVC-env/bin/activate

# generate requirements.txt
python -m pip install pipreqs # package to autogenerate requirements.txt
pipreqs autovc --savepath ./requirements_autogenerated.txt #will overwrite requirementsfile