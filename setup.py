import autovc
from setuptools import setup

setup(name='autovc',
      version='0.0.1',
      # install_requires=['pandas', 'numpy', 'torch'],  # And any other dependencies foo needs
      package_data = {'gym_cool_data':[  # remember to include files in a file called MANIFEST.in
        #   "gym_cool_data/data/cooling_reports/*.csv",
      ]},
)