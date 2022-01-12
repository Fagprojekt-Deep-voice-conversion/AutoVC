import os
import numpy as np


def retrieve_file_paths(paths):
    """
    Takes a path and returns all files in the folder and subfolders
    
    Params
    ------
    path:
        if it is a directory the files in this path will be returned as a list
    """

    if isinstance(paths, str):
        if os.path.isfile(paths):
            return [paths]
        
        if os.path.isdir(paths):
            

            walks = [w for w in os.walk(paths)]#[0]
            paths = []
            # return walk
            # print(walk)
            for walk in walks:
                root, _, data = [w for w in walk]
                paths.extend([root + "/" + d for d in data])

    return paths

class pformat:
   """Class to contain different values for printing in different formats"""
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



# print(np.shape(retrieve_file_paths("data/SMK_train/newest_trial")))