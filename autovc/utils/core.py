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

# print(np.shape(retrieve_file_paths("data/SMK_train/newest_trial")))