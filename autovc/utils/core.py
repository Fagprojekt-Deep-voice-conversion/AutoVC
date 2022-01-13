import os


def retrieve_file_paths(paths, excluded = []):
    """
    Takes a path and returns all files in the folder and subfolders
    
    Params
    ------
    paths:
        if it is a directory the files in this path will be returned as a list
    """
    if excluded != []:
        excluded = retrieve_file_paths(excluded)

    if isinstance(paths, str):
        if os.path.isfile(paths):
            return [paths] if paths not in excluded else []
        
        elif os.path.isdir(paths):
            walk = os.walk(paths)
            paths = list()
            
            for (dirpath, dirnames, filenames) in walk:
                paths += [os.path.join(dirpath, file).replace("\\", "/") for file in filenames]
        
        else:
            raise ValueError(f"No file or a directory named {paths}")
    
    if isinstance(paths, list):
        paths = sum([retrieve_file_paths(path, excluded = excluded) for path in paths], [])

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


if __name__ == "__main__":
    print(retrieve_file_paths(["data/samples"], excluded="data/samples/test/chooped7.wav"))
    # print(calls)



# print(np.shape(retrieve_file_paths("data/SMK_train/newest_trial")))