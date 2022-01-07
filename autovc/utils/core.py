import os


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
            walk = [w for w in os.walk(paths)][0]
            root, _, data = [w for w in walk]
            paths = [root + "/" + d for d in data]

    return paths