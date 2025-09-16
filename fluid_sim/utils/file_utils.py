"""
File utility functions.
"""
import os


def create_dir_if_not_exists(dir_path: str):
    """
    Checks if a directory exists at the specified path and creates it if it doesn't.
    
    Parameters:
    -----------
    dir_path : str
        Path of the directory to check/create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)