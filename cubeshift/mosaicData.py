## adapted from Anna McLeod's script for mosaicking MUSE cubes

import numpy as np
import glob
from pathlib import Path

import montage_wrapper as m

from astropy import units
from astropy.io import fits


class MosaicData:
    def __init__(self, files: str|list[str], output_dir: str) -> None:
        
        # if the input is a list, check that the components of the list are all
        # existing files
        if type(files) is list:
            for file in files:
                if Path(file).is_file() is False:
                    raise FileNotFoundError('File does not exist: %s' %file)
        # if the input is a str, check that it is a directory and make a list 
        # of all the files in the directory
        elif type(files) is str:
            if Path(files).is_dir() is False:
                raise FileNotFoundError('Input directory does not exist, or is not a directory: %s' %files)
            elif Path(files).is_dir() is True:
                files = glob.glob(files)
                if len(files) < 1:
                    raise FileNotFoundError('No files in the input directory')
                
        self.list_of_files = files 
        self.output_dir = output_dir

    def make_mosaic(self, lam_min, lam_max, mosaic_name):

        # check out memmap!!!

        # limit the wavelength range of all the cubes

        # save them and create a new list of the limited cubes

        # extract the individual slices from each cube and save them 

        # get a list of all the slices

        # for each slice, mosaic it with the same slice from the other cubes

        # create a list of each slice mosaic

        # reconstruct the entire cube
        




