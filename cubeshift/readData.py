import numpy as np 

from astropy.io import fits 
from astropy.wcs import WCS 

from mpdaf.obj import Cube, Image


class MyData:
    def __init__(self, filename: str, ext: tuple[int, ...] = (1,2)) -> Cube|Image:
        # make sure it's a string, not a list, or some other type 
        if type(filename) is not str:
            raise TypeError('filename must be str, not %s' % type(filename))
        
        # check the number of axes in the fits file
        self.ndim = self._check_num_axes(filename)

        # depending on the number of axes, open as a cube or an image
        if self.ndim == 2:
            mydata = read_in_dataim(filename, ext=ext)
        elif self.ndim == 3:
            mydata = read_in_datacube(filename, ext=ext)

        self.mydata = mydata

    
    def _check_num_axes(self, filename: str) -> int:
        """Checks the number of axes in the data in the fits file.

        Parameters
        ----------
        filename : str
            The fits file location

        Returns
        -------
        int
            the number of axes in the data.
        """
        hdr = fits.getheader(filename, ext=0)
        ndim = hdr['NAXIS']
        if ndim == 0:
            hdr = fits.getheader(filename, ext=1)
            ndim = hdr['NAXIS']

        return ndim
    
    
    def milky_way_extinction_correction(self, Av: float = 0.031, inplace: bool = False):
        """
        Corrects for the foreground galactic extinction caused by light travelling 
        through the dust and gas of the Milky Way, as described in 
        Cardelli et al. 1989.

        Parameters
        ----------
        Av : float 
            The extinction from the Milky Way, found using NED which uses the 
            Schlafly & Finkbeiner (2011) recalibration of the Schlegel, Finkbeiner 
            & Davis (1998) extinction map based on dust emission measured by 
            COBE/DIRBE and IRAS/ISSA.
            Default is 0.031, which is the value for NGC300 using the ACS clear
            filter, which is centred at 0.6211um.
        inplace : bool
            If False, return the data corrected for Milky Way extinction.
            If True, correct the data for Milky Way extinction inplace, and return
            the original cube.  Default is False.

        Returns
        -------
        data : :obj:'~numpy.ndarray'
            the data corrected for extinction
        """
        # create a copy of the input data if the user doesn't want to overwrite
        # the current cube 
        res = self if inplace else self.copy()

        # convert lamdas from Angstroms into micrometers
        lamdas = res.mydata.wave.coord()
        lamdas = lamdas/10000

        # define the equations from the paper
        y = lamdas**(-1) - 1.82
        a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
        b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

        # define the constants
        Rv = 3.1

        # find A(lambda)
        A_lam = (a_x + b_x/Rv)*Av

        # apply to the data
        data = (10**(0.4*A_lam[:, None, None]))*res.mydata.data

        # update res so that inplace=True works
        res.mydata.data = data 

        return res 
    



def read_in_datacube(filename: str, ext: tuple[int, ...] = (1,2)) -> Cube:
    """Reads in the data from a filename to an mpdaf Cube

    Parameters
    ----------
    filename : str
        the location of the file to read in, should be a fits cube
    ext : int or (int, int) or str or (str, str)
        the optional number/name of the data extension or the numbers/names of 
        the data and variance extensions

    Returns
    -------
    `mpdaf.obj.Cube` obj
        A data cube
    """
    datacube = Cube(filename, ext=ext)

    return datacube 

def read_in_dataim(filename: str, ext: tuple[int, ...] = (1,2)) -> Image:
    """Reads in the data from a filename, into an mpdaf Image

    Parameters
    ----------
    filename : str
        the location of the file to read in, should be a fits cube
    ext : int or (int, int) or str or (str, str)
        the optional number/name of the data extension or the numbers/names of 
        the data and variance extensions

    Returns
    -------
    `mpdaf.obj.Image` obj
        A data image
    """
    dataim = Image(filename, ext=ext)

    return dataim