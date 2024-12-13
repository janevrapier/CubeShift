from astropy.io import fits 
from astropy.wcs import WCS 

from mpdaf.obj import Cube, Image


class MyData:
    def __init__(self, filename: str) -> None:
        # make sure it's a string, not a list, or some other type 
        if type(filename) is not str:
            raise TypeError('filename must be str, not %s' % type(filename))
        
        # check the number of axes in the fits file
        ndim = _check_num_axes(filename)

        # depending on the number of axes, open as a cube or an image
        if ndim == 2:
            mydata = read_in_dataim(filename)
        elif ndim == 3:
            mydata = read_in_datacube(filename)

        self.data = mydata

    
    def _check_num_axes(filename: str) -> int:
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



def read_in_datacube(filename: str, ext: tuple[int, ...] = (1,2)) -> mpdaf.obj.cube.Cube:
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

def read_in_dataim(filename: str, ext: tuple[int, ...] = (1,2)) -> mpdaf.obj.image.Image:
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