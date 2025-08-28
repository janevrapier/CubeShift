from astropy.io import fits
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np



from astropy.io import fits
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


def read_in_data(filename):
    from mpdaf.obj import Cube
    cube = Cube(filename)

    data = cube.data
    lamdas = cube.wave.coord()  # this should be 1D, length = nz
    print(f"[DEBUG] Loaded cube with {data.shape[0]} spectral channels")
    print(f"[DEBUG] λ range: {lamdas[0]/1e4:.3f} – {lamdas[-1]/1e4:.3f} μm")  # convert Å to µm

    var = cube.var if cube.var is not None else None
    header = cube.primary_header

    if var is not None:
        return lamdas, data, var, header
    else:
        return lamdas, data, header



