
import numpy as np
from mpdaf.obj import WaveCoord
from mpdaf.obj import Cube
from astropy.io import fits
from mpdaf.obj import WaveCoord
import matplotlib.pyplot as plt
import numpy as np
from mpdaf.obj import Cube, WaveCoord
from scipy.interpolate import interp1d
from mpdaf.obj import Cube, WaveCoord
import numpy as np
from copy import deepcopy
from astropy.io import fits
from mpdaf.obj import WCS as mpdaf_WCS
from mpdaf.obj import Cube, WaveCoord
from astropy.wcs import WCS
import numpy as np
from ioTools import read_in_data

def redshift_wavelength_axis(filename, z_obs, z_sim):
    """
    Redshift the wavelength axis of a cube from z_obs to z_sim,
    returning a new cube with unchanged data but redshifted wavelengths.
    Handles rectangular pixels correctly.

    Parameters
    ----------
    filename : str
        Path to the FITS cube.
    z_obs : float
        Original redshift of the source.
    z_sim : float
        New redshift to shift the cube to.

    Returns
    -------
    redshifted_cube : mpdaf.obj.Cube
        Cube with same data, mask, var, but redshifted wavelengths.
    lam_new : np.ndarray
        New wavelength array.
    """

    # --- load data & wavelength axis from FITS ---
    lam_old, data, *rest = read_in_data(filename)

    header = None
    var = None

    if len(rest) == 2:
        var, header = rest
    elif len(rest) == 1:
        header = rest[0]
    # if len(rest) == 0 → leave header=None, var=None


    # --- redshift wavelength axis ---
    lam_new = lam_old * (1 + z_sim) / (1 + z_obs)
    wave = WaveCoord(
        crval=lam_new[0],
        cdelt=lam_new[1] - lam_new[0],
        crpix=1,
        cunit='Angstrom',
        ctype='WAVE'
    )

    # --- build Cube ---
    redshifted_cube = Cube(data=data, var=var, header=header, wave=wave)


    # --- update header explicitly ---
    hdr = redshifted_cube.data_header
    hdr['CRVAL3'] = lam_new[0]
    hdr['CDELT3'] = lam_new[1] - lam_new[0]
    hdr['CTYPE3'] = 'VWAV'
    hdr['CUNIT3'] = 'Angstrom'
    hdr['REDSHIFT'] = z_sim

    # --- ensure spatial units exist ---
    if 'CUNIT1' not in hdr or not hdr['CUNIT1']:
        hdr['CUNIT1'] = 'arcsec'
    if 'CUNIT2' not in hdr or not hdr['CUNIT2']:
        hdr['CUNIT2'] = 'arcsec'

    # --- rebuild WCS from header ---

    redshifted_cube.wcs = mpdaf_WCS(hdr)

    return redshifted_cube, lam_new


def check_redshifted_cube(redshifted_cube):
    print("\n--- Redshifted Cube Check ---")
    print(f"Cube shape (λ, y, x): {redshifted_cube.data.shape}")

    wcs_obj = getattr(redshifted_cube, 'wcs', None)

    if wcs_obj is None:
        print("WCS is None")
    else:
        # Try to get CDELT, CRVAL, CRPIX from WCS header
        hdr = redshifted_cube.data_header
        for i, ax in enumerate(['x', 'y', 'λ']):
            crval = hdr.get(f'CRVAL{i+1}', None)
            crpix = hdr.get(f'CRPIX{i+1}', None)
            cdelt = hdr.get(f'CDELT{i+1}', None)
            cunit = hdr.get(f'CUNIT{i+1}', None)
            print(f"{ax.upper()}: CRVAL={crval}, CRPIX={crpix}, CDELT={cdelt}, CUNIT={cunit}")

    dx = getattr(redshifted_cube, 'dx_arcsec', None)
    dy = getattr(redshifted_cube, 'dy_arcsec', None)
    print(f"Stored pixel scales: dx_arcsec={dx}, dy_arcsec={dy}")

    if redshifted_cube.data.shape[1] < 2 or redshifted_cube.data.shape[2] < 2:
        print("Warning: Spatial dimensions are very small. Check WCS / CDELT values!")



if __name__ == "__main__":

    # Set redshift parameters


    # Load a test cube

    galaxy_name = "CGCG453"
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    z_obs = 0.025
    z_sim = 3
    

    # galaxy_name = "IRAS08"
    # file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/IRAS08/IRAS08_combined_final_metacube.fits"
    # z_obs = 0.019 
    # z_sim = 3

    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/{galaxy_name}_redshifted_cube_{z_sim}.fits"
    
    lam_old, data, _ = read_in_data_fits(file_path)


    # Run redshift function
    redshifted_cube, lam_new = redshift_wavelength_axis(file_path, z_obs, z_sim)

    # Print wavelength range before/after
    print(f"Original λ range: {lam_old[0]:.2f} – {lam_old[-1]:.2f}")
    print(f"Redshifted λ range: {lam_new[0]:.2f} – {lam_new[-1]:.2f}")
    check_redshifted_cube(redshifted_cube)
    print(f"--- Redshifted Cube Check ---")
    print(f"Cube shape (λ, y, x): {redshifted_cube.shape}")
    print("Min flux:", np.nanmin(data))
    print("Max flux:", np.nanmax(data))

    hdr = redshifted_cube.data_header
    print(f"X: CRVAL={hdr.get('CRVAL1')}, CRPIX={hdr.get('CRPIX1')}, "
        f"CDELT={hdr.get('CDELT1')}, CUNIT={hdr.get('CUNIT1')}")
    print(f"Y: CRVAL={hdr.get('CRVAL2')}, CRPIX={hdr.get('CRPIX2')}, "
        f"CDELT={hdr.get('CDELT2')}, CUNIT={hdr.get('CUNIT2')}")
    print(f"Λ: CRVAL={hdr.get('CRVAL3')}, CRPIX={hdr.get('CRPIX3')}, "
        f"CDELT={hdr.get('CDELT3')}, CUNIT={hdr.get('CUNIT3')}")



    # Plot spectrum before and after
    y, x = 12, 17  # Pick a spaxel
    original_spectrum = data[:, y, x]
    redshifted_spectrum = redshifted_cube.data[:, y, x]

    plt.figure(figsize=(10, 4))
    plt.plot(lam_old, original_spectrum, label='Original', alpha=0.7)
    plt.plot(lam_new, redshifted_spectrum, label='Redshifted', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title('Spectrum Before and After Redshifting')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  write redshifted cube to file
    redshifted_cube.write(output_path)
    print(f"redshifted cube has been stored in: {output_path}")
    data = fits.getdata(output_path)
    print("Cube shape:", data.shape)
    print("Min flux:", np.nanmin(data))
    print("Max flux:", np.nanmax(data))

