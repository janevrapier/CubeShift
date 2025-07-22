
import numpy as np
from mpdaf.obj import WaveCoord
from mpdaf.obj import Cube
from astropy.io import fits
from mpdaf.obj import WaveCoord
import matplotlib.pyplot as plt


# not working, ignore this
"""
def redshift_wavelength_axis(cube, z_old, z_new):

    Shift the spectral axis of an MPDAF Cube to simulate redshift.

    This scales the wavelength axis by (1+z_new)/(1+z_old), updating
    CRVAL3 and CDELT3 in the FITS header, and returns a new Cube.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The input data cube at original redshift z_old.
    z_old : float
        Original redshift of the cube.
    z_new : float
        Target redshift to simulate.

    Returns
    -------
    mpdaf.obj.Cube
        New cube with the wavelength axis redshifted and header updated.


    # Compute scale factor for wavelength axis
    scale_factor = (1 + z_new) / (1 + z_old)

    # Copy original FITS header to modify wavelength axis info
    header = cube.primary_header.copy()

    # Update starting wavelength (CRVAL3)
    header['CRVAL3'] *= scale_factor

    # Update wavelength step size (CDELT3)
    header['CDELT3'] *= scale_factor

    # Optionally add/update a keyword recording simulated redshift
    header['Z_SIM'] = (z_new, 'Simulated redshift')

    # Existing data shifting code...
    data = cube.data.copy()
    lam_old = cube.wave.coord()
    lam_new = lam_old * (1 + z_new) / (1 + z_old)

    # New: carry over mask and variance
    mask = cube.mask
    var = cube.var

    # Create new Cube and set wave
    new_cube = Cube(data=data, mask=mask, var=var)
    from mpdaf.obj import WaveCoord

    wave = WaveCoord(start=lam_new[0], step=lam_new[1] - lam_new[0], nsteps=len(lam_new))
    new_cube.wave = wave


    return new_cube
    """



# NOTE this version does not write to header -- fixed i think!
def redshift_wavelength_axis(cube, z_obs, z_sim):
    """
    Redshift the wavelength axis of a cube from z_old to z_new,
    returning a new cube with unchanged data, but redshifted wavelengths.

    Parameters:
    - cube (mpdaf.obj.Cube): The input data cube.
    - z_old (float): Original redshift of the source.
    - z_new (float): New redshift to shift the cube to.

    Returns:
    - redshifted_cube (mpdaf.obj.Cube): Cube with same data, mask, var.
    - lam_new (np.ndarray): New redshifted wavelength array.
    """

    hdr = cube.data_header
    print(hdr['CRVAL3'], hdr['CDELT3'], hdr['CTYPE3'], hdr['CUNIT3'])

    lam_old = cube.wave.coord()
    lam_new = lam_old * (1 + z_sim) / (1 + z_obs)

    # Build new WaveCoord with same metadata but new wavelengths
    wave = WaveCoord(crval=lam_new[0],
                     cdelt=lam_new[1] - lam_new[0],
                     crpix=1,
                     cunit='Angstrom',
                     ctype='WAVE')

    # Copy data, mask, var
    data = cube.data.copy()
    mask = cube.mask.copy() if cube.mask is not None else None
    var = cube.var.copy() if cube.var is not None else None
    header = cube.data_header.copy()

    # Create new cube with redshifted wavelength axis
    redshifted_cube = Cube(data=data, mask=mask, var=var, header=header, wave=wave)

    # Update FITS header explicitly
    hdr = redshifted_cube.data_header
    hdr['CRVAL3'] = lam_new[0]
    hdr['CDELT3'] = lam_new[1] - lam_new[0]
    hdr['CTYPE3'] = 'WAVE'
    hdr['CUNIT3'] = 'Angstrom'
    hdr['REDSHIFT'] = z_sim
    redshifted_cube.wcs = cube.wcs.copy()

    print("Returning redshifted cube")
    
    return redshifted_cube, lam_new

if __name__ == "__main__":


    # Load a test cube
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    # Set redshift parameters
    z_obs = 0.025
    z_sim = 2.0

    # Run redshift function
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, z_obs, z_sim)

    # Print wavelength range before/after
    lam_old = cube.wave.coord()
    print(f"Original λ range: {lam_old[0]:.2f} – {lam_old[-1]:.2f}")
    print(f"Redshifted λ range: {lam_new[0]:.2f} – {lam_new[-1]:.2f}")

    # Plot spectrum before and after
    y, x = 50, 50  # Pick a spaxel
    original_spectrum = cube.data[:, y, x]
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

    # Optional: write redshifted cube to file
    redshifted_cube.write('test_redshifted_cube.fits')
    print("redshifted cube has been stored in: test_redshifted_cube.fits")
