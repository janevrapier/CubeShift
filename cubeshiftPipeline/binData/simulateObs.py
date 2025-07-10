"""
simulateObs.py

This module simulates how a low-redshift IFU data cube (e.g. from Keck/KCWI) would appear 
if observed with a different telescope (e.g. JWST NIRCam) at higher redshift.

It includes a full simulation pipeline that:
    1. Rebins the cube to lower spatial resolution.
    2. Rescales observed luminosity to account for angular diameter distance effects.
    3. Convolves each spatial slice with a Gaussian PSF (instrument resolution).
    4. Adds observational Gaussian noise.

These tools are used to investigate how galaxy features—like outflows or emission structure—
would be altered by distance and instrument limitations. This is useful for comparing local
galaxies with their high-redshift analogs or for testing whether scaling relations (e.g. 
between outflow velocity and SFR) are preserved across redshift.

Main functions:
---------------
- scale_luminosity_for_redshift()  — Adjusts spaxel fluxes based on physical area scaling.
- apply_psf()                      — Smooths spatial resolution using a Gaussian kernel.
- add_noise()                      — Adds Gaussian noise across all wavelengths.
- generate_mock_jwst_cube()       — Full pipeline: bin, scale, smooth, and add noise.

Example usage (in `__main__`):
-------------------------------
- Load a low-z FITS cube.
- Simulate appearance at z=1.0 under JWST resolution.
- Plot and compare slices before and after transformation.
"""


import numpy as np
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from binData import bin_cube
from astropy.convolution import convolve, Gaussian2DKernel


def scale_luminosity_for_redshift(cube, redshift_old, redshift_new):
    """
    Rescale the observed luminosity of each spaxel based on change in angular scale
    due to redshift, assuming the intrinsic (actual) luminosity is fixed.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The data cube observed at redshift_old.
    redshift_old : float
        The redshift at which the cube was originally observed.
    redshift_new : float
        The redshift you want to simulate (e.g. a more distant galaxy).
    
    Returns
    -------
    new_data : numpy.ndarray
        The new spaxel luminosities as they would appear at redshift_new.
    """
    # Get the angular size in arcsec per pixel (assume square pixels)
    # NOTE: Change to not assume square pixels 
    arcsec_per_pixel = cube.wcs.get_axis_increments(unit=u.arcsec)[0]

    # Get proper distances per arcsec at both redshifts
    proper_dist_old = cosmo.kpc_proper_per_arcmin(redshift_old).to(u.kpc/u.arcsec)
    proper_dist_new = cosmo.kpc_proper_per_arcmin(redshift_new).to(u.kpc/u.arcsec)

    # Compute spaxel area at each redshift (in kpc^2)
    #
    spaxel_area_old = (arcsec_per_pixel * proper_dist_old)**2
    spaxel_area_new = (arcsec_per_pixel * proper_dist_new)**2

    # Recover intrinsic surface brightness (luminosity per area)
    lum_per_area = cube.data / spaxel_area_old.value

    # Apply new spaxel area to simulate observation at new redshift
    new_data = lum_per_area * spaxel_area_new.value

    return new_data




def apply_psf(cube, fwhm_arcsec, pixel_scale_arcsec):
    """
    Convolve spatial dimensions with a Gaussian PSF in a NaN-aware way.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The input data cube.
    fwhm_arcsec : float
        Full width at half maximum of the target PSF, in arcseconds.
    pixel_scale_arcsec : float
        Pixel scale of the cube in arcseconds/pixel.

    Returns
    -------
    cube : mpdaf.obj.Cube
        Cube with data convolved by a Gaussian PSF.
    """
    # Convert FWHM to sigma (stddev in pixels)
    sigma_pixels = fwhm_arcsec / (2.355 * pixel_scale_arcsec)
    kernel = Gaussian2DKernel(x_stddev=sigma_pixels)

    new_data = np.empty_like(cube.data)

    # Apply convolution slice by slice
    for i in range(cube.shape[0]):  # loop over wavelength axis
        slice_2d = cube.data[i]
        new_data[i] = convolve(
            slice_2d,
            kernel,
            boundary='extend',               # extend edges to avoid artifacts
            nan_treatment='interpolate',     # interpolate across NaNs
            preserve_nan=True                # preserve NaNs if they dominate
        )

    cube.data = new_data
    return cube

# we already have noise 
def add_noise(cube, noise_level):
    """
    Add Gaussian noise to the cube data.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
    noise_level : float
        Standard deviation of Gaussian noise to add.

    Returns
    -------
    noisy_cube : mpdaf.obj.Cube
    """
    noisy_data = cube.data + np.random.normal(0, noise_level, size=cube.data.shape)
    cube.data = noisy_data
    return cube

def generate_mock_jwst_cube(original_cube, x_factor, y_factor, redshift_old, redshift_new, fwhm_arcsec, noise_level):
    """
    Run full simulation pipeline:
    bin → scale luminosity → convolve PSF → add noise
    """

    # Step 1: bin
    binned_cube = bin_cube(x_factor, y_factor, original_cube)

    # Step 2: scale luminosity
    scaled_data = scale_luminosity_for_redshift(binned_cube, redshift_old, redshift_new)
    binned_cube.data = scaled_data

    # Step 3: apply PSF convolution
    pixel_scale = binned_cube.wcs.get_axis_increments(unit=u.arcsec)[0]
    convolved_cube = apply_psf(binned_cube, fwhm_arcsec, pixel_scale)

    # Step 4: add noise
    noisy_cube = add_noise(convolved_cube, noise_level)

    return noisy_cube

def plot_masked_slice(data_cube, slice_index=0, title="Slice"):
    masked = np.ma.masked_invalid(data_cube[slice_index])
    plt.imshow(masked, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()



if __name__ == "__main__":

    cube = Cube("/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits")
    x_factor = 3
    y_factor = 3

    # Step 1: binning
    binned_cube = bin_cube(x_factor, y_factor, cube)
    print("After binning:", np.isnan(binned_cube.data).sum())
    print("Total NaNs in original cube:", np.isnan(cube.data).sum())

    # Step 2: scale luminosity
    redshift_old = 0.025
    redshift_new = 1.0
    scaled_data = scale_luminosity_for_redshift(binned_cube, redshift_old, redshift_new)
    binned_cube.data = scaled_data
    print("After luminosity scaling:", np.isnan(binned_cube.data).sum())

    # Step 3: apply PSF convolution
    pixel_scale = binned_cube.wcs.get_axis_increments(unit=u.arcsec)[0]
    convolved_cube = apply_psf(binned_cube, fwhm_arcsec=0.1, pixel_scale_arcsec=pixel_scale)
    print("After PSF convolution:", np.isnan(convolved_cube.data).sum())

    # Visualize original vs scaled luminosity
    plt.subplot(1, 2, 1)
    plt.title("Original data slice")
    im = plt.imshow(cube.data[0], origin='lower')
    plt.colorbar(im)

    plt.subplot(1, 2, 2)
    plt.title("Scaled data slice")
    im2 = plt.imshow(scaled_data[0], origin='lower')
    plt.colorbar(im2)
    plt.show()
    

    # Visualize before and after PSF
    smoothed_cube = apply_psf(cube.copy(), fwhm_arcsec=0.1, pixel_scale_arcsec=0.03)
    plt.subplot(1, 2, 1)
    plt.title("Before PSF")
    plt.imshow(cube.data[0], origin='lower')

    plt.subplot(1, 2, 2)
    plt.title("After PSF")
    plt.imshow(smoothed_cube.data[0], origin='lower')
    plt.show()

    # Add noise and check noise stddev
    noisy_cube = add_noise(cube.copy(), noise_level=0.1)
    noise_only = noisy_cube.data - cube.data
    print("Noise stddev:", np.std(noise_only))

    # Test full pipeline
    mock_cube = generate_mock_jwst_cube(
        original_cube=cube,
        x_factor=3,
        y_factor=3,
        redshift_old=0.025,
        redshift_new=1.0,
        fwhm_arcsec=0.07,
        noise_level=0.05
    )

    plt.imshow(mock_cube.data[0], origin='lower')
    plt.title("Mock JWST simulated cube slice")
    plt.colorbar()
    plt.show()