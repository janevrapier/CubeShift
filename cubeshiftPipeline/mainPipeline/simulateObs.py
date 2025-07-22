"""
simulateObs.py

This module simulates how a galaxy observed at low redshift with one telescope 
(e.g. Keck/KCWI) would appear if observed at higher redshift with another 
(e.g. JWST NIRCam).

The simulation includes:
    1. Binning the data cube to a lower spatial resolution (to match the target instrument's pixel scale).
    2. Adjusting the brightness in each pixel to account for redshift effects:
        - Change in angular size (affecting pixel area)
        - Dimming due to luminosity distance
    3. Blurring the image to match the target telescope's point spread function (PSF),
       using a Gaussian kernel that accounts for the redshift-dependent change in resolution.

These steps help us understand how features like star formation, outflows, or emission structure 
might look when viewed at higher redshift, where both resolution and brightness are affected 
by cosmology and the observing instrument.

Main functions:
---------------
- bin_cube(x_factor, y_factor, cube) — Bins the cube to lower spatial resolution.
- scale_luminosity_for_redshift(...) — Scales flux to simulate brightness at higher redshift.
- convolve_to_match_psf(...) — Applies Gaussian blur to simulate PSF at new redshift.
- generate_mock_jwst_cube(...) — Runs the full pipeline from binning to final simulated cube.

Example:
--------
- Load a low-redshift IFU cube.
- Simulate how it would appear at z = 1.0 using JWST's pixel scale and resolution.
- Visualize and compare results before and after simulation.
"""


import numpy as np
from scipy.ndimage import gaussian_filter
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from binData import bin_cube
from astropy.convolution import convolve, Gaussian2DKernel


def scale_luminosity_for_redshift(cube, redshift_old, redshift_new, method="angular"):
    """
    Rescale the observed luminosity of each spaxel based on redshift.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The data cube observed at redshift_old.
    redshift_old : float
        The original redshift of the observation.
    redshift_new : float
        The target redshift to simulate.
    method : str
        Scaling method to use. Options:
            - "angular": angular size change only (Da)
            - "luminosity": flux dimming only (Dl)
            - "both": includes both effects (Da and Dl)

    Returns
    -------
    new_data : numpy.ndarray
        The new spaxel luminosities as they would appear at redshift_new.
    """
    # find how many arcseconds one pixel covers
    arcsec_per_pixel = cube.wcs.get_axis_increments(unit=u.arcsec)[0]
    
    if method == "angular" or method == "both":
        # Angular size per arcsec
        # find out how much physical size (in kiloparsecs) is covered by one arcsecond at the old and new redshifts
        proper_dist_old = cosmo.kpc_proper_per_arcmin(redshift_old).to(u.kpc/u.arcsec)
        proper_dist_new = cosmo.kpc_proper_per_arcmin(redshift_new).to(u.kpc/u.arcsec)

        #  how much area on the sky each pixel covers at each redshift 
        spaxel_area_old = (arcsec_per_pixel * proper_dist_old)**2
        spaxel_area_new = (arcsec_per_pixel * proper_dist_new)**2

        # convert the brightness data to luminosity per unit area 
        lum_per_area = cube.data / spaxel_area_old.value
        # scales the luminosity up or down to reflect how much area that same pixel would cover at the new redshift
        scaled_data = lum_per_area * spaxel_area_new.value
    else:
        scaled_data = cube.data.copy()

    if method == "luminosity" or method == "both":
        # get the luminosity distances to the galaxy at the two redshifts
        Dl_old = cosmo.luminosity_distance(redshift_old)
        Dl_new = cosmo.luminosity_distance(redshift_new)

        # calculate the inverse-square law brightness scaling (the farther away the galaxy is, the dimmer it looks)
        flux_scaling = (Dl_old / Dl_new)**2
        flux_scaling = flux_scaling.to(u.dimensionless_unscaled).value
        # Apply the dimming factor to the image
        scaled_data *= flux_scaling

    return scaled_data


def convolve_to_match_psf(cube, fwhm_real_arcsec, fwhm_target_arcsec, z_old, z_new):
    """
    Convolve cube spatially to match target PSF, accounting for redshift scaling.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The cube to be convolved.
    fwhm_real_arcsec : float
        The original PSF FWHM in arcseconds (e.g., KCWI ≈ 0.29").
    fwhm_target_arcsec : float
        The telescope's PSF you want to simulate (e.g., JWST ≈ 0.07").
    z_old : float
        Original redshift.
    z_new : float
        New redshift.

    Returns
    -------
    cube_convolved : mpdaf.obj.Cube
        Cube convolved with matching kernel to simulate PSF at new z.
    """
    
    # Step 1: Scale original PSF to simulate how it would appear at high-z
    Da_old = cosmo.angular_diameter_distance(z_old)
    Da_new = cosmo.angular_diameter_distance(z_new)

    fwhm_sim = fwhm_real_arcsec * (Da_old / Da_new).value

    if fwhm_target_arcsec <= fwhm_sim:
        print("Target PSF is finer than simulated real PSF — skipping convolution.")
        return cube.copy()

    # Step 2: Compute matching kernel (in arcsec)
    fwhm_kernel = np.sqrt(fwhm_target_arcsec**2 - fwhm_sim**2)

    # Convert FWHM to sigma (Gaussian: sigma = FWHM / 2.355)
    sigma_arcsec = fwhm_kernel / 2.355
    pixel_scale = cube.wcs.get_axis_increments(unit=u.arcsec)[0]  # arcsec/pixel
    sigma_pixels = sigma_arcsec / pixel_scale

    print(f"PSF sim (arcsec): {fwhm_sim:.4f}")
    print(f"Matching kernel FWHM: {fwhm_kernel:.4f} arcsec → σ = {sigma_pixels:.2f} pixels")

    # Step 3: Apply 2D Gaussian convolution per wavelength slice
    convolved_cube = cube.copy()
    for i in range(cube.shape[0]):  # over wavelength slices
        convolved_cube.data[i] = gaussian_filter(cube.data[i], sigma=sigma_pixels)

    return convolved_cube

def generate_mock_jwst_cube(original_cube, x_factor, y_factor, redshift_old, redshift_new, fwhm_real_arcsec, fwhm_target_arcsec):
    """
    Run full simulation pipeline:
    bin → scale luminosity → convolve PSF
    """

    # Step 1: bin
    binned_cube = bin_cube(x_factor, y_factor, original_cube)

    # Step 2: scale luminosity
    scaled_data = scale_luminosity_for_redshift(binned_cube, redshift_old, redshift_new)
    binned_cube.data = scaled_data

    # Step 3: convolve to match PSF at new redshift
    convolved_cube = convolve_to_match_psf(
        binned_cube,
        fwhm_real_arcsec=fwhm_real_arcsec,
        fwhm_target_arcsec=fwhm_target_arcsec,
        z_old=redshift_old,
        z_new=redshift_new
    )

    return convolved_cube

def plot_masked_slice(data_cube, slice_index=0, title="Slice"):
    masked = np.ma.masked_invalid(data_cube[slice_index])
    plt.imshow(masked, origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

def print_scaling_factors(z1, z2):
    Da1 = cosmo.angular_diameter_distance(z1)
    Da2 = cosmo.angular_diameter_distance(z2)
    Dl1 = cosmo.luminosity_distance(z1)
    Dl2 = cosmo.luminosity_distance(z2)

    angular_scaling = (Da1 / Da2)**2
    luminosity_scaling = (Dl1 / Dl2)**2
    combined_scaling = angular_scaling * luminosity_scaling

    print(f"Da(z={z1:.3f}) = {Da1:.2f}, Da(z={z2:.3f}) = {Da2:.2f}")
    print(f"Dl(z={z1:.3f}) = {Dl1:.2f}, Dl(z={z2:.3f}) = {Dl2:.2f}")
    print(f"Angular (Da²) scaling factor:     {angular_scaling:.4e}")
    print(f"Luminosity (Dl²) scaling factor:  {luminosity_scaling:.4e}")
    print(f"Combined (Da² * Dl²) scaling:     {combined_scaling:.4e}\n")

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
    # Convolve to match simulated PSF
    fwhm_real = 0.29  # KCWI resolution in arcsec
    fwhm_target = 0.07  # JWST NIRCam at ~2 μm
    z1 = redshift_old
    z2 = redshift_new

    cube_blurred = convolve_to_match_psf(
        binned_cube,
        fwhm_real_arcsec=fwhm_real,
        fwhm_target_arcsec=fwhm_target,
        z_old=z1,
        z_new=z2
    )

    # Visualize PSF-convolved image
    plt.figure(figsize=(6, 5))
    plt.imshow(cube_blurred.data[0], origin='lower', cmap='viridis')
    plt.title("Cube after PSF Matching")
    plt.colorbar(label="Flux")
    plt.show()


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
    



    # Test full pipeline
    mock_cube = generate_mock_jwst_cube(
        original_cube=cube,
        x_factor=3,
        y_factor=3,
        redshift_old=0.025,
        redshift_new=1.0,
        fwhm_real_arcsec=0.29,
        fwhm_target_arcsec=0.07
    )


    im3 = plt.imshow(mock_cube.data[0], origin='lower')
    plt.title("Mock JWST simulated cube slice")
    plt.colorbar(im3)
    plt.show()

    print("\n--- Scaling factor comparisons ---")


    print_scaling_factors(redshift_old, redshift_new)

    # Generate cubes with different scaling methods
    binned_cube_copy1 = bin_cube(x_factor, y_factor, cube)
    scaled_ang = scale_luminosity_for_redshift(binned_cube_copy1, redshift_old, redshift_new, method="angular")

    binned_cube_copy2 = bin_cube(x_factor, y_factor, cube)
    scaled_lum = scale_luminosity_for_redshift(binned_cube_copy2, redshift_old, redshift_new, method="luminosity")

    binned_cube_copy3 = bin_cube(x_factor, y_factor, cube)
    scaled_both = scale_luminosity_for_redshift(binned_cube_copy3, redshift_old, redshift_new, method="both")

    # Visualize the different scaling methods
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.title("Angular Scaling Only")
    im4 = plt.imshow(scaled_ang[0], origin='lower', cmap='viridis')
    plt.colorbar(im4)

    plt.subplot(1, 3, 2)
    plt.title("Luminosity Scaling Only")
    im5 = plt.imshow(scaled_lum[0], origin='lower', cmap='viridis')
    plt.colorbar(im5)

    plt.subplot(1, 3, 3)
    plt.title("Both Angular + Luminosity")
    im6 = plt.imshow(scaled_both[0], origin='lower', cmap='viridis')
    plt.colorbar(im6)

    plt.tight_layout()
    plt.show()