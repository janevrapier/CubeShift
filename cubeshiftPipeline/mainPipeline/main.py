
from mpdaf.obj import Cube
from astropy import units as u
import matplotlib.pyplot as plt
from matchSpectral import precomputed_match_spectral_resolution_variable_kernel, resample_spectral_axis
from zWavelengths import redshift_wavelength_axis
from binData import bin_cube, calculate_spatial_resampling_factor, abs_calculate_spatial_resampling_factor
from simulateObs import scale_luminosity_for_redshift, convolve_to_match_psf
from reprojectBinData import construct_target_header, reproject_cube
import numpy as np


# Get rid of provisional numbers from tests and use telescope dictionary instead

# Telescope class:

class Telescope:
    def __init__(self, name, z, spatial_fwhm, pixel_scale_x, pixel_scale_y, spectral_resolution,
                 spectral_sampling=None):
        self.name = name
        self.z = z
        self.spatial_fwhm = spatial_fwhm  # arcsec
        self.pixel_scale_x = pixel_scale_x    # arcsec/pixel
        self.pixel_scale_y = pixel_scale_y    # arcsec/pixel
        self.spectral_resolution = spectral_resolution  # R = λ/Δλ
        self.spectral_sampling = spectral_sampling  # Δλ in Å 

# Telescope dict
# Holds telescope OBJECTS (!)
telescope_specs = {
    "JWST_NIRCam": Telescope(
        name="JWST NIRCam",
        z=2.0,
        pixel_scale_x=0.031,  # arcsec/pixel
        pixel_scale_y=0.063,  # arcsec/pixel
        spatial_fwhm=0.07,    # arcsec -- simulating around F200W filter 
        spectral_resolution=1000 # resolving power!
    ),

    # Add other telescopes
}



# Pipeline:



def simulate_observation(cube, telescope_name, z_obs=None):
    """
    Simulates how a cube would appear if observed with a different telescope at a different redshift.
    Pipeline:
        1. Redshift wavelength axis
        2. Spatially rebin to match resolution
        3. Scale luminosity for redshift effects
        4. Convolve spatially to match PSF
        5. Match spectral resolution (convolve in lambda)
        6. Match spatial sampling (reproject to new pixel grid)
        7. Match spectral sampling (resample lambda axis)

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        Original data cube (e.g. from Keck/KCWI).
    telescope_name : str
        Key to look up specs from the telescope_specs dictionary.
    z_obs : float, optional
        Original redshift. If None, assumed 0.

    Returns
    -------
    cube_sim : mpdaf.obj.Cube
        The transformed cube simulating the target telescope's observation.
    """

    telescope = telescope_specs[telescope_name]
    z_old = z_obs if z_obs is not None else 0.0
    z_new = telescope.z

    print(f"Simulating observation as seen by {telescope.name} at z = {z_new}")

    # STEP 1: Redshift wavelength axis
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, z_old, z_new)
    print("WCS after redshifting:", redshifted_cube.wcs)


    # STEP 2: Rebin spatially to match physical resolution
    x_factor, y_factor = abs_calculate_spatial_resampling_factor(
        pixel_scale_x=cube.wcs.get_axis_increments(unit=u.arcsec)[0],
        pixel_scale_y=cube.wcs.get_axis_increments(unit=u.arcsec)[1],
        target_pixel_scale_x=telescope.pixel_scale_x,
        target_pixel_scale_y=telescope.pixel_scale_y,
        z_old=z_old,
        z_new=z_new
    )


    # Round and validate
    if x_factor < 1 or y_factor < 1:
        print("Skipping binning (factors < 1); relying on reprojection instead.")
        rebinned_cube = redshifted_cube
    else:
        rebinned_cube = bin_cube(
            x_factor=int(np.round(x_factor)),
            y_factor=int(np.round(y_factor)),
            data_cube=redshifted_cube,
            method='sum'
        )


    print(f"Cube shape: {redshifted_cube.shape}  (z, y, x)")
    print(f"x_factor: {x_factor}, y_factor: {y_factor}")


    # STEP 3: Scale luminosity to account for angular size and flux dimming
    scaled_data = scale_luminosity_for_redshift(rebinned_cube, z_old, z_new, method='both')
    rebinned_cube.data = scaled_data

    # STEP 4: Convolve spatially to match target PSF
    cube_psf = convolve_to_match_psf(
        rebinned_cube,
        fwhm_real_arcsec=cube.wcs.get_axis_increments(unit=u.arcsec)[0],
        fwhm_target_arcsec=telescope.spatial_fwhm,
        z_old=z_old,
        z_new=z_new
    )
    # Step 5 
    cube = precomputed_match_spectral_resolution_variable_kernel(cube_psf, R_input=3000, R_target=1000)

    # Step 6
    target_header = construct_target_header(
        cube,
        x_pixel_scale_arcsec=telescope.pixel_scale_x,
        y_pixel_scale_arcsec=telescope.pixel_scale_y
    )
    print(f"Target header CDELT1 (RA): {target_header['CDELT1'] * 3600:.4f} arcsec")
    print(f"Target header CDELT2 (DEC): {target_header['CDELT2'] * 3600:.4f} arcsec")


    cube = reproject_cube(cube, target_header)

    # Step 7 
    telescope = telescope_specs[telescope_name]  # e.g., "JWST_NIRCam"
    R = telescope.spectral_resolution           
    lam_center = np.median(cube.wave.coord())
    bin_width = lam_center / R

    wavelengths = cube.wave.coord()
    start = wavelengths[0]
    end = wavelengths[-1]

    new_wave_axis = np.arange(start, end + bin_width, bin_width)
    cube = resample_spectral_axis(cube, new_wave_axis)

    return cube


file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
cube = Cube(file_path)

z_obs = 0.025  # Original observed redshift

simulated_cube = simulate_observation(cube, "JWST_NIRCam", z_obs)

simulated_cube.write("JWST_simulated_cube.fits")


