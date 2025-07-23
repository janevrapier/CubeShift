
# matchSpectral.py

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy.ma as ma
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from astropy import units as u
import numpy as np
from mpdaf.obj import Cube, WaveCoord
import copy



def to_nan_array(arr):
    """Convert a masked array to a float array with NaNs where masked."""
    if isinstance(arr, ma.MaskedArray):
        return arr.filled(np.nan)
    return arr


def fill_nans_spectrum(spectrum):
    """
    Interpolates over NaNs in a 1D spectrum. If too many NaNs, fills with 0.
    """
    idx = np.arange(len(spectrum))
    good = ~np.isnan(spectrum)
    
    if good.sum() < 2:
        return np.nan_to_num(spectrum)  # fallback: fill with zeros

    interp = interp1d(idx[good], spectrum[good], bounds_error=False, fill_value="extrapolate")
    return interp(idx)


def match_spectral_resolution(cube, R_input, R_target):
    # VERSION 1 (stationary - not variable)
    """
    Matches spectral resolution of an MPDAF cube by convolving each spectrum 
    with a 1D Gaussian kernel.

    Parameters:
        cube      -- MPDAF Cube object
        R_input   -- original spectral resolution (λ / Δλ)
        R_target  -- desired spectral resolution (must be lower than R_input)

    Returns:
        cube      -- resolution-matched cube
    """
    print("Matching spectral resolution...")

    wave = cube.wave.coord()  # get wavelength array
    delta_lambda = wave[1] - wave[0]  # spectral pixel size in Å

    fwhm_input = wave / R_input
    fwhm_target = wave / R_target
    fwhm_kernel = np.sqrt(np.maximum(fwhm_target**2 - fwhm_input**2, 0))  # avoid negative sqrt

    # Convert FWHM to sigma (in pixels)
    sigma_pix = (fwhm_kernel / delta_lambda) / 2.355

    # Use central value for constant-kernel convolution
    sigma = sigma_pix[len(sigma_pix) // 2]
    kernel = Gaussian1DKernel(sigma)

    print("Cube shape:", cube.data.shape)
    print("Cube dtype:", cube.data.dtype)
    print("NaNs in cube:", np.isnan(cube.data).sum())


    for y in range(cube.shape[1]):
        for x in range(cube.shape[2]):
            spectrum = to_nan_array(cube.data[:, y, x])
            idx = np.arange(len(spectrum))
            good = np.isfinite(spectrum)

            if good.sum() < 2:
                print(f" Skipping spaxel (y={y}, x={x}) — only {good.sum()} good points")
                print("Spectrum preview:", spectrum[:10])
                continue

            filled = fill_nans_spectrum(spectrum)
            cube.data[:, y, x] = convolve(filled, kernel, boundary='extend')

    if isinstance(cube.data[:, y, x], ma.MaskedArray):
        print(f"Spaxel ({y},{x}) is a masked array.")



    print("Spectral resolution matching complete.")
    return cube



def match_spectral_resolution_variable_kernel(cube, R_input, R_target):
    # Version 2 (variable but very time consuming)
    """
    Matches spectral resolution by convolving each wavelength bin with 
    a wavelength-dependent Gaussian kernel.

    Parameters:
        cube      -- MPDAF Cube
        R_input   -- original spectral resolution (λ / Δλ)
        R_target  -- desired spectral resolution

    Returns:
        cube      -- resolution-matched cube
    """
    print("Matching spectral resolution with variable kernel...")

    new_cube = cube.copy()
    wave = cube.wave.coord()
    delta_lambda = wave[1] - wave[0]

    fwhm_input = wave / R_input
    fwhm_target = wave / R_target
    fwhm_kernel = np.sqrt(np.maximum(fwhm_target**2 - fwhm_input**2, 0))
    sigma_kernel_pix = fwhm_kernel / delta_lambda / 2.355

    n_wave = cube.shape[0]

    for y in range(cube.shape[1]):
        for x in range(cube.shape[2]):
            spectrum = to_nan_array(cube.data[:, y, x])
            spectrum = fill_nans_spectrum(spectrum)
            smoothed = np.zeros_like(spectrum)

            for i in range(n_wave):
                delta = np.zeros_like(spectrum)
                delta[i] = 1.0
                kernel = gaussian_filter1d(delta, sigma=sigma_kernel_pix[i], mode='constant')
                smoothed[i] = np.nansum(spectrum * kernel)

            new_cube.data[:, y, x] = smoothed

    print(" Spectral resolution matched using variable kernel.")
    return new_cube



def precomputed_match_spectral_resolution_variable_kernel(cube, R_input, R_target):
    # Version 3 (final version - )
    """
    Matches spectral resolution by convolving each wavelength bin with 
    a wavelength-dependent Gaussian kernel. Uses precomputed kernel bank. 
    Rolls precomputed kernels per wavelength rather than recalculating Gaussian.

    Parameters:
        cube      -- MPDAF Cube
        R_input   -- original spectral resolution (λ / Δλ)
        R_target  -- desired spectral resolution

    Returns:
        cube      -- resolution-matched cube
    """
    print("Matching precomputed spectral resolution with variable kernel...")

    new_cube = cube.copy()
    wave = cube.wave.coord()
    # Calculate the wavelength step size (assumes constant spacing)
    delta_lambda = wave[1] - wave[0]
    # Get number of wavelength bins (spectral dimension)
    n_wave = cube.shape[0]

    # --- STEP 1: Compute Gaussian kernel widths ---
    # FWHM (Full Width at Half Maximum) of the current spectral resolution
    fwhm_input = wave / R_input
    # FWHM of the target (lower) spectral resolution
    fwhm_target = wave / R_target
    # Compute the FWHM of the Gaussian kernel needed to degrade from input to target resolution
    # Use quadrature (if that makes sense?): FWHM_kernel^2 = FWHM_target^2 - FWHM_input^2
    fwhm_kernel = np.sqrt(np.maximum(fwhm_target**2 - fwhm_input**2, 0))
    # Convert FWHM to standard deviation (sigma) in pixel units, since convolution uses sigma
    sigma_kernel_pix = fwhm_kernel / delta_lambda / 2.355

    # --- STEP 2: Precompute all Gaussian kernels ---
    print(" Precomputing kernels...")
    kernel_bank = []
    for sigma in sigma_kernel_pix:
        # Create a delta function (1 at center, 0 elsewhere) the same length as the spectrum
        delta = np.zeros(n_wave)
        delta[n_wave // 2] = 1.0  # centered delta function
        # Apply Gaussian smoothing to the delta function to create the kernel
        # Each kernel is centered and ready to be rolled into place later
        kernel = gaussian_filter1d(delta, sigma=sigma, mode='constant')
        kernel_bank.append(kernel)

    # --- STEP 3: Convolve each spaxel spectrum using precomputed kernels ---
    for y in range(cube.shape[1]): # loop over spatial Y pixels
        for x in range(cube.shape[2]): # loop over spatial X pixels
            spectrum = to_nan_array(cube.data[:, y, x])
            # Fill in any NaNs with interpolated values
            spectrum = fill_nans_spectrum(spectrum)
            # Create an empty array to store the smoothed spectrum
            smoothed = np.zeros_like(spectrum) # move to outside loop

            # Loop over each wavelength bin
            for i in range(n_wave):
                # Roll the kernel so it's centered at the current wavelength index i
                kernel = np.roll(kernel_bank[i], i - n_wave // 2)
                # Apply the kernel to the spectrum using a dot product
                # This is equivalent to Gaussian smoothing around index i
                smoothed[i] = np.nansum(spectrum * kernel)
            # Store the smoothed spectrum back into the new cube
            new_cube.data[:, y, x] = smoothed

    print(" Spectral resolution matched using variable kernel.")
    return new_cube




def create_wavecoord_from_axis(new_wave_axis):
    """
    Creates a new wavelength coordinate system (WaveCoord) from an array of wavelengths.
    This is needed because MPDAF cubes use a WaveCoord object to define their spectral axis.
    The function uses the wavelength step size, starting wavelength, and assumes the first pixel 
    corresponds to the start.

    Returns:
        WaveCoord -- MPDAF object that defines the spectral axis of a cube
    """
    # Calculate the step size (Δλ) between each wavelength bin
    cdelt = new_wave_axis[1] - new_wave_axis[0] 
    # Set the reference value (starting wavelength)
    crval = new_wave_axis[0] 
    # Set the reference pixel (assume first bin is pixel 1 for simplicity)
    crpix = 1  
    return WaveCoord(cdelt=cdelt, crval=crval, crpix=crpix, cunit='Angstrom')


def resample_spectral_axis(cube, new_wave_axis):
    """
    Resamples (regrids) the spectral axis of the cube by interpolating every spectrum
    onto a new set of wavelength values.

    This is used when we want to change the *sampling* of the spectrum — for example, to match
    the wavelength grid of another instrument, or to downsample after resolution degradation.

    Parameters:
        cube           -- the original MPDAF cube to resample
        new_wave_axis  -- 1D array of new wavelength values to interpolate to

    Returns:
        resampled_cube -- new MPDAF Cube object with data interpolated onto new_wave_axis
    """
    print("Resampling spectral axis...")

    # Get the spatial dimensions of the cube
    ny, nx = cube.shape[1], cube.shape[2]
    # Number of wavelength bins in the new spectral axis
    new_nwave = len(new_wave_axis)
    # Create an empty array to hold the resampled cube data
    new_data = np.zeros((len(new_wave_axis), ny, nx))

    # Get the original wavelength axis from the cube
    old_wave = cube.wave.coord()

    # --- STEP 1: Loop through every spaxel and interpolate its spectrum to the new wavelengths ---
    for y in range(ny):
        for x in range(nx):
            # Extract the 1D spectrum at position (y, x)
            spectrum = cube.data[:, y, x]
             # Create an interpolator from the original spectrum
            # If the new wavelength is outside the old range, fill it with 0
            interp = interp1d(old_wave, spectrum, bounds_error=False, fill_value=0)
            new_data[:, y, x] = interp(new_wave_axis)

    # --- STEP 2: Define a new spectral coordinate system for the resampled axis ---
    new_wave = create_wavecoord_from_axis(new_wave_axis)
   
    # --- STEP 3: Build a new cube with resampled data and new spectral axis ---
    # Keep the spatial WCS (coordinate system) the same
    resampled_cube = Cube(data=new_data, wave=new_wave, wcs=cube.wcs)

    print("Spectral axis resampled.")
    return resampled_cube


if __name__ == "__main__":
    # save to a seperate file
    # Load your cube
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    # Define new wavelength axis you want to resample to
    new_wave_axis = np.linspace(11000, 16000, 88)  # Angstroms, e.g. JWST grism bins
    print("Original wave min/max:", cube.wave.coord()[0], cube.wave.coord()[-1])
    print("New wave min/max:", new_wave_axis[0], new_wave_axis[-1])
    
    # After loading and blurring the cube:
    start = cube.wave.coord()[0]
    end = cube.wave.coord()[-1]
    delta_lambda = 10  # desired new bin width in Angstroms
    

    new_wave_axis = np.arange(start, end, delta_lambda)
    print("Interpolating within range:", start, "to", end)
    print("New axis length:", len(new_wave_axis))

    # Resample
    resampled_cube = resample_spectral_axis(cube, new_wave_axis)

    # Plot to compare original vs resampled spectrum at central spaxel
    # 
    y0, x0 = cube.shape[1] // 2, cube.shape[2] // 2
    original_spec = cube.data[:, y0, x0]
    resampled_spec = resampled_cube.data[:, y0, x0]

    plt.figure()
    plt.plot(cube.wave.coord(), original_spec, label="Original")
    plt.plot(new_wave_axis, resampled_spec, label="Resampled", alpha=0.7)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Spectral Resampling")
    plt.show()

    # Assuming `cube` is your MPDAF cube object
    # And you're going from e.g. R = 4000 to R = 1000
    # Later this will be determined by a telescope/instrument dictionary
    R_input = 4000
    R_target = 1000

    # Pick a bright spaxel manually or automatically
    # Pick a spaxel and extract original spectrum (before matching)
    print(type(redshifted_cube))
    y, x = 50, 50
    wave = redshifted_cube.wave.coord()
    original_spectrum = redshifted_cube.data[:, y, x].copy()  

    # Apply spectral resolution matching
    cube = precomputed_match_spectral_resolution_variable_kernel(cube, R_input, R_target)

    # Extract smoothed spectrum at same spaxel
    smoothed_spectrum = cube.data[:, y, x]

    # Plot before/after
    plt.figure(figsize=(10, 4))
    plt.plot(wave, original_spectrum, label='Original', alpha=0.7)
    plt.plot(wave, smoothed_spectrum, label='Matched (Variable Kernel)', alpha=0.7)
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.title(f'Spectral Resolution Matching at spaxel ({y},{x})')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Spectral resolution matching complete.")



