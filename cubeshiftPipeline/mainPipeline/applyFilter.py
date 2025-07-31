from astropy.io import fits
from scipy.interpolate import interp1d
import numpy as np
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
from mpdaf.obj import Cube, WaveCoord
from zWavelengths import redshift_wavelength_axis
from astropy import units as u
import os



# Bandpass ranges in microns, based on JWST/NIRSpec documentation
NIRSPEC_THROUGHPUT_RANGES = {
    "f140x": (0.8, 2.0),      # Target acquisition
    "f110w": (1.0, 1.3),      # Bright target acquisition
    "f070lp": (0.7, 1.3),     # Used with G140M
    "f100lp": (1.0, 1.9),     # Used with G140M
    "f170lp": (1.7, 3.2),     # Used with G235M
    "f290lp": (2.9, 5.3),     # Used with G395M
    "clear": (0.6, 5.3),      # Used with PRISM or TA
}



def plot_throughput_and_spectrum(cube, wave_filter, thr_filter, x=0, y=0):
    """
    Plots the filter throughput and compares the cube spectrum before and after applying filter at (x,y).

    Parameters:
        cube: MPDAF Cube (already filtered)
        wave_filter: wavelength array of filter (in microns)
        thr_filter: throughput array (0 to 1)
        x, y: spatial pixel coordinates to extract spectrum
    """

    # Convert cube wavelengths from Å to microns
    wave_cube = cube.wave.coord() / 1e4  

    # Extract the filtered spectrum at pixel (x, y)
    spec_filtered = cube.data[:, y, x]  

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Throughput curve
    plt.plot(wave_filter, thr_filter, label='Filter Throughput', color='green', alpha=0.7)
    
    # Filtered spectrum
    plt.plot(wave_cube, spec_filtered / np.max(spec_filtered), label='Filtered Spectrum (norm)', color='blue')

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Normalized Flux / Throughput")
    plt.title(f"Spectrum at pixel (x={x}, y={y}) with Filter Applied")
    plt.legend()
    plt.grid(True)
    plt.show()



def load_filter_transmission(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        wl = data['WAVELENGTH']  # μm
        thr = data['THROUGHPUT']
    return wl, thr


def select_nirspec_mode(lambda_obs_micron):
    for mode, (lam_min, lam_max) in NIRSPEC_THROUGHPUT_RANGES.items():
        if lam_min <= lambda_obs_micron <= lam_max:
            return mode
    return None




def apply_nirspec_throughput(cube, z, line_rest_wavelength, filter_dir):
    """
    Applies the appropriate NIRSpec transmission filter to a cube based on redshifted line.
    
    Parameters:
        cube: MPDAF Cube
        z: redshift of the galaxy
        line_rest_wavelength: rest wavelength in Angstroms (e.g., 5007 for [OIII])
        filter_dir: path to directory containing *_trans.fits files

    Returns:
        MPDAF Cube after throughput correction.
    """
    lambda_obs = (line_rest_wavelength * (1 + z)) / 1e4  # Convert to microns
    print(f"Observed wavelength: {lambda_obs:.3f} µm at z = {z}")

    # Load all *_trans.fits files in directory
    filters = {}
    for fname in os.listdir(filter_dir):
        if fname.endswith("_trans.fits"):
            path = os.path.join(filter_dir, fname)
            with fits.open(path) as hdul:
                wave = hdul[1].data['WAVELENGTH']  # µm
                trans = hdul[1].data['THROUGHPUT']
                filters[fname] = (wave, trans)
    # debug
    print(f"Filter wavelength range: {wave.min():.3f} - {wave.max():.3f} µm")


    # Find all filters that cover lambda_obs
    candidates = []
    for fname, (wave, trans) in filters.items():
        if wave.min() <= lambda_obs <= wave.max():
            candidates.append((fname, wave, trans))

    if not candidates:
        raise ValueError(f"No transmission filter covers λ = {lambda_obs:.3f} µm at z = {z}")

    # Sort by coverage or priority if needed — currently just use the first match
    fname, wave, trans = candidates[0]
    print(f"Applying filter: {fname} (covers {wave.min():.2f}–{wave.max():.2f} µm)")

    # Interpolate transmission at cube wavelengths
    if cube.wave is None:
        raise ValueError("Cube has no spectral axis (cube.wave is None)")

    cube_waves_um = cube.wave.coord() / 1e4  # Cube wavelengths in µm
    interp = interp1d(wave, trans, bounds_error=False, fill_value=0.0)
    throughput = interp(cube_waves_um)

    print(f"Throughput range: min={throughput.min()}, max={throughput.max()}")

    # Apply to cube
    cube.data *= throughput[:, np.newaxis, np.newaxis]
    print(f"Cube wavelength range: {cube.wave.coord().min()/1e4:.3f} – {cube.wave.coord().max()/1e4:.3f} µm")


    return cube, wave, trans

def plot_results(filtered_cube):
    # test that filter is valid
    hdu = fits.open(r"/Users/janev/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite/OneDrive - Queen's University.noindex/OneDrive - Queen's University/MNU 2025/Transmission Filters/jwst_nirspec_f070lp_trans.fits")
    filter_data = hdu[1].data  # Assuming extension 1 contains throughput
    lam_filter = filter_data['WAVELENGTH']  # µm
    throughput = filter_data['THROUGHPUT']

    print(f"Filter wavelength range: {lam_filter.min()} - {lam_filter.max()} µm")
    print(f"Throughput min/max: {throughput.min()}, {throughput.max()}")

    plt.plot(lam_filter, throughput)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Throughput")
    plt.title("Filter Transmission Curve")
    plt.grid()
    plt.show()

    # Choose a bright pixel 
    x, y = 30, 96 
    spectrum = filtered_cube[:, y, x].data
    wavelengths = filtered_cube.wave.coord()  # in Angstroms

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectrum, color='blue')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux (filtered)')
    plt.title(f'Spectrum at pixel (y={y}, x={x})')
    plt.grid()
    plt.show()

    wave = filtered_cube.wave.coord()
    center = 1.252 * 1e4  # 1.252 μm in Å
    width = 20  # Narrowband width in Å

    # Find the slice indices within this narrow window
    mask = (wave > center - width/2) & (wave < center + width/2)
    narrowband_image = np.nansum(filtered_cube.data[mask], axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(narrowband_image, origin='lower', cmap='inferno')
    plt.colorbar(label='Flux')
    plt.title(f'Narrowband image around [O III] λ = {center:.0f} Å')
    plt.show()

    broadband_image = np.nansum(filtered_cube.data, axis=0)

    plt.figure(figsize=(6, 6))
    plt.imshow(broadband_image, origin='lower', cmap='magma')
    plt.colorbar(label='Flux')
    plt.title('Broadband image after filter applied')
    plt.show()

def apply_transmission_filter(cube, z, line_rest_wavelength):
    """
    Applies the appropriate transmission filter and returns cube + filter name.
    """
    # path to NIRSpec transmission filters:
    filter_dir = r"/Users/janev/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite/OneDrive - Queen's University.noindex/OneDrive - Queen's University/MNU 2025/Transmission Filters"
    print(f"Cube type: {type(cube)}")
    print(f"Wave object: {cube.wave}")

    filtered_cube, wave, throughput = apply_nirspec_throughput(cube, z, line_rest_wavelength, filter_dir)
    
    # Extract name from filename (e.g., jwst_nirspec_f070lp_trans.fits → f070lp)
    filter_name = "unknown"
    for fname in os.listdir(filter_dir):
        if fname.endswith("_trans.fits"):
            wl, thr = load_filter_transmission(os.path.join(filter_dir, fname))
            if np.allclose(wl, wave) and np.allclose(thr, throughput):
                filter_name = fname.split("_")[2]  # 'f070lp' from 'jwst_nirspec_f070lp_trans.fits'
                break

    return filtered_cube, filter_name

if __name__ == "__main__":
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    # Set the galaxy redshift and emission line rest wavelength
    z_obs = 0.025
    z_sim = 1.5
    rest_line = 5007  # [OIII]


    redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)
    # Your local transmission filter path (note the raw string for safety with spaces)
    filter_dir = r"/Users/janev/Library/Group Containers/UBF8T346G9.OneDriveStandaloneSuite/OneDrive - Queen's University.noindex/OneDrive - Queen's University/MNU 2025/Transmission Filters"
    cube_waves = cube.wave.coord() / 1e4  # in microns
    print(f"Cube wavelength range: {cube_waves.min():.3f} - {cube_waves.max():.3f} µm")

    # Apply throughput correction
    filtered_cube, wave_filter, thr_filter = apply_nirspec_throughput(redshifted_cube, z_sim, rest_line, filter_dir)
    

    # Plot 
    plot_throughput_and_spectrum(filtered_cube, wave_filter, thr_filter, x=30, y=96)

    # Test 
    plot_results(filtered_cube)

    # Save output cube
    filtered_cube.write("/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/filtered_cube_with_throughput.fits")