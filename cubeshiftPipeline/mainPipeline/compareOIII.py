from astropy.io import fits
from mpdaf.obj import Cube
import numpy as np
import matplotlib.pyplot as plt
from matchSpectral import select_best_disperser_with_partial, crop_cube_to_wavelength_range, apply_dispersion_pipeline
from zWavelengths import redshift_wavelength_axis
from applyFilter import apply_transmission_filter

"""
def make_narrowband_map(cube, z_obs, z_sim, line_rest_wav=5007, width_angstrom=20):
    
    Create a narrowband [O III] image at a new redshift (z_sim), 
    selecting the best JWST NIRSpec disperser to observe it.

    Parameters:
    - cube : mpdaf.obj.Cube
        The input cube (e.g. KCWI) at original redshift z_obs.
    - z_obs : float
        Original redshift of the cube.
    - z_sim : float
        Desired simulated redshift.
    - line_rest_wav : float
        Rest-frame wavelength of [O III] (default: 5007 Å).
    - width_angstrom : float
        Width of narrowband in observed frame.

    Returns:
    - narrowband_2d : 2D array
        Collapsed narrowband image.
    - disperser_name : str
        Name of selected filter/disperser.
   


    # Step 1: Redshift cube
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, z_obs, z_sim)

    # Step 2: Define observed wavelength band for [O III]
    lam_obs = line_rest_wav * (1 + z_sim)
    lam_min = lam_obs - width_angstrom / 2
    lam_max = lam_obs + width_angstrom / 2

    # Step 3: Select best disperser + crop if needed
    disperser_name, sim_cube = select_best_disperser_with_partial(
        redshifted_cube, lam_min * 1e-4, lam_max * 1e-4
    )

    # Step 4: Collapse to narrowband image
    wave_axis = sim_cube.wave.coord()
    mask = (wave_axis >= lam_min) & (wave_axis <= lam_max)

    if not np.any(mask):
        raise ValueError(f"No valid wavelength coverage at z = {z_sim}")

    narrowband = sim_cube.data[mask, :, :].mean(axis=0)

    return narrowband, disperser_name

"""


def make_narrowband_map(cube, z_obs, z_sim, line_rest_wavelength=5007.0):
    """
    Shifts cube from z_obs to z_sim, applies transmission and dispersion filters, 
    and extracts a 2D [O III] narrowband image.

    Returns:
        map_2d : 2D numpy array
        filter_info : dict with 'transmission' and 'dispersion' filter names
    """

    if z_sim != z_obs:
        # Step 1: Redshift cube
        redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)
        # Step 2: Apply transmission filter
        cube_trans, trans_filter = apply_transmission_filter_for_compareOIII(redshifted_cube, z_sim, line_rest_wavelength)
        # Step 3: Apply dispersion filter
        cube_disp, disperser_info = apply_dispersion_pipeline(cube_trans, z_obs, z_sim)
        print(f"Disperser used: {disperser_info['name']} with R ≈ {disperser_info['R']}")
        if cube.wave is None:
            raise ValueError("Cube has no wavelength axis. Did you pass in a 2D map instead of a spectral cube?")

        cube_to_use = cube_disp
        filter_info = {
            "transmission": trans_filter,
            "dispersion": disperser_info['name']
        }

    else:
        # Use the original cube without filtering or redshifting
        if not hasattr(cube, 'wave') or cube.wave is None:
            raise ValueError("Original cube has no wavelength axis — check cube format.")
        cube_to_use = cube
        filter_info = {
            "transmission": None,
            "dispersion": None
        }

    # Step 4: Extract [O III] narrowband map at simulated redshift (or observed redshift)
    line_obs_wavelength = line_rest_wavelength * (1 + z_sim)
    map_2d = extract_oiii_map(cube_to_use, z_sim)
    if map_2d is None:
        raise ValueError(f"make_narrowband_map returned None at z = {z_sim}")

    return map_2d, filter_info

def full_pipeline(cube, z_obs, z_sim):

    # STEP 1: redshift 
    redshifted_cube, _ = redshift_wavelength_axis(cube, z_obs, z_sim)



# Load your KCWI cube (z ≈ 0.025)
file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
cube = Cube(file_path) 

z_obs = 0.025
redshifts = [0.025, 2.5, 3.0, 4.0]
maps = []
filters = []

for z_sim in redshifts:
    print(f"\nGenerating [O III] map at z = {z_sim}")
    if z_sim == z_obs:
        # No redshift needed for the first cube
        map_2d, filt = make_narrowband_map(cube, z_obs, z_obs)
    else:
        map_2d, filt = make_narrowband_map(cube, z_obs, z_sim)
    maps.append(map_2d)
    filters.append(filt)

# Plotting the maps

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, ax in enumerate(axes):
    im = ax.imshow(maps[i], origin='lower', cmap='viridis')
    ax.set_title(
        f"z = {redshifts[i]}\nTrans: {filters[i]['transmission']}\nDisp: {filters[i]['dispersion']}",
        fontsize=10
    )
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("[O III] Narrowband Maps at Different Redshifts", fontsize=14)
plt.tight_layout()
plt.show()
