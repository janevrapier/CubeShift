
from mpdaf.obj import Cube
from astropy import units as u
import matplotlib.pyplot as plt
from matchSpectral import precomputed_match_spectral_resolution_variable_kernel, resample_spectral_axis, apply_dispersion_pipeline
from zWavelengths import redshift_wavelength_axis
from binData import bin_cube, calculate_spatial_resampling_factor, abs_calculate_spatial_resampling_factor
from simulateObs import scale_luminosity_for_redshift, convolve_to_match_psf
from reprojectBinData import reproject_cube_preserve_wcs
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs import WCS
from cropCube import auto_crop_cube, trim_empty_edges
from applyFilter import apply_transmission_filter
from scipy.ndimage import gaussian_filter1d
from mpdaf.obj import Image, plot_rgb
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch




# Get rid of provisional numbers from tests and use telescope dictionary instead
# Telescope class:

class Telescope:
    def __init__(self, name, spatial_fwhm, pixel_scale_x, pixel_scale_y, spectral_resolution,
                 spectral_sampling=None):
        self.name = name
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
        pixel_scale_x=0.031,  # arcsec/pixel
        pixel_scale_y=0.063,  # arcsec/pixel
        spatial_fwhm=0.07,    # arcsec -- simulating around F200W filter 
        spectral_resolution=1000 # resolving power!
    ),
        "VLT_MUSE": Telescope(
        name="VLT MUSE",
        pixel_scale_x=0.2,     # arcsec/pixel
        pixel_scale_y=0.2,
        spatial_fwhm=0.6,      # arcsec (typical seeing-limited PSF in WFM)
        spectral_resolution=3000,
        spectral_sampling=1.25  # Å (typical for MUSE WFM)
    ),
    # Add NIRSpec 
    # Add other telescopes
}



# Pipeline:



def simulate_observation(cube, telescope_name, z_obs, z_sim):
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

    print(f"Simulating observation as seen by {telescope.name} at z = {z_sim}")

    # STEP 1: Redshift wavelength axis
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, z_obs, z_sim)
    print("WCS after redshifting:", redshifted_cube.wcs)


    # STEP 2: Apply transmission filter 
    line_rest_wavelength = 5007 # For [OIII], change this for other emission lines!!!!
    transmission_cube, trans_filter_name = apply_transmission_filter(redshifted_cube, z_sim, line_rest_wavelength)


    # STEP 2: Rebin spatially to match physical resolution
    x_factor, y_factor = abs_calculate_spatial_resampling_factor(
        pixel_scale_x=cube.wcs.get_axis_increments(unit=u.arcsec)[0],
        pixel_scale_y=cube.wcs.get_axis_increments(unit=u.arcsec)[1],
        target_pixel_scale_x=telescope.pixel_scale_x,
        target_pixel_scale_y=telescope.pixel_scale_y,
        z_obs=z_obs,
        z_sim=z_sim
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
    scaled_data = scale_luminosity_for_redshift(rebinned_cube, z_obs, z_sim, method='both')
    rebinned_cube.data = scaled_data

    # STEP 4: Convolve spatially to match target PSF
    cube_psf = convolve_to_match_psf(
        rebinned_cube,
        fwhm_real_arcsec=cube.wcs.get_axis_increments(unit=u.arcsec)[0],
        fwhm_target_arcsec=telescope.spatial_fwhm,
        z_obs=z_obs,
        z_sim=z_sim
    )

    # STEP 5: Apply dispersion filter and spectral resolution matching

    blurred_cube, best_disperser = apply_dispersion_pipeline(cube_psf, z_obs, z_sim)

    # this is the step that gives a really weird result 

    # STEP 6: Reproject spatially to match target pixel scale
    # original_header = cube.wcs.to_header()
    # original_wcs_astropy = AstropyWCS(original_header)

    # # Extract spatial 2D WCS
    # spatial_wcs = original_wcs_astropy.sub([1, 2])
    # target_header = spatial_wcs.to_header()
    # target_header['CDELT1'] = -telescope.pixel_scale_x / 3600.0
    # target_header['CDELT2'] = telescope.pixel_scale_y / 3600.0

    # ny, nx = cube.shape[1:]  # Spatial shape
    # target_header['NAXIS1'] = nx
    # target_header['NAXIS2'] = ny

    # target_spatial_wcs_astropy = AstropyWCS(target_header)
    # shape_out = (ny, nx)

    # cube = reproject_cube_preserve_wcs(cube, target_spatial_wcs_astropy, shape_out)
    

    # Step 7 
    telescope = telescope_specs[telescope_name]  # e.g., "JWST_NIRCam"
    R = telescope.spectral_resolution           
    lam_center = np.median(blurred_cube.wave.coord())
    bin_width = lam_center / R

    wavelengths = blurred_cube.wave.coord()
    start = wavelengths[0]
    end = wavelengths[-1]

    new_wave_axis = np.arange(start, end + bin_width, bin_width)
    cube = resample_spectral_axis(blurred_cube, new_wave_axis)

    return cube, trans_filter_name, best_disperser


def test_JWST_full_pipeline():
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    z_obs = 0.025  # Original observed redshift
    z_sim = 2.5

    simulated_cube, trans_filter_name, disp_filter = simulate_observation(cube, "JWST_NIRCam", z_obs, z_sim)
    disp_filter_name = disp_filter["name"]
    output_path = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/final_simulated_cube_{z_sim}.fits"
    simulated_cube.write(output_path)
    print(f"Transmission filter used: {trans_filter_name}")
    print(f"Dispersion filter used: {disp_filter_name}")
    print(f" Simulated FITS cube saved to: {output_path}")


def test_MUSE_FOV():
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    # Step 1: Redshift wavelength axis
    z_old = 0.025
    telescope = telescope_specs["VLT_MUSE"]
    z_new = 1

    print(f"Redshifting from z = {z_old} to z = {z_new}")
    redshifted_cube, _ = redshift_wavelength_axis(cube, z_old, z_new)

    # Step 2: Reproject spatially to match target pixel scale
    # Get original WCS
    original_header = redshifted_cube.wcs.to_header()
    original_wcs_astropy = AstropyWCS(original_header)

    # Build 2D spatial WCS
    spatial_wcs = original_wcs_astropy.sub([1, 2])
    target_header = spatial_wcs.to_header()
    target_header['CDELT1'] = -telescope.pixel_scale_x / 3600.0
    target_header['CDELT2'] = telescope.pixel_scale_y / 3600.0

    # Set output shape same as input (you can adjust this later if needed)
    ny, nx = redshifted_cube.shape[1:]
    target_header['NAXIS1'] = nx
    target_header['NAXIS2'] = ny

    target_wcs = AstropyWCS(target_header)
    shape_out = (ny, nx)

    print("Reprojecting to match MUSE pixel scale")
    reprojected_cube = reproject_cube_preserve_wcs(redshifted_cube, target_wcs, shape_out)

    # Step 3: Crop to nonzero region (if needed)
    print("Cropping to region with signal...")
    cropped_cube = trim_empty_edges(reprojected_cube, buffer=2, debug=True)

    # Save output
    output_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/kcwi_to_muse_prelim.fits"
    cropped_cube.write(output_path)

    print(f" Saved redshifted, reprojected, cropped cube to:\n{output_path}")




def extract_centered_flux_map(
    cube,
    line_rest,
    z,
    broad_width=30,  # Angstroms for finding peak
    narrow_width=10  # Angstroms for final integration
):
    """
    Extracts a flux map centered on the brightest emission region of a spectral line.
    
    Parameters:
    - cube: MPDAF Cube object (assumed to be at redshift z)
    - line_rest: rest-frame wavelength (e.g. 5007 for [O III])
    - z: redshift of the cube
    - broad_width: width (Å) of initial search window
    - narrow_width: width (Å) for final flux map

    Returns:
    - flux_map: 2D numpy array centered on emission
    """
    lam_obs = line_rest * (1 + z)
    lam = cube.wave.coord()

    print(f"[DEBUG] Observed line center: {lam_obs:.2f} Å")
    print(f"[DEBUG] Cube wavelength range: {lam.min():.2f} - {lam.max():.2f} Å")


    # Step 1: Broad wavelength mask around line center
    broad_mask = (lam > lam_obs - broad_width/2) & (lam < lam_obs + broad_width/2)
    cube_broad = cube.copy().select_lambda(lam_obs - broad_width/2, lam_obs + broad_width/2)
    subcube_data = cube_broad.data

    print(f"[DEBUG] Broad mask wavelength range: {lam_obs - broad_width/2:.2f} to {lam_obs + broad_width/2:.2f}")
    print(f"[DEBUG] Number of spectral channels in broad mask: {np.sum(broad_mask)}")
    print(f"[DEBUG] Broad cube shape: {subcube_data.shape}")


    # Step 2: Collapse over spectral axis and find brightest (y,x)
    collapsed_broad = np.nansum(subcube_data, axis=0)
    max_pos = np.unravel_index(np.nanargmax(collapsed_broad), collapsed_broad.shape)
    y_peak, x_peak = max_pos

    print(f"[DEBUG] Peak position in broad mask: (y, x) = ({y_peak}, {x_peak})")
    print(f"[DEBUG] Peak flux value in broad collapsed cube: {collapsed_broad[y_peak, x_peak]:.3e}")


    # Step 3: Apply narrower spectral mask
    narrow_mask = (lam > lam_obs - narrow_width/2) & (lam < lam_obs + narrow_width/2)
    cube_narrow = cube.copy().select_lambda(lam_obs - narrow_width/2, lam_obs + narrow_width/2)

    print(f"[DEBUG] Narrow mask wavelength range: {lam_obs - narrow_width/2:.2f} to {lam_obs + narrow_width/2:.2f}")
    print(f"[DEBUG] Number of spectral channels in narrow mask: {np.sum(narrow_mask)}")
    print(f"[DEBUG] Narrow cube shape: {cube_narrow.data.shape}")


    # Step 4: Extract spectrum at bright location (optional debug)
    spectrum = cube_narrow[:, y_peak, x_peak]
    print(f"[DEBUG] Spectrum shape at peak position: {spectrum.shape}")
    print(f"[DEBUG] Spectrum at peak (first 5 channels): {spectrum[:5]}")


    # Step 5: Sum over spectral axis to create narrowband map
    flux_map = np.nansum(cube_narrow.data, axis=0)
    print(f"[DEBUG] Flux map shape: {flux_map.shape}")

    return flux_map

def extract_centered_flux_map_with_z(cube, line_rest, z_obs, z, width_narrow_rest=20, width_broad_rest=40):
    """
    Extracts a flux map by integrating over a redshifted emission line using a difference-of-Gaussians approach.
    """
    # apply same vmin and vmax to all of them (from first slice)
    print(f"\nExtracting flux map for z = {z}...")

    # Redshift line center and widths
    line_obs = line_rest * (1 + z)
    width_narrow_obs = width_narrow_rest * (1 + z) # 20 angstroms covers the OIII emisson line
    width_broad_obs = width_broad_rest * (1 + z)
    
    print(f"Observed line center: {line_obs:.1f} Å")
    print(f"Observed widths: narrow = {width_narrow_obs:.1f} Å, broad = {width_broad_obs:.1f} Å")

    # Get wavelength axis
    lam = cube.wave.coord()  # angstroms

    # Create masks for narrow and broad integration ranges
    mask_narrow = np.abs(lam - line_obs) <= (width_narrow_obs / 2)
    mask_broad = np.abs(lam - line_obs) <= (width_broad_obs / 2)

    print(f"Spectral channels in narrow mask: {mask_narrow.sum()}")
    print(f"Spectral channels in broad mask: {mask_broad.sum()}")

    if mask_narrow.sum() == 0 or mask_broad.sum() == 0:
        print("Warning: No channels found in mask — returning empty flux map.")
        return np.zeros(cube.shape[1:])  # (ny, nx)

    # Extract subcubes and integrate
    subcube_narrow = cube.data[mask_narrow, :, :]
    subcube_broad = cube.data[mask_broad, :, :]

    flux_map_narrow = np.nansum(subcube_narrow, axis=0)
    flux_map_broad = np.nansum(subcube_broad, axis=0)

    # Flux = narrow band - broad band (DoG-style background subtraction)
    flux_map = flux_map_narrow - flux_map_broad

    return flux_map


def plot_flux_map_for_z(ax, cube, z, z_obs, line_rest, fig,
                        trans_filter_name=None, disp_filter_name=None):
    """Plot a [O III] flux map on the given axis."""
    flux_map = extract_centered_flux_map_with_z(cube, line_rest, z_obs, z)

    im = ax.imshow(flux_map, origin='lower', cmap='inferno')
    
    if z == z_obs:
        title = f"[O III] Flux Map for KCWI z = {z}"
    elif trans_filter_name and disp_filter_name:
        title = f"[O III] z={z}\n{trans_filter_name}, {disp_filter_name}"
    else:
        title = f"[O III] z={z}"

    ax.set_title(title)
    ax.set_xlabel("X (pix)")
    ax.set_ylabel("Y (pix)")
    fig.colorbar(im, ax=ax, label='Flux')

    return flux_map



def plot_rgb_map_for_z(ax, cube, z):
    """Extract RGB images and plot on the given axis."""
    rgb_imgs = extract_rgb_images_from_cube(
        cube, z,
        rest_waves=(4861, 5007, 4500), # Hβ 4861Å, [O III] 5007Å, Stellar Continuum ~4500Å
        width=20
    )
    plot_rgb_from_images(ax, rgb_imgs)
    ax.set_title(f"RGB z={z}")
    return rgb_imgs


def extract_rgb_images_from_cube(cube, z, rest_waves, width=20):
    """
    Create MPDAF Images for R, G, B channels by slicing the cube around observed-frame wavelengths.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The spectral cube (already redshifted).
    z : float
        Observed redshift.
    rest_waves : tuple
        Tuple of three rest-frame wavelengths (blue, green, red), e.g., (4861, 5007, 6563).
    width : float
        Width of the band in angstroms, in rest frame. Will be redshifted accordingly.

    Returns
    -------
    Tuple of three mpdaf.obj.Image objects for R, G, B
    """
    obs_waves = [(1 + z) * lam for lam in rest_waves]
    obs_widths = [(1 + z) * width for _ in rest_waves]

    images = []
    for lam, dw in zip(obs_waves, obs_widths):
        img = cube.get_image((lam - dw/2, lam + dw/2))

        images.append(img)
    return images  # [B, G, R] order expected


def plot_rgb(images, vmin=None, vmax=None, ax=None, use_wcs=False, **kwargs):
    """Generate an RGB image from 3 MPDAF Image objects and plot it."""
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    # Get image data, and replace NaNs with 0
    b, g, r = [np.nan_to_num(im.data) for im in images]

    # Normalize based on vmin and vmax
    # Double check this 
    def norm(data, vmin, vmax):
        return np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)

    b_img = norm(b, vmin[0], vmax[0])
    g_img = norm(g, vmin[1], vmax[1])
    r_img = norm(r, vmin[2], vmax[2])

    rgb = np.dstack((r_img, g_img, b_img))

    if use_wcs:
        ax.imshow(rgb, origin='lower', **kwargs)
    else:
        ax.imshow(rgb, origin='lower', **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])

    return ax, rgb

def plot_rgb_from_images(ax, images, vmin=None, vmax=None, **kwargs):
    """Plot RGB image using given MPDAF Images (B, G, R order)."""
    # Use this ONCE to get vmin and vmax from the first RGB image set
    vmin, vmax = None, None
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': cube.wcs})
    for i, images in enumerate(rgb_image_list):  # assuming you have 4 sets of images
        ax = axes.flat[i]
        if i == 0:
            # Compute from first image set
            b_back, _ = images[0].background()
            g_back, _ = images[1].background()
            r_back, _ = images[2].background()
            b_max = images[0].data.max()
            g_max = images[1].data.max()
            r_max = images[2].data.max()
            vmin = (b_back, g_back, r_back)
            vmax = (b_max, g_max, r_max)
        
        plot_rgb_from_images(ax, images, vmin=vmin, vmax=vmax)

    ax, _ = plot_rgb(
        images,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        use_wcs=True,
        **kwargs
    )

    return ax




def plot_hbeta_hgamma_ratio(cube, z=0.0, ax=None, hbeta_width=20, hgamma_width=20):
    """
    Plot the map of Hβ / Hγ ratio from a data cube into the given axis.

    Parameters:
    - cube: mpdaf.obj.Cube instance
    - z: redshift of the source
    - ax: matplotlib axis to plot into
    - hbeta_width: Width (Å) around Hβ line for integration
    - hgamma_width: Width (Å) around Hγ line for integration
    """

    # Rest-frame line centers in Angstroms
    HBETA_REST = 4861.0
    HGAMMA_REST = 4341.0

    # Convert to observed wavelengths
    hbeta_obs = HBETA_REST * (1 + z)
    hgamma_obs = HGAMMA_REST * (1 + z)

    # Extract flux maps by integrating around the emission lines
    hbeta_map = cube.get_image(wave=(hbeta_obs - hbeta_width/2, hbeta_obs + hbeta_width/2))
    hgamma_map = cube.get_image(wave=(hgamma_obs - hgamma_width/2, hgamma_obs + hgamma_width/2))

    # Mask division warnings and invalid ratios
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_data = hbeta_map.data / hgamma_map.data
        ratio_mask = np.logical_or(hbeta_map.mask, hgamma_map.mask)
        ratio_data[ratio_mask] = np.nan

    # Plot into the given axis
    im = ax.imshow(ratio_data, origin='lower', cmap='plasma', vmin=0, vmax=4)
    ax.set_title(f"Hβ / Hγ at z={z}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    return im  # for colorbar



# === Load original KCWI cube ===

file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
cube = Cube(file_path) 
print(type(cube))

z_obs = 0.025  # Observed redshift of the real galaxy

# === Target redshifts for simulation ===
redshifts = [z_obs, 2.5, 3.0, 4.0]
telescope = "JWST_NIRCam"  

# === [OIII] 5007 in rest-frame ===
oiii_rest = 5007  # Angstrom

# Set up flux map figure
fig_flux, axes_flux = plt.subplots(2, 2, figsize=(12, 10))
axes_flux = axes_flux.flatten()
# Set up RGB map figure
fig_rgb, axes_rgb = plt.subplots(2, 2, figsize=(12, 10))
axes_rgb = axes_rgb.flatten()
# Set up Hbeta/Hgamma ratio figure 
fig_ratio, axes_ratio = plt.subplots(2, 2, figsize=(12, 10))
axes_ratio = axes_ratio.flatten()


for i, z in enumerate(redshifts):
    ax_flux = axes_flux[i]
    ax_rgb = axes_rgb[i]
    ax_ratio = axes_ratio[i]

    if z == z_obs:
        cube_z = cube
        im = plot_hbeta_hgamma_ratio(cube_z, z, ax=ax_ratio)

        flux_map = extract_centered_flux_map_with_z(cube, oiii_rest, z_obs, z)
        plot_flux_map_for_z(ax_flux, cube_z, z, z_obs, oiii_rest, fig_flux)


        plot_rgb_map_for_z(ax_rgb, cube_z, z)

    else:
        try:
            cube_z, trans_filter_name, disp_filter = simulate_observation(
                cube, telescope, z_obs, z
            )
            disp_filter_name = disp_filter["name"]

            im = plot_hbeta_hgamma_ratio(cube_z, z, ax=ax_ratio)

            plot_flux_map_for_z(
                ax_flux, cube_z, z, z_obs, oiii_rest, fig_flux,
                trans_filter_name=trans_filter_name,
                disp_filter_name=disp_filter_name
            )
            plot_rgb_map_for_z(ax_rgb, cube_z, z)

        except Exception as e:
            print(f"Error at z={z}: {e}")
            ax_flux.set_visible(False)
            ax_rgb.set_visible(False)
            ax_ratio.set_visible(False)


# Final display steps
# Why are none of the titles showing up??
fig_flux.suptitle("[O III] Flux Maps at Various Redshifts", fontsize=16)

fig_flux.tight_layout()
fig_flux.subplots_adjust(top=0.92)

fig_rgb.suptitle("RGB Maps at Various Redshifts", fontsize=16)

fig_rgb.tight_layout()
fig_rgb.subplots_adjust(top=0.92, bottom=0.1)

legend_elements = [
    Patch(facecolor='red', label='Hβ 4861 (Red)'),
    Patch(facecolor='green', label='[O III] 5007 (Green)'),
    Patch(facecolor='blue', label='Stellar continuum ~4500 Å (Blue)')
]

# Add legend to the overall figure (fig_rgb), not to a specific subplot
fig_rgb.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=3,
    fontsize=12,
    frameon=False,
    bbox_to_anchor=(0.5, 0.05)
)
fig_ratio.suptitle("Hβ / Hγ Line Ratio Maps at Various Redshifts", fontsize=16)
fig_ratio.tight_layout()
fig_ratio.subplots_adjust(top=0.92)

# Optional: Add a shared colorbar
cbar_ax = fig_ratio.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig_ratio.colorbar(im, cax=cbar_ax, label=r'H$\beta$ / H$\gamma$')


plt.show()
