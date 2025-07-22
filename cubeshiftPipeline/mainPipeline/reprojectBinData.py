import numpy as np
from mpdaf.obj import Cube
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from reproject import reproject_interp


def calc_proper_dist(z):
    # calculates angular diameter distance to a galaxy at redshift z
    # returns it in parsecs
    return cosmo.angular_diameter_distance(z).to(u.pc)


def construct_target_header(original_cube, x_pixel_scale_arcsec, y_pixel_scale_arcsec):
    # creates a new FITS header that describes the target spatial resolution (i.e. the pixel scale) for reprojecting a cube.
    header = original_cube.wcs.to_header()
    header['CDELT1'] = -x_pixel_scale_arcsec / 3600.0  # deg/pixel (RA is negative)
    header['CDELT2'] = y_pixel_scale_arcsec / 3600.0   # deg/pixel
    header['NAXIS1'] = original_cube.shape[-1]  # X
    header['NAXIS2'] = original_cube.shape[-2]  # Y
    return header


def reproject_cube(cube, target_header):
    # changes the spatial grid of the cube to match the pixel scale and dimensions specified in the target_header.
    print("Reprojecting cube ...")
    data = cube.data.data
    wcs_in = cube.wcs
    shape_out = (data.shape[0], target_header['NAXIS2'], target_header['NAXIS1'])
    reprojected_data = np.empty(shape_out)

    for i in range(data.shape[0]):
        slice2d = data[i]
        reprojected_slice, _ = reproject_interp(
            (slice2d, wcs_in.to_header()),
            target_header,
            shape_out=(target_header['NAXIS2'], target_header['NAXIS1'])
        )
        reprojected_data[i] = reprojected_slice

    # Use FITS header, not WCS
    new_cube = Cube(data=reprojected_data, header=target_header)
    new_cube.wave = cube.wave.copy()
    return new_cube



def calculate_spatial_resampling_factor(z_old, z_new, original_pixel_scale_arcsec, target_telescope_resolution_arcsec):
    # figures out how much you need to rebin (resize) your original cube 
    # to simulate how it would look at a different redshift and telescope resolution
    Da_old = cosmo.angular_diameter_distance(z_old)
    Da_new = cosmo.angular_diameter_distance(z_new)
    physical_size_kpc = original_pixel_scale_arcsec * u.arcsec * cosmo.kpc_proper_per_arcmin(z_old).to(u.kpc/u.arcsec)
    new_spaxel_size_arcsec = (physical_size_kpc / cosmo.kpc_proper_per_arcmin(z_new).to(u.kpc/u.arcsec)).to(u.arcsec)
    new_spaxel_size_arcsec = max(new_spaxel_size_arcsec.value, target_telescope_resolution_arcsec)
    rebin_factor = new_spaxel_size_arcsec / original_pixel_scale_arcsec
    return rebin_factor, new_spaxel_size_arcsec

def crop_valid_fov(cube):
    # crops cube to remove empty edges after reprojection x
    data = cube.data
    valid_mask = np.isfinite(data).any(axis=0)  # (y, x)
    y_valid, x_valid = np.where(valid_mask)
    
    y_min, y_max = y_valid.min(), y_valid.max() + 1
    x_min, x_max = x_valid.min(), x_valid.max() + 1

    return cube[:, y_min:y_max, x_min:x_max]

if __name__ == "__main__":
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    z1 = 0.02
    z2 = 2.0
    orig_pixscale = 0.29  # arcsec/pixel
    sim_telescope_resolution = 0.031  # arcsec/pixel (e.g. JWST)

    factor, new_pix = calculate_spatial_resampling_factor(z1, z2, orig_pixscale, sim_telescope_resolution)
    print(f"\nAngular resampling factor: {factor:.2f}")
    print(f"New simulated pixel size: {new_pix:.4f} arcsec")

    kpc_per_arcsec_old = cosmo.kpc_proper_per_arcmin(z1).to(u.kpc/u.arcsec)
    kpc_per_arcsec_new = cosmo.kpc_proper_per_arcmin(z2).to(u.kpc/u.arcsec)
    physical_res_old = orig_pixscale * kpc_per_arcsec_old
    physical_res_new = sim_telescope_resolution * kpc_per_arcsec_new

    print(f"\nOriginal resolution: {physical_res_old:.2f}")
    print(f"Simulated resolution at z={z2}: {physical_res_new:.2f}")

    if physical_res_new > physical_res_old:
        print(" Resolution gets worse — binning required.")
        bin_factor = int(np.round((physical_res_new / physical_res_old).value))
        x_scale = orig_pixscale * bin_factor
        y_scale = orig_pixscale * bin_factor  # You can change this to be different if needed

        target_header = construct_target_header(cube, x_pixel_scale_arcsec=x_scale, y_pixel_scale_arcsec=y_scale)
        cube_resampled = reproject_cube(cube, target_header)
        cube_resampled = crop_valid_fov(cube_resampled)
    else:
        print(" Resolution gets better — skipping upsampling.")
        cube_resampled = cube.copy()

    # === Visualize ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Cube (z ≈ 0.02)")
    plt.imshow(cube.data[0], origin='lower', cmap='viridis')
    plt.colorbar(label='Flux')
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")

    plt.subplot(1, 2, 2)
    plt.title(f"Simulated Cube at z={z2}")
    plt.imshow(cube_resampled.data[0], origin='lower', cmap='viridis')
    plt.colorbar(label='Flux')
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")

    plt.tight_layout()
    plt.show()
