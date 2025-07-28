import numpy as np
import matplotlib.pyplot as plt
from reprojectBinData import reproject_cube_preserve_wcs
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy import units as u
from reproject import reproject_interp
from mpdaf.obj import Cube, coords
from astropy.wcs import WCS as AstropyWCS

def auto_crop_cube(cube, sigma_thresh=3, threshold_nan=0.1, buffer=2, debug=False):
    """
        Automatically crop an MPDAF cube to the smallest bounding box containing significant flux.
        Uses either:
        - Background-subtracted signal (default)
        - Fallback to finite-value mask if too many NaNs (e.g., after reprojection)

        Parameters:
            cube          : MPDAF Cube
            sigma_thresh  : Signal threshold above background (in σ)
            threshold_nan : NaN fraction cutoff to trigger fallback method
            buffer        : Padding in pixels around valid region
            debug         : If True, shows plot of cropping region

        Returns:
            Cropped MPDAF Cube
    """
    img = cube.sum(axis=0)
    data = img.data
    nan_fraction = np.isnan(data).sum() / data.size

    print(f"NaN fraction in projected image: {nan_fraction:.3f}")

    if nan_fraction <= threshold_nan:
        # Use background-based cropping
        print("Cropping using signal above background ...")
        try:
            bkg, std = img.background()
            signal_mask = data > (bkg + sigma_thresh * std)

            if not signal_mask.any():
                raise ValueError("No signal found above background.")

            y_valid, x_valid = np.where(signal_mask)
        except Exception as e:
            print(f"[Warning] Background cropping failed: {e}")
            print("Falling back to NaN-based cropping ...")
            return crop_nan_mask_fallback(cube, buffer, debug)
    else:
        print("Too many NaNs — cropping based on finite values ...")
        return crop_nan_mask_fallback(cube, buffer, debug)

    ny, nx = data.shape
    y_min = max(y_valid.min() - buffer, 0)
    y_max = min(y_valid.max() + 1 + buffer, ny)
    x_min = max(x_valid.min() - buffer, 0)
    x_max = min(x_valid.max() + 1 + buffer, nx)

    print(f"Final cropped FOV: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")

    if debug:
        _plot_crop_region(data, x_min, x_max, y_min, y_max, title="Background-based Crop")

    return cube[:, y_min:y_max, x_min:x_max]


def crop_nan_mask_fallback(cube, buffer=0, debug=False):
    """
    Fallback: Crop MPDAF cube based on where any spectral layer is finite.
    """
    data = cube.data
    valid_mask = np.isfinite(data).any(axis=0)
    y_valid, x_valid = np.where(valid_mask)

    if len(y_valid) == 0 or len(x_valid) == 0:
        raise ValueError("No finite spatial pixels found.")

    ny, nx = data.shape[1:]

    y_min = max(y_valid.min() - buffer, 0)
    y_max = min(y_valid.max() + 1 + buffer, ny)
    x_min = max(x_valid.min() - buffer, 0)
    x_max = min(x_valid.max() + 1 + buffer, nx)

    print(f"[Fallback] Cropped to: y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")

    if debug:
        img = cube.sum(axis=0).data
        _plot_crop_region(img, x_min, x_max, y_min, y_max, title="NaN-based Crop")

    return cube[:, y_min:y_max, x_min:x_max]


def _plot_crop_region(image_data, x_min, x_max, y_min, y_max, title=""):
    """
    Helper function to show the cropping region on the 2D image.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image_data, origin='lower', cmap='viridis', vmin=np.nanpercentile(image_data, 5), vmax=np.nanpercentile(image_data, 99))
    plt.colorbar(label='Integrated Flux')
    plt.axhline(y_min, color='r', linestyle='--')
    plt.axhline(y_max, color='r', linestyle='--')
    plt.axvline(x_min, color='r', linestyle='--')
    plt.axvline(x_max, color='r', linestyle='--')
    plt.title(f"Cropping Region ({title})")
    plt.xlabel('X [pix]')
    plt.ylabel('Y [pix]')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    output_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cropped_cube.fits"

    cube = Cube(file_path)

    # Get original spatial WCS and modify pixel scale (0.03 arcsec/pix)
    original_wcs_astropy = AstropyWCS(cube.wcs.to_header()).sub([1, 2])
    target_header = original_wcs_astropy.to_header()
    target_header['CDELT1'] = -0.03 / 3600.0  # arcsec → deg
    target_header['CDELT2'] =  0.03 / 3600.0
    target_header['NAXIS1'], target_header['NAXIS2'] = cube.shape[2], cube.shape[1]

    # Reproject and crop
    target_wcs = AstropyWCS(target_header)
    shape_out = (cube.shape[1], cube.shape[2])
    cube_reprojected = reproject_cube_preserve_wcs(cube, target_wcs, shape_out)

    cropped_cube = auto_crop_cube(cube_reprojected, sigma_thresh=3, buffer=2, debug=True)
    cropped_cube.write(output_path, savemask='primary')

    print(f"Cropped cube saved to: {output_path}")

