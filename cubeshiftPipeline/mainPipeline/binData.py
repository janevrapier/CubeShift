import numpy as np 
from mpdaf.obj import Cube
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from reprojectBinData import construct_target_header, reproject_cube


def calc_proper_dist(z):
    """Return angular diameter distance in parsecs at redshift z"""
    return cosmo.angular_diameter_distance(z).to(u.pc)



def _reduction_factor_to_array(x_factor, y_factor, data_cube):
    """Turns the reduction factors (x and y) into an array.  We keep the 
    z-direction as 1 since we don't want to bin in the wavelength direction.

    Parameters
    ----------
    x_factor : int
        The integer reduction factor along the image x axis
    y_factor : int
        The integer reduction factor along the image y axis
    data_cube : `mpdaf.obj.Cube`
        mpdaf cube object of the data

    Returns
    -------
    `~numpy.ndarray`
        An array of the reduction factors dependent on the shape of the data array
    """
    
    # putting the first reduction factor as 1 since we don't want to bin 
    # in the wavelength direction
    if len(data_cube.shape)==3:
        factor = np.asarray([1, y_factor, x_factor])
    elif len(data_cube.shape)==2:
        factor = np.asarray([y_factor, x_factor])
    elif len(data_cube.shape)==1:
        factor = np.asarray([x_factor])

    return factor 



def bin_cube(x_factor, y_factor, data_cube, margin='center', method='sum', inplace=False):
    """Combine the neighbouring pixels to reduce the spatial size of the array 
    by integer factors along the x and y axes.  Each output pixel is the sum of 
    n pixels, where n is the product of the reduction factors x_factor and 
    y_factor.  Adapted from mpdaf.DataArray._rebin(), changed to allow the choice 
    to sum the pixels, take the median or take the mean while binning.

    Parameters
    ----------
    x_factor : int
        The integer reduction factor along the image x axis.
    y_factor : int
        The integer reduction factor along the image y axis.
    data_cube : `mpdaf.obj.Cube`
        mpdaf cube object of the data
    margin : str, optional
        When the dimensions of the input array are not integer multiples of the 
        reduction factor, the array is truncated to remove just enough pixels
        that its dimensions are multiples of the reduction factor.  This subarray 
        is then rebinned in place of the original array.  The margin parameter 
        determines which pixels of the input array are truncated, and which remain.
        By default 'center'.

        The options are:
            'origin' or 'left':
                The starts of the axes of the output array are coincident with 
                the starts of the axes of the input array.
            'center':
                The centre of the output array is aligned with the centre of the 
                input array, within one pixel along each axis.
            'right':
                The ends of the axes of the output array are coincident with the 
                ends of the axes of the input array.
    method : str, optional
        The method used to combine pixels when binning.  By default 'sum'.
        
        The options are:
            'sum':
                Takes the sum of the included pixels
            'median':
                Takes the median of the included pixels
            'mean':
                Takes the mean of the included pixels 
    inplace : bool, optional
        If False, return a rebinned copy of the data array (the default).
        If True, rebin the original data array in-place, and return that.

    Returns
    -------
    `mpdaf.obj.Cube`
        The rebinned cube

    Raises
    ------
    ValueError
        'Unsupported margin parameter' - the margin mode is not one of 'center',
        'origin', 'left', or 'right'.
    ValueError
        'The reduction factors must be from 1 to shape' - the shape of the array 
        is not larger than the reduction factors, so the array can't be divided
        into bins of that size.
    """
    # If inplace is false, create a copy of the cube to change
    data_cube = data_cube if inplace else data_cube.copy()
    
    # Reject unsupported margin modes.
    if margin not in ('center', 'origin', 'left', 'right'):
        raise ValueError('Unsupported margin parameter: %s' % margin)

    # turn the reduction factors into an array
    factor = _reduction_factor_to_array(x_factor, y_factor, data_cube)
    
    # check that the reduction factors are in the range 1 to shape-1
    if np.any(factor < 1) or np.any(factor>=data_cube.shape):
        raise ValueError('The reduction factors must be from 1 to shape')

    #compute the number of pixels by which each axis dimension is more than
    #an integer multiple of the reduction factor
    n = np.mod(data_cube.shape, factor).astype(int)

    #if necessary, compute the slices needed to shorten the dimensions to be 
    #integer multiples of the axis reduction
    if np.any(n != 0):

        # Add a slice for each axis to a list of slices.
        slices = []
        for k in range(data_cube.ndim):
            # Compute the slice of axis k needed to truncate this axis.
            if margin == 'origin' or margin == 'left':
                nstart = 0
            elif margin == 'center':
                nstart = n[k] // 2
            elif margin == 'right':
                nstart = n[k]
            slices.append(slice(nstart, data_cube.shape[k] - n[k] + nstart))

        slices = tuple(slices)

        # Get a sliced copy of the input object.
        tmp = data_cube[slices]

        # Copy the sliced data back into data_cube, so that inplace=True works.
        data_cube._data = tmp._data
        data_cube._var = tmp._var
        data_cube._mask = tmp._mask
        data_cube.wcs = tmp.wcs
        data_cube.wave = tmp.wave

    # Now the dimensions should be integer multiples of the reduction factors.
    # Need to figure out the shape of the output image 
    newshape = data_cube.shape // factor 

    # create a list of array dimensions that are made up of each of the final 
    # dimensions of the array, followed by the corresponding axis reduction
    # factor.  Reshaping with these dimensions places all of the pixels from 
    # each axis that are to be summed on their own axis.
    preshape = np.column_stack((newshape, factor)).ravel()

    # compute the number of unmasked pixels of the data cube that will contribute
    # to each summed pixel in the output array 
    # Count how many unmasked pixels are in each binned region
    unmasked_mask = ~data_cube.data.mask  # True = unmasked
    unmasked = unmasked_mask.reshape(preshape).astype(int)

    if data_cube.ndim == 3:
        unmasked = unmasked.sum(axis=(1, 3, 5))
    elif data_cube.ndim == 2:
        unmasked = unmasked.sum(axis=(1, 3))


    # Reshape the data array to prepare for binning
    newdata = data_cube.data.reshape(preshape)

    # Determine which axes to reduce (these are the axes of the binning dimensions)
    if data_cube.ndim == 3:
        bin_axes = (1, 3, 5)
    elif data_cube.ndim == 2:
        bin_axes = (1, 3)
    else:
        raise ValueError("Unsupported data cube dimensionality")

    # Apply the chosen reduction method across the binning axes
    if method == 'sum':
        newdata = np.nansum(newdata, axis=bin_axes)
    elif method == 'mean':
        newdata = np.nanmean(newdata, axis=bin_axes)
    elif method == 'median':
        newdata = np.nanmedian(newdata, axis=bin_axes)
    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Store the result
    data_cube._data = newdata


    # the treatment of the variance array is complicated by the possibility 
    # of masked pixels in the data array. 
    if data_cube._var is not None:
        newvar = data_cube.var.reshape(preshape)
        # When calculating the sum: 
        # A sum of N data pixels p[i] with variance v[i] has a variance of 
        # sum(v[i]) 
        if method == 'sum':
            for k in range(1, data_cube.ndim+1):
                newvar = newvar.sum(k)
        # When calculating the mean: 
        # A sum of N data pixels p[i] with variance v[i] has a variance of 
        # sum(v[i]/N^2) 
        # where N^2 is the number of unmasked pixels in that particular sum.
        elif method == 'mean':
            for k in range(1, data_cube.ndim+1):
                newvar = newvar.sum(k)
            newvar /= unmasked**2
        # When calculating the median: 
        # A sum of N data pixels p[i] has an estimated variance of 
        # (1.253 * stdev(p[i]))^2 / N 
        # where N is the number of unmasked pixels in that particular sum.
        elif method == 'median':
            for k in range(1, data_cube.ndim+1):
                newvar = (1.253 * np.nanstd(newdata, axis=k))**2
            newvar /= unmasked
        # add whichever one it was to the data_cube
        data_cube._var = newvar.data

    # Any pixels in the output array that come from zero unmasked pixels of the 
    # input array should be masked
    data_cube._mask = unmasked < 1

    # update spatial world coordinates
    if data_cube._has_wcs and data_cube.wcs is not None and data_cube.ndim > 1:
        data_cube.wcs = data_cube.wcs.rebin([factor[-2], factor[-1]])
    

    return data_cube




def remove_var(data_cube):
    """Creates a cube or image without the variance extension

    Parameters
    ----------
    data_cube : `~mpdaf.obj.Cube` or `~mpdaf.obj.Image` object
        The mpdaf Cube/Image object

    Returns
    -------
    `~mpdaf.obj.Cube` or `~mpdaf.obj.Image` object
        The input data without the variance extension
    """
    # clone the cube 
    data_cube_novar = data_cube.clone()

    # give it the data extension
    data_cube_novar.data = data_cube.data 

    return data_cube_novar


def bin_cubes_and_remove_var(x_factor_list, y_factor_list, cube_list, redshift=None, **kwargs):
    """Takes the input cube list and bins the data x_factor x y_factor according 
    to the input lists.  Will save the output and name the results using either 
    the parsecs of the binned spaxels or the x by y factors depending on if the 
    redshift information has been given.

    Parameters
    ----------
    x_factor_list : list
        the integers by which to bin in the x direction
    y_factor_list : list
        the integers by which to bin in the y direction.  Must be the same length 
        as x_factor_list.
    cube_list : list
        the filenames of the cubes to read in
    redshift : float or list or None, optional
        the redshift of the cubes, either a list of redshifts or a single value 
        for all cubes.  Used to calculate the proper distance, and thereby find 
        the new physical size of the binned spaxels, which is used in the saved 
        file name.  If None, the filename uses the x_factor by y_factor instead.
        By default None.
    """

    # check that x_factor_list and y_factor_list are the same length 
    assert len(x_factor_list) == len(y_factor_list), "x_factor_list must be same length as y_factor_list"
    
    # iterate through cube list 
    for i, file in enumerate(cube_list):
        # open the file as a cube
        this_cube = Cube(file)
        this_cube.filename = file  # manually attach filename


        # if redshift given calculate the proper distance
        if redshift:
            if type(redshift)==list:
                proper_dist = calc_proper_dist(redshift[i])
            else:
                proper_dist = calc_proper_dist(redshift)

        # bin the cube - iterate through the x/y_factor lists 
        for j in range(len(x_factor_list)):
            binned_cube = bin_cube(x_factor_list[j], y_factor_list[j], this_cube, **kwargs)

            # calculate the new bin size in pc (if the redshift was given)
            if redshift:
                pixel_scale_rad = binned_cube.wcs.get_axis_increments(u.arcsec)[0] * u.arcsec
                bin_size = proper_dist * pixel_scale_rad.to(u.radian)

                new_filename = binned_cube.filename.split('.fits')[0] + "_binned_{:0>3d}pc.fits".format(int(bin_size.value))
            else:
                new_filename = binned_cube.filename.split('.fits')[0] + "_binned_{:0>3d}x{:0>3d}spax.fits".format(x_factor_list[j], y_factor_list[j])

            # save the cube
            binned_cube.write(new_filename, savemask=False)

            # remove the variance 
            binned_cube_novar = remove_var(binned_cube)

            # save the cube
            new_filename = new_filename.split('.fits')[0] + "_novar.fits"
            binned_cube_novar.write(new_filename, savemask=False)

def calculate_rebin_factors(z_old, z_new, original_pixel_scale_arcsec, desired_new_pixel_scale_arcsec=None):
    """
    Calculate rebinning factor to match angular resolution at high redshift.

    Parameters
    ----------
    z_old : float
        Redshift of the original cube (e.g. 0.02)
    z_new : float
        Redshift you want to simulate (e.g. 2.0)
    original_pixel_scale_arcsec : float
        Original cube's spatial pixel scale in arcsec/pixel (e.g. 0.29")
    desired_new_pixel_scale_arcsec : float, optional
        Pixel scale of the new telescope. If None, it will be derived from redshift.

    Returns
    -------
    rebin_factor : float
        How much to bin the original cube spatially to match simulated resolution.
    """

    # Proper kpc per arcsec
    kpc_per_arcsec_old = cosmo.kpc_proper_per_arcmin(z_old).to(u.kpc/u.arcsec)
    kpc_per_arcsec_new = cosmo.kpc_proper_per_arcmin(z_new).to(u.kpc/u.arcsec)

    # Physical size of 1 pixel in original cube
    spaxel_size_kpc = original_pixel_scale_arcsec * u.arcsec * kpc_per_arcsec_old

    # Convert back to angular size at new redshift
    new_spaxel_size_arcsec = (spaxel_size_kpc / kpc_per_arcsec_new).to(u.arcsec)

    # If telescope has finite pixel scale (e.g. JWST), use it as a limit
    if desired_new_pixel_scale_arcsec:
        new_spaxel_size_arcsec = max(new_spaxel_size_arcsec.value, desired_new_pixel_scale_arcsec) * u.arcsec

    # Calculate rebin factor
    rebin_factor = (new_spaxel_size_arcsec / (original_pixel_scale_arcsec * u.arcsec)).value
    

    return rebin_factor

def calculate_spatial_resampling_factor(
    z_old, z_new,
    original_pixel_scale_arcsec,
    target_telescope_resolution_arcsec
):
    """
    Calculate the spatial binning (or interpolation) factor needed to simulate
    observations of a nearby galaxy at high redshift using a given telescope.

    Parameters
    ----------
    z_old : float
        Redshift of the original observation (e.g. 0.02).
    z_new : float
        Redshift you want to simulate (e.g. 2.0).
    original_pixel_scale_arcsec : float
        Spatial resolution of original cube (arcsec/pixel).
    target_telescope_resolution_arcsec : float
        Angular resolution limit (e.g. pixel scale or PSF FWHM) of new telescope.

    Returns
    -------
    rebin_factor : float
        Ratio of new to old spaxel size (arcsec), after accounting for angular scaling.
        >1 → binning (decrease resolution)
        <1 → interpolate (increase resolution)
    new_spaxel_size_arcsec : float
        Effective spaxel size at high z in arcsec/pixel.
    """
    # Get angular diameter distances
    Da_old = cosmo.angular_diameter_distance(z_old)
    Da_new = cosmo.angular_diameter_distance(z_new)

    # Calculate how the *physical* size of a spaxel appears at new redshift
    physical_size_kpc = original_pixel_scale_arcsec * u.arcsec * cosmo.kpc_proper_per_arcmin(z_old).to(u.kpc/u.arcsec)
    new_spaxel_size_arcsec = (physical_size_kpc / cosmo.kpc_proper_per_arcmin(z_new).to(u.kpc/u.arcsec)).to(u.arcsec)

    # But telescope has a floor — we cannot get better than its resolution
    new_spaxel_size_arcsec = max(new_spaxel_size_arcsec.value, target_telescope_resolution_arcsec)

    # Rebin factor = how many original pixels make up one simulated pixel
    rebin_factor = new_spaxel_size_arcsec / original_pixel_scale_arcsec

    return rebin_factor, new_spaxel_size_arcsec

def abs_calculate_spatial_resampling_factor(pixel_scale_x, pixel_scale_y,
                                            target_pixel_scale_x, target_pixel_scale_y,
                                            z_old, z_new):

    """
    Return how much to spatially bin (in x and y) to match the physical size per pixel
    that you would have if this galaxy were at z_new and observed with the new telescope.
    """

    # Convert angular scales to physical sizes
    dA_old = cosmo.angular_diameter_distance(z_old)
    dA_new = cosmo.angular_diameter_distance(z_new)

    # Physical size per pixel at original redshift
    phys_size_x_kpc = (pixel_scale_x * u.arcsec).to(u.radian).value * dA_old.to(u.kpc).value
    phys_size_y_kpc = (pixel_scale_y * u.arcsec).to(u.radian).value * dA_old.to(u.kpc).value

    # Desired physical pixel size at z_new (target telescope scale)
    target_phys_x_kpc = (target_pixel_scale_x * u.arcsec).to(u.radian).value * dA_new.to(u.kpc).value
    target_phys_y_kpc = (target_pixel_scale_y * u.arcsec).to(u.radian).value * dA_new.to(u.kpc).value

    # How much you need to bin to match that
    x_factor = target_phys_x_kpc / phys_size_x_kpc
    y_factor = target_phys_y_kpc / phys_size_y_kpc

    return x_factor, y_factor



if __name__ == "__main__":

    from mpdaf.obj import Cube
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    import numpy as np

    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(file_path)

    # Optional pre-binning
    bin_cubes_and_remove_var(
        x_factor_list=[2, 4],
        y_factor_list=[2, 4],
        cube_list=[file_path],
        redshift=0.025
    )

    # === Simulation setup ===
    z1 = 0.02
    z2 = 2.0
    orig_pixscale = 0.29   # arcsec/pixel
    sim_telescope_resolution = 0.031  # arcsec/pixel (e.g., JWST NIRCam native)

    # Angular scaling (for info)
    factor, new_pix = calculate_spatial_resampling_factor(
        z_old=z1,
        z_new=z2,
        original_pixel_scale_arcsec=orig_pixscale,
        target_telescope_resolution_arcsec=sim_telescope_resolution
    )
    print(f"\nAngular resampling factor: {factor:.2f}")
    print(f"New simulated pixel size: {new_pix:.4f} arcsec")

    # === Physical resolution scaling (this determines what to do) ===
    kpc_per_arcsec_old = cosmo.kpc_proper_per_arcmin(z1).to(u.kpc/u.arcsec)
    kpc_per_arcsec_new = cosmo.kpc_proper_per_arcmin(z2).to(u.kpc/u.arcsec)

    physical_res_old = orig_pixscale * kpc_per_arcsec_old
    physical_res_new = sim_telescope_resolution * kpc_per_arcsec_new

    print(f"\nOriginal resolution: {physical_res_old:.2f}")
    print(f"Simulated resolution at z={z2}: {physical_res_new:.2f}")

    # Decide on resampling strategy
    if physical_res_new > physical_res_old:
        print(" Resolution gets worse — binning required.")
        rebin_factor = physical_res_new / physical_res_old
        bin_factor = int(np.round(rebin_factor.value))
        print(f"Rebinning with factor: {bin_factor}")

        if cube._has_wcs and cube.wcs is not None:
            # Construct the new target header with updated pixel scale
            x_scale = orig_pixscale * bin_factor
            y_scale = orig_pixscale * bin_factor
            target_header = construct_target_header(cube, x_pixel_scale_arcsec=x_scale, y_pixel_scale_arcsec=y_scale)

            # Use your reproject function to resample with WCS
            cube_resampled = reproject_cube(cube, target_header)
        else:
            print(" No valid WCS found — falling back to simple binning.")
            cube_resampled = bin_cube(bin_factor, bin_factor, cube)

    else:
        print(" Resolution gets better — do NOT upsample (we skip resampling).")
        cube_resampled = cube.copy()


    # === Visualize original vs resampled (first wavelength slice) ===
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

