import numpy as np 



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

def _compute_num_of_extra_pixels(factor, data_cube):
    """Computes the number of pixels by which each axis dimension is more than
    an integer multiple of the reduction factor.

    Parameters
    ----------
    factor : `~numpy.ndarray`
        The array containing the reduction factor for each axis
    data_cube : `mpdaf.obj.Cube`
        mpdaf Cube object of the data

    Returns
    -------
    `~numpy.ndarray`
        The integer number of pixels extra in each axis dimension
    """
    num_extra_pixels = np.mod(data_cube.shape, factor).astype(int)

    return num_extra_pixels

def _compute_shortened_axis_dimensions(data_cube, num_extra_pixels, margin='center'):
    """Computes the slices that will be taken out to shorten the dimensions of 
    the data cube to be an integer multiple of the axis reduction factor.

    Parameters
    ----------
    data_cube : `mpdaf.obj.Cube`
        mpdaf Cube object of the data 
    num_extra_pixels : `~numpy.ndarray`
        An array containing the number of extra pixels in each axis
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

    Returns
    -------
    tuple
        The slices to apply to the data to make it the right dimensions for
        binning
    """

    # Add a slice for each axis to a list of slices
    slices = []

    # Iterate through the cube dimensions
    for k in range(data_cube.ndim):
        # Compute the slice of axis k needed to truncate this axis
        if margin == 'origin' or margin == 'left':
            nstart = 0
        elif margin == 'center':
            nstart = num_extra_pixels[k] // 2
        elif margin == 'right':
            nstart = num_extra_pixels[k]
        slices.append(slice(nstart, data_cube.shape[k] - num_extra_pixels[k] + nstart))

    slices = tuple(slices)

    return slices 

def _shorten_data_axis_dimensions(data_cube, slices):
    """Applies the slices to the data_cube, shortening the dimensions so that it 
    has an integer multiple of the reduction factor along each axis.

    Parameters
    ----------
    data_cube : `mpdaf.obj.Cube`
        An mpdaf Cube of the data 
    slices : tuple
        The slices to be applied to the data to shorten it

    Returns
    -------
    `mpdaf.obj.Cube`
        The data cube with the shortened dimensions.
    """

    # Create a sliced copy of the input data cube 
    tmp = data_cube[slices]

    # Copy the sliced data back into the data_cube, so that inplace=True works
    data_cube._data = tmp._data
    data_cube._var = tmp._var
    data_cube._mask = tmp._mask
    data_cube.wcs = tmp.wcs
    data_cube.wave = tmp.wave

    return data_cube
        
def _determine_new_data_shape(data_cube, factor):
    """Determines the new data shape after binning.  With the cropped data, the 
    dimensions should be integer multiples of the reduction factors.  This
    function returns the shape of the output image.

    Parameters
    ----------
    data_cube : `mpdaf.obj.Cube`
        An mpdaf object of the data cube
    factor : `~numpy.ndarray`
        An array of the reduction factor for each axis

    Returns
    -------
    `~numpy.ndarray`
        The new shape of the data after binning
    """
    newshape = data_cube.shape // factor 

    return newshape 

def _create_preshaping_array(newshape, factor):
    """Takes the newshape the data will have after binning, and the reduction
    factors, and creates a list of array dimensions made up of each of the final
    dimensions of the array, followed by the corresponding axis reduction factor.
    Reshaping the data with these dimensions places all of the pixels from each 
    axis that are to be summed on their own axis.

    Parameters
    ----------
    newshape : `~numpy.ndarray`
        The new shape the data will have after binning
    factor : `~numpy.ndarray`
        An array of the reduction factor for each axis

    Returns
    -------
    `~numpy.ndarray`
        A preshape to apply to the data to put all pixels from each axis that
        are to be summed on their own axis.
    """
    preshape = np.column_stack((newshape, factor)).ravel()

    return preshape

def _count_unmasked_pixels(data_cube, preshape):
    """Counts the number of unmasked pixels in the data cube that will contribute
    to each summed pixel in the final data cube.

    Parameters
    ----------
    data_cube : `mpdaf.obj.Cube`
        The mpdaf Cube object of the data 
    preshape : `~numpy.ndarray`
        The preshape to apply to the data to put all pixels from each axis that 
        are to be summed on their own axis.

    Returns
    -------
    `~numpy.ndarray`
        The number of unmasked pixels in each axis.    
    """
    unmasked = data_cube.data.reshape(preshape).count(1)
    for k in range(2, data_cube.ndim+1):
        unmasked = unmasked.sum(k)

    return unmasked

def _bin_data(data_cube, preshape, method='sum'):
    """The actual binning occurs here.  Reduces the size of the data array by 
    taking the sum, mean or median of successive groups of 'factor[0] x factor[1]'
    pixels.

    Parameters
    ----------
    data_cube : `mpdaf.obj.Cube`
        The data cube as an mpdaf Cube object
    preshape : `~numpy.ndarray`
        A preshape to apply to the data so that all pixels that are to be binned
        are put on their own axis.
    method : str, optional
        The method used to combine pixels when binning.  By default 'sum'.
        
        The options are:
            'sum':
                Takes the sum of the included pixels
            'median':
                Takes the median of the included pixels
            'mean':
                Takes the mean of the included pixels

    Returns
    -------
    `mpdaf.obj.Cube`
        The binned data cube
    """
    newdata = data_cube.data.reshape(preshape)
    for k in range(1, data_cube.ndim+1):
        if method == 'sum':
            newdata = np.nansum(newdata, axis=k)
        elif method == 'mean':
            newdata = np.nanmean(newdata, axis=k)
        elif method == 'median':
            newdata = np.nanmedian(newdata, axis=k)
    data_cube._data = newdata.data 

    return data_cube

def bin_data(x_factor, y_factor, data_cube, margin='center', method='sum', inplace=False):
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

    # compute the number of pixels by which each axis dimension is more than
    # an integer multiple of the reduction factor
    num_extra_pixels = _compute_num_of_extra_pixels(factor, data_cube)

    # if necessary, compute the slices needed to shorten the dimensions to be 
    # integer multiples of the axis reduction
    if np.any(num_extra_pixels != 0):
        slices = _compute_shortened_axis_dimensions(data_cube, 
                                                    num_extra_pixels,
                                                    margin=margin
                                                    )
        
        data_cube = _shorten_data_axis_dimensions(data_cube,
                                                  slices
                                                  )

    # Now the dimensions should be integer multiples of the reduction factors.
    # Need to figure out the shape of the output image 
    newshape = _determine_new_data_shape(data_cube, factor) 

    # create a list of array dimensions that are made up of each of the final 
    # dimensions of the array, followed by the corresponding axis reduction
    # factor.  Reshaping with these dimensions places all of the pixels from 
    # each axis that are to be summed on their own axis.
    preshape = _create_preshaping_array(newshape, factor)

    # compute the number of unmasked pixels of the data cube that will contribute
    # to each summed pixel in the output array 
    unmasked = _count_unmasked_pixels(data_cube, preshape)

    # reduce the size of the data array by taking the sum of the successive 
    # groups of 'factor[0] x factor[1]' pixels.
    data_cube = _bin_data(data_cube, preshape, method=method)

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
                bin_size = (proper_dist * (binned_cube.wcs.get_axis_increments(u.arcsec)[0]*u.arcsec)).to('pc')

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