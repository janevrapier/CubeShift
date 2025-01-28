def air_to_vac(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Implements the air to vacuum wavelength conversion described in eqn 64 and
        65 of Greisen 2006. The error in the index of refraction amounts to 1:10^9,
        which is less than the empirical formula.
        Function slightly altered from specutils.utils.wcs_utils.

        You can tell whether a file is in air or vacuum wavelengths by checking 
        the header value of CTYPE3, which should be one of AWAV, AWAV-LOG, WAVE
        or WAVE-LOG.

        Parameters
        ----------
        wavelength : :obj:'~numpy.ndarray'
            the air wavelength(s) in Angstroms

        Returns
        -------
        wavelength : :obj:'~numpy.ndarray'
            the vacuum wavelength(s) in Angstroms
        """
        #convert wavelength to um from Angstroms
        wlum = wavelength/10000
        #apply the equation from the paper
        return (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wavelength