from mpdaf.obj import Cube

import numpy as np
import matplotlib.pyplot as plt

def measure_hbeta_flux(
    cube, z, 
    w_line=4861.0, 
    line_window=(-15, 25), 
    cont_blue=(-80, -15), 
    cont_red=(25, 80), 
    debug=False, debug_coords=(19, 13)
):
    """
    Measure Hβ flux map from an IFU cube using quadratic continuum subtraction.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The IFU cube.
    z : float
        Source redshift.
    w_line : float, optional
        Rest wavelength of the line (default=4861 Å for Hβ).
    line_window : tuple
        (min_offset, max_offset) in Å relative to Hβ center for integration band.
    cont_blue : tuple
        (min_offset, max_offset) in Å relative to Hβ center for blue continuum band.
    cont_red : tuple
        (min_offset, max_offset) in Å relative to Hβ center for red continuum band.
    debug : bool, optional
        If True, prints diagnostic info and shows a debug plot for the given spaxel.
    debug_coords : tuple
        (j, i) coordinates of the spaxel to plot when debug=True.

    Returns
    -------
    flux_map : 2D numpy.ndarray
        Map of integrated line fluxes.
    """

    lam_hbeta = w_line * (1 + z)

    # Define windows relative to Hβ center
    band_line  = (lam_hbeta + line_window[0], lam_hbeta + line_window[1])
    band_cont1 = (lam_hbeta + cont_blue[0],  lam_hbeta + cont_blue[1])
    band_cont2 = (lam_hbeta + cont_red[0],   lam_hbeta + cont_red[1])

    flux_map = np.full(cube.shape[1:], np.nan)  # (ny, nx)

    for j in range(cube.shape[1]):
        for i in range(cube.shape[2]):
            spax = cube[:, j, i]
            if np.all(np.isnan(spax.data)):
                continue

            lam = spax.wave.coord()
            flux = spax.data

            mask_cont = ((lam > band_cont1[0]) & (lam < band_cont1[1])) | \
                        ((lam > band_cont2[0]) & (lam < band_cont2[1]))
            mask_line = (lam > band_line[0]) & (lam < band_line[1])

            if debug and (j, i) == debug_coords:
                print(f"Spaxel ({j},{i}) λ range: {lam.min():.1f}-{lam.max():.1f}")
                print(f"Hβ center = {lam_hbeta:.1f}")
                print("Line band:", band_line)
                print("Continuum bands:", band_cont1, band_cont2)
                print("Continuum points:", np.sum(mask_cont))
                print("Line points:", np.sum(mask_line))

                plt.figure(figsize=(8,4))
                plt.plot(lam, flux, color="k", lw=1, label="Spectrum")
                plt.axvline(lam_hbeta, color="k", ls="--", lw=1, alpha=0.6, label="Hβ center")
                plt.axvspan(*band_cont1, color="blue", alpha=0.3, label="continuum 1")
                plt.axvspan(*band_cont2, color="red", alpha=0.3, label="continuum 2")
                plt.axvspan(*band_line,  color="green", alpha=0.3, label="line region")
                plt.legend()
                plt.xlabel("Wavelength (Å)")
                plt.ylabel("Flux")
                plt.title(f"Spectrum at spaxel (y={j}, x={i})")
                plt.tight_layout()
                plt.show()

            if np.sum(mask_cont) < 3 or np.sum(mask_line) < 1:
                continue

            # Quadratic continuum fit
            coeffs = np.polyfit(lam[mask_cont], flux[mask_cont], 2)
            cont_fit = np.polyval(coeffs, lam)

            # Subtract continuum
            flux_subtracted = flux - cont_fit

            # --- Debug plot: continuum-subtracted spectrum ---
            # --- Debug plots: stacked view ---
            if debug and (j, i) == debug_coords:
                lam_min = lam_hbeta - 100
                lam_max = lam_hbeta + 100
                mask_plot = (lam > lam_min) & (lam < lam_max)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,7), sharex=True)

                # --- Top panel: original + fit ---
                ax1.plot(lam[mask_plot], flux[mask_plot], color="black", lw=1, label="Original spectrum")
                ax1.plot(lam[mask_plot], cont_fit[mask_plot], color="orange", lw=1.5, label="Quadratic continuum fit")
                ax1.axvspan(*band_line, color="green", alpha=0.3, label="line window")
                ax1.axvspan(*band_cont1, color="blue", alpha=0.2, label="continuum 1")
                ax1.axvspan(*band_cont2, color="red", alpha=0.2, label="continuum 2")
                ax1.set_ylabel("Flux")
                ax1.set_title(f"Spectrum at (y={j}, x={i}) — original + continuum fit")
                ax1.legend()

                # --- Bottom panel: continuum-subtracted ---
                ax2.plot(lam[mask_plot], flux_subtracted[mask_plot], color="purple", lw=1, label="Continuum-subtracted")
                ax2.axhline(0, color="k", ls="--", lw=1, alpha=0.6)
                ax2.axvspan(*band_line, color="green", alpha=0.3, label="line window")
                ax2.axvspan(*band_cont1, color="blue", alpha=0.2, label="continuum 1")
                ax2.axvspan(*band_cont2, color="red", alpha=0.2, label="continuum 2")
                ax2.set_xlabel("Wavelength (Å)")
                ax2.set_ylabel("Flux (subtracted)")
                ax2.set_title("Continuum-subtracted spectrum")
                ax2.legend()

                plt.tight_layout()
                plt.show()

            # Integrate continuum-subtracted line flux
            # before doing np.trapz subtract the line from the polynomial across the entire spectrum -- create plot to see that it has been subtracted
            flux_map[j, i] = np.trapz(flux[mask_line] - cont_fit[mask_line],
                                      lam[mask_line])

    return flux_map




if __name__ == "__main__":

    galaxy_name = "CGCG453"
    file_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    z_obs = 0.025  # Original observed redshift
    z_sim = 3 # Simulated redshift
    final_output_path_ext_invert = f"/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/Output_cubes/{galaxy_name}/z_{z_sim}_final_pipeline_with_ext_{galaxy_name}_cube.fits"
    
    cube = Cube(final_output_path_ext_invert)

    lam = cube.wave.coord()
    print("Cube wavelength coverage:", lam.min(), "-", lam.max())
    lam_hbeta = 4861.0 * (1 + z_sim)
    print("Expected Hβ center:", lam_hbeta)

    flux_map = measure_hbeta_flux(cube, z_sim, debug=True, debug_coords=(17,10))

    print("Flux map stats:")
    print(" min =", np.nanmin(flux_map))
    print(" max =", np.nanmax(flux_map))
    print(" mean =", np.nanmean(flux_map))
    print(" fraction NaNs =", np.isnan(flux_map).sum() / flux_map.size)

    # Plot with percentile scaling

    vmin = np.nanpercentile(flux_map, 5)
    vmax = np.nanpercentile(flux_map, 95)

    plt.figure(figsize=(8,6))
    im = plt.imshow(flux_map, origin='lower', cmap='inferno',
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=r'H$\beta$ flux')
    plt.title(r'H$\beta$ Flux Map')
    plt.xlabel('x [spaxel]')
    plt.ylabel('y [spaxel]')



    plt.show()