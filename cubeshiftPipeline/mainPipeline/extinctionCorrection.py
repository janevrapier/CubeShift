from zWavelengths import redshift_wavelength_axis
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from dustmaps.sfd import SFDQuery
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import io, re

NED_EXTINCT_URL = "https://ned.ipac.caltech.edu/cgi-bin/nph-calc_extinct"

def fetch_ned_extinction_table(ra, dec, timeout=30):
    """
    Query NED extinction calculator and return a DataFrame with columns:
      'band' (str), 'wavelength_um' (float), 'A_lambda_mag' (float)
    ra/dec may be strings (hh:mm:ss, ±dd:mm:ss) or floats (degrees).
    """

    params = {
        "in_csys": "Equatorial",
        "in_equinox": "J2000.0",
        "obs_epoch": "2000.0",
        "lon": ra_deg,
        "lat": dec_deg,
        "pa": "0.0",
        "out_csys": "Equatorial",
        "out_equinox": "J2000.0"
    }

    r = requests.get(NED_EXTINCT_URL, params=params, timeout=timeout,
                     headers={"User-Agent":"python-requests (astro pipeline)"})
    r.raise_for_status()
    html = r.text

    # Parse HTML and look for the table that contains Band / lamEff / A(lam)
    soup = BeautifulSoup(html, "lxml")

    # find the table which has headers that include "Band" and "lamEff" or "A(lambda)"
    candidate_tables = soup.find_all("table")
    df = None
    for table in candidate_tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        hdr_text = " ".join(headers).lower()
        if ("band" in hdr_text and ("lameff" in hdr_text or "a(lam)" in hdr_text or "a(lambda)" in hdr_text)):
            # parse rows
            rows = []
            for tr in table.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                if not tds:
                    continue
                # Typical NED table columns (observed layout can vary):
                # index | Band | lamEff (um) | A(lam) | Reference
                # so we expect len(tds) >= 4
                if len(tds) >= 4:
                    band = tds[1].strip()
                    # try to parse wavelength and A(lam)
                    try:
                        lam_eff = float(re.sub(r'[^\d\.Ee+-]', '', tds[2]))
                    except Exception:
                        lam_eff = np.nan
                    try:
                        a_lam = float(re.sub(r'[^\d\.Ee+-]', '', tds[3]))
                    except Exception:
                        a_lam = np.nan
                    rows.append((band, lam_eff, a_lam))
            if rows:
                df = pd.DataFrame(rows, columns=["band", "wavelength_um", "A_lambda_mag"])
                # drop rows where wavelength is nan
                df = df.dropna(subset=["wavelength_um", "A_lambda_mag"])
                break

    # Fallback: if BS4 parsing failed, try to parse the ASCII text that can also be present
    if df is None:
        lines = html.splitlines()
        # try to find the start of the table by header keywords
        start = None
        for i, L in enumerate(lines):
            if re.search(r'\bBand\b', L) and re.search(r'lam', L, re.I):
                start = i
                break
        if start is None:
            raise RuntimeError("Couldn't find extinction table in NED response; page layout may have changed.")
        text_table = "\n".join(lines[start:])
        # Use pandas to read whitespace-delimited table (best-effort)
        try:
            df_raw = pd.read_csv(io.StringIO(text_table), delim_whitespace=True, comment="#", engine="python")
            # try to find columns for band, wavelength and A(lam)
            # this is heuristic; inspect df_raw columns
            cols = [c.lower() for c in df_raw.columns]
            band_col = next((c for c in df_raw.columns if 'band' in c.lower()), None)
            lam_col = next((c for c in df_raw.columns if 'lam' in c.lower() and 'eff' in c.lower()), None)
            A_col = next((c for c in df_raw.columns if 'a(' in c.lower() or 'a_lam' in c.lower() or 'a(lambda)' in c.lower()), None)
            if band_col and lam_col and A_col:
                df = df_raw[[band_col, lam_col, A_col]].copy()
                df.columns = ["band", "wavelength_um", "A_lambda_mag"]
            else:
                raise RuntimeError("Fallback text parsing couldn't locate expected columns.")
        except Exception as e:
            raise RuntimeError("Failed to parse NED extinction output: " + str(e))

    # ensure numeric types and units: NED gives lamEff in microns already
    df["wavelength_um"] = pd.to_numeric(df["wavelength_um"], errors="coerce")
    df["A_lambda_mag"] = pd.to_numeric(df["A_lambda_mag"], errors="coerce")
    df = df.dropna(subset=["wavelength_um", "A_lambda_mag"]).reset_index(drop=True)
    return df


def get_A_lambda_at_wavelength(ra, dec, wavelength_um):
    """
    Return interpolated A_lambda (mag) at a specific wavelength (microns).
    Also returns the dataframe of NED band info for inspection.
    """
    df = fetch_ned_extinction_table(ra, dec)
    # sort by wavelength
    df = df.sort_values("wavelength_um").reset_index(drop=True)

    lams = df["wavelength_um"].values
    Avals = df["A_lambda_mag"].values

    # If the requested wavelength is outside the table range, np.interp will
    # clamp to the endpoint values. Optionally raise a warning instead.
    A_interp = float(np.interp(wavelength_um, lams, Avals))
    return A_interp, df


def compute_A_lambda(lamdas, Av=0.2511, Rv=3.1):
    """
    Compute A_lambda extinction curve from Cardelli et al. (1989).
    lamdas must be in Angstroms.
    """
    lam_um = lamdas / 1e4
    y = lam_um**(-1) - 1.82
    a_x = (1.0 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 +
           0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b_x = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 -
           5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    return (a_x + b_x/Rv) * Av

def milky_way_extinction_correction(lamdas, data, Av=0.2511, undo=False):
    """
    Corrects for the extinction caused by light travelling through the dust and
    gas of the Milky Way, as described in Cardelli et al. 1989.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    Av : float 
        The extinction from the Milky Way, found using NED which uses the Schlafly
        & Finkbeiner (2011) recalibration of the Schlegel, Finkbeiner & Davis (1998)
        extinction map based on dust emission measured by COBE/DIRBE and IRAS/ISSA.
        Default is 0.2511, which is the value for IRAS 08339+6517.

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av
    #return A_lam

    #apply to the data
    # if 
    if undo:
        data = data / (10**(0.4*A_lam[:, None, None]))  # put extinction back
    else:
        data = (10**(0.4*A_lam[:, None, None])) * data  # correct extinction

    print(f" Extinction correction applied with Av of {Av} undo = {undo}")
    return data


if __name__ == "__main__":
    
    cube_path = "/Users/janev/Library/CloudStorage/OneDrive-Queen'sUniversity/MNU 2025/cgcg453_red_mosaic.fits"
    cube = Cube(cube_path)


    lamdas = cube.wave.coord()    
    data = cube.data     

    print(f"Go to https://ned.ipac.caltech.edu/byname")
    print("Step 1: enter the galaxy name")
    print("Step 2: copy/paste the RA and Dec coordinates into the extinction calculator: https://ned.ipac.caltech.edu/extinction_calculator")
    ra_deg = 346.235578   
    dec_deg = 19.552296   
    lam_central_um = lamdas.mean() / 1e4 
    print(f" Your central wavelength is {lam_central_um}")
    print(f" Filter for => {lam_central_um} and =< {lam_central_um} in the Central Wavelength column ")



    # --- Step 1: Get Av from NED ---
    # Using Sloan g': 0.385 (central wavelength 0.4925)
    Av = float(input( " Enter the Galactic Extinction (mag) number from the extinction calculator: "))

    # Step 2: Undo correction
    # Edited: DO THE CORRECTION
    data_with_ext = milky_way_extinction_correction(lamdas, data, Av, undo=False)

    # Step 3: Redshift the cube
    # REDSHIFT
    # recalc central wavelength for redshifted cube
    # input new av
    redshifted_cube, lam_new = redshift_wavelength_axis(cube, 0.025, 1.5)
    lamdas_z = redshifted_cube.wave.coord()    
    data_z = redshifted_cube.data 
    lam_central_um = lamdas_z.mean() / 1e4 
    print(f" Your redshifted central wavelength is {lam_central_um}")
    print(f" Filter for => {lam_central_um} and =< {lam_central_um} in the Central Wavelength column ")
    # Using UKIRT J: 0.072 (central wavelength 1.2483)
    Av_z = float(input( " Enter the new Galactic Extinction (mag) number from the extinction calculator: "))


    # Step 4: Reapply extinction correction
    # UNDO CORRECTION
    data_final = milky_way_extinction_correction(lamdas_z, data_z, Av_z, undo=True)

    # --- CHECKS: spectra & extinction curve plots ---
    ny, nx = data.shape[1:]
    y0, x0 = ny // 2, nx // 2
    spec_orig = data[:, y0, x0]
    spec_with_ext = data_with_ext[:, y0, x0]
    spec_final = data_final[:, y0, x0]


    plt.figure(figsize=(10, 6))
    plt.plot(lamdas, spec_orig, label="Original")
    plt.plot(lamdas, spec_with_ext, label="Original with correction")
    plt.plot(lamdas_z, spec_final, label="Redshifted + un-corrected")
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux [cube units]")
    plt.legend()
    plt.title("Check extinction correction + redshift pipeline (central spaxel)")
    plt.show()

    A_lam = compute_A_lambda(lamdas, Av)
    plt.figure(figsize=(8, 5))
    plt.plot(lamdas, A_lam)
    plt.xlabel("Wavelength [Å]")
    plt.ylabel("A(λ) [mag]")
    plt.title("Milky Way Extinction Curve")
    plt.show()
