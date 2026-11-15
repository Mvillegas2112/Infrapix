"""
spherex_fetch.py  -  Prototype: search, download, read and display
SPHEREx spectral images (multi-extension FITS) from NASA/IPAC IRSA.

Requirements:
    pip install astropy astroquery matplotlib numpy

Usage examples:
    # 1) Search by coordinates and list what's available (no download)
    python spherex_fetch.py --ra 127.69444 --dec -39.17760 --list

    # 2) Download rows 0 and 3 of the search result as 0.1-deg cutouts and plot them
    python spherex_fetch.py --ra 127.69444 --dec -39.17760 --pick 0 3 --cutout 0.1

    # 3) Use an object name instead of coordinates
    python spherex_fetch.py --name "M51" --pick 0

    # 4) Open a FITS file you already downloaded
    python spherex_fetch.py --file my_spherex_image.fits
"""

import argparse
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import download_file, conf
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from astropy.wcs import WCS

# SPHEREx files are large (~70 MB full-frame); give the download more time.
conf.remote_timeout = 300

# SPHEREx collections at IRSA change as new releases come out:
#   QR1 "spherex_qr"  -> retired Feb 2026
#   QR2 "spherex_qr2" -> superseded by QR3 + DR1
#   DR1 "spherex_dr1" -> Data Release 1 (Nov 2026), reprocessed year-1 images
#   QR3 "spherex_qr3" -> newer weekly images not yet in DR1
# IRSA's own tutorial searches DR1 + QR3 together. A list is allowed.
# If searches return 0 images, run with --collections to see current names.
DEFAULT_COLLECTION = ["spherex_dr1", "spherex_qr3"]

# ---------------------------------------------------------------------------
# QUICK SETTINGS - used when you press "Run" with no command-line arguments
# (VS Code, Spyder, PyCharm, Jupyter %run ...). Edit these and run.
# ---------------------------------------------------------------------------
RUN_SETTINGS = dict(
    file=None,              # path to a local FITS file -> skips the search
    name=None,              # object name, e.g. "M51" (overrides ra/dec);
                            #   None = ask "Pick an object" when you press Run
    ra=127.69444,           # degrees
    dec=-39.17760,          # degrees
    radius=10,              # search radius, arcsec
    collection=DEFAULT_COLLECTION,
    list=False,             # True = only print the search results
    pick="per_detector",    # "per_detector" = 1 good exposure from each of the 6
                            #   detectors (covers 0.75-5 um); or a list of row
                            #   numbers like [0, 3, 7]; or "all"
    cutout=0.25,            # cutout size in degrees (None = full ~70 MB frame)
    grid=True,              # True = all exposures side by side in ONE window
    # --- blink comparator ---
    blink=True,             # also open a blink comparator window
    blink_mode="same_detector",  # "same_detector": one detector, different DATES
                            #    -> spots things that MOVE or CHANGE brightness
                            # "per_detector": the 6 detectors, different WAVELENGTHS
                            #    -> shows how the target's colour changes
    blink_detector=3,       # which detector for "same_detector" (1-6)
    blink_frames=6,         # how many dates to blink between
    blink_interval_ms=700,  # time each frame stays on screen
    save_gif=True,          # also save the blink as spherex_data/blink.gif
    no_zodi=False,
    mask_flags=False,
    save=False,             # True = save PNGs instead of opening a window
    collections=False,      # True = just list SPHEREx collections at IRSA
)


# ---------------------------------------------------------------------------
# 1. SEARCH
# ---------------------------------------------------------------------------
def list_collections():
    """Print the SPHEREx collections IRSA currently offers via SIA."""
    from astroquery.ipac.irsa import Irsa
    cols = Irsa.list_collections(servicetype="SIA", filter="spherex")
    print(cols)
    return cols


IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"


def _sia_direct(coord, radius_arcsec, collection, timeout=120):
    """
    Plain HTTP GET to IRSA's SIA2 endpoint, parsed as a VOTable.
    Skips the /SIA/capabilities lookup that astroquery/pyvo do first
    (that lookup is what fails when IRSA returns 502 on it).
    """
    import io
    import requests
    from astropy.io.votable import parse_single_table
    params = {
        "COLLECTION": collection,
        "POS": f"CIRCLE {coord.ra.deg} {coord.dec.deg} {radius_arcsec / 3600.0}",
        "RESPONSEFORMAT": "VOTABLE",
    }
    r = requests.get(IRSA_SIA_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return parse_single_table(io.BytesIO(r.content)).to_table()


def _tap_fallback(coord):
    """Second route: IRSA's TAP (ADQL) service over the SPHEREx metadata tables."""
    import pyvo
    query = f"""
        SELECT 'https://irsa.ipac.caltech.edu/' || a.uri AS access_url,
               p.energy_bandpassname, p.time_bounds_lower AS t_min
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {coord.ra.deg}, {coord.dec.deg}), p.poly)
        ORDER BY p.time_bounds_lower
    """
    return pyvo.dal.TAPService(IRSA_TAP_URL).search(query).to_table()


def search_images(coord, radius_arcsec=10, collection=DEFAULT_COLLECTION,
                  retries=3, wait=10):
    """
    Find SPHEREx images covering `coord`. Tries, in order:
      1. astroquery Irsa.query_sia      (the normal way)
      2. direct HTTP call to the SIA2 endpoint (no capabilities lookup)
      3. TAP/ADQL query on the spherex.* tables
    Each network route is retried `retries` times, `wait` s apart.
    """
    import time
    print(f"Searching {collection} around {coord.to_string('hmsdms')} "
          f"(r = {radius_arcsec}\") ...")

    def astroquery_route():
        from astroquery.ipac.irsa import Irsa
        return Irsa.query_sia(pos=(coord, radius_arcsec * u.arcsec),
                              collection=collection)

    routes = [("astroquery SIA", astroquery_route),
              ("direct SIA2 HTTP", lambda: _sia_direct(coord, radius_arcsec, collection)),
              ("TAP fallback", lambda: _tap_fallback(coord))]

    for label, fn in routes:
        for attempt in range(1, retries + 1):
            try:
                results = fn()
                print(f"  [{label}] OK - found {len(results)} images.")
                return results
            except Exception as e:                          # noqa: BLE001
                msg = str(e).splitlines()[0][:150]
                print(f"  [{label}] attempt {attempt}/{retries} failed: {msg}")
                if attempt < retries:
                    time.sleep(wait)
    raise RuntimeError(
        "All IRSA search routes failed. IRSA is probably down or overloaded - "
        "check https://irsa.ipac.caltech.edu in a browser and try again later.")


def summarize(results, max_rows=30):
    """Show the useful columns of the search result (whichever exist)."""
    wanted = ["obs_id", "energy_bandpassname", "em_min", "em_max",
              "t_min", "access_estsize", "access_url"]
    cols = [c for c in wanted if c in results.colnames]
    tbl = results[cols]
    # em_min / em_max are in metres -> show microns
    for c in ("em_min", "em_max"):
        if c in tbl.colnames:
            tbl[c] = np.round(tbl[c] * 1e6, 3)
            tbl[c].unit = u.um
    tbl.pprint(max_lines=max_rows + 5, max_width=200)
    return tbl


# ---------------------------------------------------------------------------
# 2. DOWNLOAD
# ---------------------------------------------------------------------------
def _download_with_retry(url, retries=4, first_wait=15):
    """
    IRSA answers 502/503/504 when it's busy. Wait and try again,
    doubling the wait each time (15 s, 30 s, 60 s ...), and honour the
    server's Retry-After header if it sends one.
    """
    import time
    from urllib.error import HTTPError, URLError
    wait = first_wait
    for attempt in range(1, retries + 1):
        try:
            return download_file(url, cache=True, show_progress=True)
        except HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            pause = int(ra) if ra and ra.isdigit() else wait
            print(f"  IRSA busy (HTTP {e.code}) - waiting {pause} s, "
                  f"retry {attempt}/{retries - 1}")
        except (URLError, TimeoutError) as e:
            if attempt == retries:
                raise
            pause = wait
            print(f"  network problem ({e}) - waiting {pause} s, retry {attempt}/{retries - 1}")
        time.sleep(pause)
        wait *= 2


def download(url, coord=None, cutout_deg=None, outdir="spherex_data"):
    """
    Download one SPHEREx MEF. If `cutout_deg` is given, IRSA cuts a square
    of that size centred on `coord` server-side (much smaller download).
    """
    os.makedirs(outdir, exist_ok=True)
    if cutout_deg and coord is not None:
        url = f"{url}?center={coord.ra.deg},{coord.dec.deg}&size={cutout_deg}"
    print(f"Downloading:\n  {url}")
    tmp = _download_with_retry(url)

    base = url.split("/")[-1].split("?")[0]
    if cutout_deg:
        base = base.replace(".fits", f"_cut{cutout_deg}deg.fits")
    out = os.path.join(outdir, base)
    if not os.path.exists(out):
        shutil.copy(tmp, out)          # keep a named copy; astropy keeps its cache
    print(f"  saved -> {out}")
    return out


# ---------------------------------------------------------------------------
# 3. READ
# ---------------------------------------------------------------------------
def read_spherex(path):
    """
    Open a SPHEREx spectral-image MEF and return a dict with the arrays.

    Extensions (per the SPHEREx Explanatory Supplement):
        IMAGE     calibrated surface brightness (MJy/sr)
        FLAGS     bitmask of bad/flagged pixels
        VARIANCE  per-pixel variance
        ZODI      modelled zodiacal-light background
        PSF       PSF cube for the detector
        WCS-WAVE  lookup table giving wavelength per pixel
    """
    data = {"path": path}
    with fits.open(path) as hdul:
        hdul.info()
        img_hdu = hdul["IMAGE"]
        data["header"] = img_hdu.header.copy()
        data["image"] = img_hdu.data.astype(float)
        data["unit"] = img_hdu.header.get("BUNIT", "")
        data["wcs"] = WCS(img_hdu.header)              # sky coords (RA/Dec)

        for name in ("FLAGS", "VARIANCE", "ZODI"):
            data[name.lower()] = hdul[name].data if name in hdul else None

        # Spectral WCS: each pixel sees a different wavelength (linear variable
        # filter). It is stored as alternate WCS 'W' using the WCS-WAVE table.
        data["wave"] = None
        try:
            wwcs = WCS(img_hdu.header, hdul, key="W")
            ny, nx = data["image"].shape
            yy, xx = np.mgrid[0:ny, 0:nx]
            wave, _bw = wwcs.pixel_to_world_values(xx, yy)
            data["wave"] = np.asarray(wave)              # microns
        except Exception as e:                           # noqa: BLE001
            print(f"  (no spectral WCS decoded: {e})")

    for k in ("OBSID", "DETECTOR", "DATE-OBS", "MJD-AVG"):
        if k in data["header"]:
            print(f"  {k:9s} = {data['header'][k]}")
    return data


def pixel_value_at(data, coord):
    """Return (x, y, value, wavelength) at a sky position."""
    x, y = data["wcs"].world_to_pixel(coord)
    xi, yi = int(round(float(x))), int(round(float(y)))
    ny, nx = data["image"].shape
    if not (0 <= xi < nx and 0 <= yi < ny):
        return None
    wl = data["wave"][yi, xi] if data["wave"] is not None else np.nan
    return xi, yi, data["image"][yi, xi], wl


# ---------------------------------------------------------------------------
# 4. DISPLAY
# ---------------------------------------------------------------------------
def target_inside(data, coord, margin=3):
    """True if the target falls on the image, at least `margin` px from the edge."""
    if coord is None:
        return True
    x, y = data["wcs"].world_to_pixel(coord)
    ny, nx = data["image"].shape
    return margin <= x < nx - margin and margin <= y < ny - margin


def _detector_of_row(results, i):
    import re
    for col in ("energy_bandpassname", "instrument_name"):
        if col in results.colnames:
            m = re.search(r"D(\d)", str(results[col][i]))
            if m:
                return int(m.group(1))
    m = re.search(r"D(\d)_spx", str(results["access_url"][i]))
    return int(m.group(1)) if m else None


def _style_sky_axes(ax, data, label_dec=True):
    """Fix RA on the bottom, Dec on the left, and draw a N/E compass.
    SPHEREx frames are rotated on the sky, so without this matplotlib
    puts the tick labels on whatever edge it likes."""
    ra, dec = ax.coords[0], ax.coords[1]
    ra.set_axislabel("Right Ascension (sky longitude)")
    dec.set_axislabel("Declination (sky latitude)" if label_dec else "")
    ra.set_ticks_position("b"); ra.set_ticklabel_position("b"); ra.set_axislabel_position("b")
    dec.set_ticks_position("l"); dec.set_ticklabel_position("l"); dec.set_axislabel_position("l")
    ra.set_ticks(number=3); dec.set_ticks(number=3)

    # compass: which way is North and East on this (rotated) image
    ny, nx = data["image"].shape
    w = data["wcs"].celestial
    c0 = w.pixel_to_world(nx * 0.85, ny * 0.18)
    L = 0.12 * min(nx, ny)
    step = 30 * u.arcsec
    for lab, c1 in (("N", c0.directional_offset_by(0 * u.deg, step)),
                    ("E", c0.directional_offset_by(90 * u.deg, step))):
        x0, y0 = w.world_to_pixel(c0)
        x1, y1 = w.world_to_pixel(c1)
        dx, dy = x1 - x0, y1 - y0
        n = np.hypot(dx, dy) or 1
        dx, dy = dx / n * L, dy / n * L
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
        ax.text(x0 + dx * 1.35, y0 + dy * 1.35, lab, color="white",
                fontsize=8, ha="center", va="center")


# Approximate wavelength coverage of each SPHEREx detector (microns).
DETECTOR_BANDS = {1: (0.75, 1.09), 2: (1.10, 1.62), 3: (1.63, 2.41),
                  4: (2.42, 3.82), 5: (3.83, 4.41), 6: (4.42, 5.00)}


def describe(data):
    """Pull human-readable facts (detector, wavelength range, date) from a file."""
    import re
    from astropy.time import Time
    h = data["header"]
    info = {}

    det = h.get("DETECTOR")
    if det is None:                                    # e.g. ..._2D3_spx_...fits
        m = re.search(r"D(\d)_spx", os.path.basename(data["path"]))
        det = int(m.group(1)) if m else None
    info["detector"] = det

    if data["wave"] is not None and np.isfinite(data["wave"]).any():
        info["wl_range"] = (np.nanmin(data["wave"]), np.nanmax(data["wave"]))
        info["wl_exact"] = True
    elif det in DETECTOR_BANDS:
        info["wl_range"] = DETECTOR_BANDS[det]
        info["wl_exact"] = False
    else:
        info["wl_range"], info["wl_exact"] = None, False

    date = h.get("DATE-OBS") or h.get("DATE")
    if not date and "MJD-AVG" in h:
        date = Time(h["MJD-AVG"], format="mjd").iso
    info["date"] = str(date)[:16].replace("T", " ") if date else "unknown date"

    ny, nx = data["image"].shape
    from astropy.wcs.utils import proj_plane_pixel_scales
    scale = float(np.mean(proj_plane_pixel_scales(data["wcs"].celestial))) * 3600  # arcsec/px
    info["size"] = f"{nx}x{ny} px  (~{nx*scale/60:.1f}' x {ny*scale/60:.1f}' of sky)"
    return info


def show(data, coord=None, subtract_zodi=True, mask_flags=False, save=None,
         index=1, total=1):
    """
    One figure per SPHEREx file:
      left  = the picture: brightness of the sky in this exposure
      right = which wavelength (colour of infrared light) each pixel measured
    """
    info = describe(data)
    img = data["image"].copy()
    if subtract_zodi and data.get("zodi") is not None:
        img = img - data["zodi"]
    if mask_flags and data.get("flags") is not None:
        img[data["flags"] != 0] = np.nan

    norm = ImageNormalize(img[np.isfinite(img)], interval=ZScaleInterval(),
                          stretch=AsinhStretch())
    has_wave = data["wave"] is not None
    fig = plt.figure(figsize=(14 if has_wave else 8, 7))

    # ---- overall title: what this file is ----
    det = info["detector"]
    wl = info["wl_range"]
    wl_txt = (f"{wl[0]:.2f}-{wl[1]:.2f} um" + ("" if info["wl_exact"] else " (approx.)")
              if wl else "wavelength unknown")
    fig.suptitle(f"SPHEREx exposure {index} of {total}   |   detector {det or '?'}"
                 f"   |   infrared light {wl_txt}   |   observed {info['date']}",
                 fontsize=12, fontweight="bold")

    # ---- left: the picture ----
    ax = fig.add_subplot(1, 2 if has_wave else 1, 1, projection=data["wcs"])
    im = ax.imshow(img, origin="lower", cmap="inferno", norm=norm)
    ax.set_title("The image: how bright each patch of sky is\n"
                 + ("(smooth zodiacal-light glow removed)" if subtract_zodi
                    else "(raw, zodiacal-light glow included)"), fontsize=10)
    _style_sky_axes(ax, data)
    ny, nx = data["image"].shape
    shrink = float(np.clip(ny / nx * 1.1, 0.35, 1.0))   # keep colorbar ~image height
    cb = fig.colorbar(im, ax=ax, fraction=0.046, shrink=shrink)
    cb.set_label("Brightness  [MJy/sr]   (brighter = more infrared light)")

    if coord is not None:
        ax.scatter(coord.ra.deg, coord.dec.deg, transform=ax.get_transform("world"),
                   s=200, facecolor="none", edgecolor="cyan", lw=1.8,
                   label="your target")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.6)
        hit = pixel_value_at(data, coord) if target_inside(data, coord, 0) else None
        if not hit:
            ax.text(0.5, 0.5, "Your target is OUTSIDE this image\n(it fell near the detector edge)",
                    transform=ax.transAxes, color="cyan", ha="center", fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))
        if hit:
            txt = f"At your target: {hit[2]:.3g} MJy/sr"
            if np.isfinite(hit[3]):
                txt += f"\nmeasured at {hit[3]:.3f} um"
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, color="white",
                    fontsize=9, va="bottom",
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))

    ax.text(0.02, 0.98, info["size"], transform=ax.transAxes, color="white",
            fontsize=8, va="top",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

    # ---- right: wavelength map ----
    if has_wave:
        ax2 = fig.add_subplot(1, 2, 2, projection=data["wcs"])
        w = ax2.imshow(data["wave"], origin="lower", cmap="viridis")
        ax2.set_title("Which wavelength each pixel measured\n"
                      "(SPHEREx's filter changes colour across the detector)",
                      fontsize=10)
        _style_sky_axes(ax2, data, label_dec=False)
        cb2 = fig.colorbar(w, ax=ax2, fraction=0.046, shrink=shrink)
        cb2.set_label("Wavelength  [microns]")

    fig.text(0.5, 0.01, f"file: {os.path.basename(data['path'])}",
             ha="center", fontsize=8, color="gray")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    if save:
        plt.savefig(save, dpi=130)
        print(f"  figure saved -> {save}")
    else:
        if total > 1 and index < total:
            print("  >> Close the window to see the next image.")
        plt.show()


def show_grid(datas, coord=None, subtract_zodi=True, mask_flags=False, save=None):
    """All downloaded exposures in ONE window, sorted by wavelength."""
    import math
    items = sorted(datas, key=lambda d: (describe(d)["wl_range"] or (99, 99))[0])
    n = len(items)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(5.2 * cols, 4.8 * rows))
    fig.suptitle(f"{n} SPHEREx exposures of your target, sorted from short to long "
                 "infrared wavelength\n(each panel = one exposure; brighter = more light; "
                 "cyan circle = your target)", fontsize=12, fontweight="bold")
    for k, d in enumerate(items, start=1):
        info = describe(d)
        img = d["image"].copy()
        if subtract_zodi and d.get("zodi") is not None:
            img = img - d["zodi"]
        if mask_flags and d.get("flags") is not None:
            img[d["flags"] != 0] = np.nan
        norm = ImageNormalize(img[np.isfinite(img)], interval=ZScaleInterval(),
                              stretch=AsinhStretch())
        ax = fig.add_subplot(rows, cols, k, projection=d["wcs"])
        ax.imshow(img, origin="lower", cmap="inferno", norm=norm)
        _style_sky_axes(ax, d, label_dec=(k - 1) % cols == 0)
        wl = info["wl_range"]
        title = f"Detector {info['detector'] or '?'}"
        if coord is not None and target_inside(d, coord, 0):
            hit = pixel_value_at(d, coord)
            title += f"  -  target seen at {hit[3]:.2f} um" if np.isfinite(hit[3]) else ""
            ax.scatter(coord.ra.deg, coord.dec.deg, transform=ax.get_transform("world"),
                       s=150, facecolor="none", edgecolor="cyan", lw=1.6)
        elif coord is not None:
            title += "  -  target off image"
        if wl:
            title += f"\nthis exposure covers {wl[0]:.2f}-{wl[1]:.2f} um, {info['date'][:10]}"
        ax.set_title(title, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if save:
        plt.savefig(save, dpi=120)
        print(f"  figure saved -> {save}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 5. BLINK COMPARATOR
# ---------------------------------------------------------------------------
def common_grid(coord, size_deg=0.25, pix_arcsec=6.15):
    """A north-up, east-left sky grid centred on the target (SPHEREx-ish pixels)."""
    n = int(round(size_deg * 3600 / pix_arcsec))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.cdelt = [-pix_arcsec / 3600, pix_arcsec / 3600]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    return w, (n, n)


def reproject_to(data, w_out, shape_out, subtract_zodi=True):
    """
    Resample one exposure onto the common grid (bilinear, pure numpy).
    Every SPHEREx frame is rotated differently on the sky; without this the
    stars would jump around when blinking.
    """
    img = data["image"].astype(float)
    if subtract_zodi and data.get("zodi") is not None:
        img = img - data["zodi"]
    ny_out, nx_out = shape_out
    yy, xx = np.mgrid[0:ny_out, 0:nx_out]
    ra, dec = w_out.pixel_to_world_values(xx, yy)
    sx, sy = data["wcs"].celestial.world_to_pixel_values(ra, dec)

    ny, nx = img.shape
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    x0c, y0c = np.clip(x0, 0, nx - 2), np.clip(y0, 0, ny - 2)
    out = (img[y0c, x0c] * (1 - fx) * (1 - fy) + img[y0c, x0c + 1] * fx * (1 - fy)
           + img[y0c + 1, x0c] * (1 - fx) * fy + img[y0c + 1, x0c + 1] * fx * fy)
    out[~ok] = np.nan
    return out


def fetch_same_detector(results, coord, detector, n_frames, cutout):
    """Pick exposures from ONE detector spread out over time (oldest -> newest)."""
    rows = [i for i in range(len(results)) if _detector_of_row(results, i) == detector]
    if not rows:
        print(f"No exposures from detector {detector} here.")
        return []
    if "t_min" in results.colnames:
        rows.sort(key=lambda i: float(results["t_min"][i]))
    # spread the picks across the whole time span, with spares for edge/failed ones
    order = list(dict.fromkeys(int(round(x)) for x in
                               np.linspace(0, len(rows) - 1, min(len(rows), n_frames * 2))))
    datas = []
    for j in order:
        if len(datas) >= n_frames:
            break
        i = rows[j]
        print(f"\nBlink frame {len(datas) + 1}/{n_frames}: detector {detector}, row {i}")
        d = _safe_get(results["access_url"][i], coord, cutout)
        if d is not None and target_inside(d, coord):
            datas.append(d)
        elif d is not None:
            print("  target too close to the edge - skipping")
    return datas


def blink(datas, coord, size_deg=0.25, interval_ms=700, same_scale=True,
          subtract_zodi=True, gif_path=None):
    """
    Blink comparator: flips between exposures aligned on the same sky grid.
      left  = the blinking image
      right = this frame minus the median of all frames
              (red = brighter than usual, blue = fainter than usual)
    Keys: SPACE pause/play,  LEFT/RIGHT step,  close window to finish.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    if len(datas) < 2:
        print("Need at least 2 exposures to blink.")
        return
    w, shape = common_grid(coord, size_deg)
    infos = [describe(d) for d in datas]
    frames = [reproject_to(d, w, shape, subtract_zodi) for d in datas]
    # sort by date (same detector) - time order matters for spotting motion
    order = np.argsort([i["date"] for i in infos])
    frames = [frames[k] for k in order]
    infos = [infos[k] for k in order]

    # brightness scale: shared (same detector) or per frame (different wavelengths)
    if same_scale:
        lo, hi = ZScaleInterval().get_limits(np.concatenate(
            [f[np.isfinite(f)] for f in frames]))
        scaled = [np.clip((f - lo) / (hi - lo), 0, 1) for f in frames]
    else:
        scaled = []
        for f in frames:
            lo, hi = ZScaleInterval().get_limits(f[np.isfinite(f)])
            scaled.append(np.clip((f - lo) / (hi - lo), 0, 1))
    med = np.nanmedian(np.stack(scaled), axis=0)
    diffs = [s - med for s in scaled]

    fig = plt.figure(figsize=(13, 6.5))
    fig.suptitle("Blink comparator - same patch of sky, frames aligned north-up\n"
                 "(things that jump or flicker between frames moved or changed "
                 "brightness)", fontsize=11, fontweight="bold")
    ax1 = fig.add_subplot(1, 2, 1, projection=w)
    ax2 = fig.add_subplot(1, 2, 2, projection=w)
    im1 = ax1.imshow(scaled[0], origin="lower", cmap="gray", vmin=0, vmax=1)
    im2 = ax2.imshow(diffs[0], origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    for ax, lab in ((ax1, True), (ax2, False)):
        ax.coords[0].set_axislabel("Right Ascension  (East is LEFT)")
        ax.coords[1].set_axislabel("Declination  (North is UP)" if lab else "")
        ax.coords[0].set_ticks(number=4); ax.coords[1].set_ticks(number=4)
        cx, cy = (shape[1] - 1) / 2, (shape[0] - 1) / 2
        ax.add_patch(Circle((cx, cy), radius=shape[0] * 0.08, fill=False,
                            edgecolor="cyan", lw=1.0, ls="--"))
    ax2.set_title("Difference from the average frame\n"
                  "red = brighter than usual   |   blue = fainter than usual", fontsize=10)
    cb = fig.colorbar(im2, ax=ax2, fraction=0.046)
    cb.set_label("change (fraction of display range)")
    status = fig.text(0.5, 0.02, "", ha="center", fontsize=9, color="gray")

    state = {"k": 0, "running": True}

    def label(k):
        i = infos[k]
        wl = i["wl_range"]
        wl_txt = f"{wl[0]:.2f}-{wl[1]:.2f} um" if wl else ""
        return (f"Frame {k + 1} of {len(frames)}   |   detector {i['detector']}"
                f"   |   {wl_txt}   |   observed {i['date']}")

    def draw(k):
        im1.set_data(scaled[k]); im2.set_data(diffs[k])
        ax1.set_title(label(k), fontsize=10)
        status.set_text("SPACE = pause/play     LEFT / RIGHT = step     "
                        + ("(playing)" if state["running"] else "(paused)"))
        return im1, im2

    def step(_frame):
        state["k"] = (state["k"] + 1) % len(frames)
        return draw(state["k"])

    def on_key(ev):
        if ev.key == " ":
            state["running"] = not state["running"]
            (anim.event_source.start if state["running"] else anim.event_source.stop)()
        elif ev.key in ("right", "left"):
            state["k"] = (state["k"] + (1 if ev.key == "right" else -1)) % len(frames)
        draw(state["k"]); fig.canvas.draw_idle()

    draw(0)
    plt.tight_layout(rect=(0, 0.04, 1, 0.92))
    anim = FuncAnimation(fig, step, interval=interval_ms, blit=False,
                         cache_frame_data=False, save_count=len(frames))
    fig.canvas.mpl_connect("key_press_event", on_key)

    if gif_path:
        # separate, index-driven animation for saving, so GIF frame N = frame N
        gif_anim = FuncAnimation(fig, lambda i: draw(i), frames=len(frames),
                                 init_func=lambda: draw(0), blit=False)
        status.set_text("")
        if os.path.dirname(gif_path):
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        gif_anim.save(gif_path, writer=PillowWriter(fps=1000 / interval_ms))
        print(f"  blink GIF saved -> {gif_path}")
        state["k"] = 0; draw(0)
    fig._spherex_anim = anim                              # keep it alive while shown
    if matplotlib_is_interactive():
        plt.show()


def matplotlib_is_interactive():
    import matplotlib
    return matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="SPHEREx FITS fetcher / viewer (prototype)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--file", help="open a local FITS file instead of searching")
    g.add_argument("--name", help="object name resolved by Sesame (e.g. 'M51')")
    p.add_argument("--ra", type=float, help="RA in degrees")
    p.add_argument("--dec", type=float, help="Dec in degrees")
    p.add_argument("--radius", type=float, default=10, help="search radius, arcsec")
    p.add_argument("--collection", nargs="+", default=DEFAULT_COLLECTION,
                   help="one or more SIA collection names")
    p.add_argument("--list", action="store_true", help="only list search results")
    p.add_argument("--pick", nargs="+", default=["per_detector"],
                   help="row numbers, 'all', or 'per_detector' (default)")
    p.add_argument("--blink", action="store_true", help="open a blink comparator")
    p.add_argument("--blink-mode", default="same_detector",
                   choices=["same_detector", "per_detector"])
    p.add_argument("--blink-detector", type=int, default=3)
    p.add_argument("--blink-frames", type=int, default=6)
    p.add_argument("--blink-interval-ms", type=int, default=700)
    p.add_argument("--save-gif", action="store_true")
    p.add_argument("--no-grid", action="store_true",
                   help="one window per exposure instead of a single grid")
    p.add_argument("--cutout", type=float, default=None,
                   help="cutout size in degrees (omit for full frame)")
    p.add_argument("--no-zodi", action="store_true", help="don't subtract ZODI")
    p.add_argument("--mask-flags", action="store_true", help="blank any flagged pixel")
    p.add_argument("--save", action="store_true", help="save PNGs instead of showing")
    p.add_argument("--collections", action="store_true",
                   help="list SPHEREx collections available at IRSA and exit")
    import sys
    if len(sys.argv) == 1:                      # pressed "Run" with no arguments
        print("No arguments given -> using RUN_SETTINGS at the top of the file.")
        a = argparse.Namespace(**RUN_SETTINGS)
        if not a.name and not a.file:
            # Asked here (not at the top of the file) so that importing this
            # module - e.g. from the Flask backend app.py - never waits for input.
            typed = input("Pick an object (blank = use RA/Dec from RUN_SETTINGS): ").strip()
            a.name = typed or None
    else:
        a = p.parse_args()

    if a.collections:
        list_collections()
        return

    if a.file:
        d = read_spherex(a.file)
        show(d, subtract_zodi=not a.no_zodi, mask_flags=a.mask_flags,
             save=a.file.replace(".fits", ".png") if a.save else None)
        return

    if a.name:
        coord = SkyCoord.from_name(a.name)
    elif a.ra is not None and a.dec is not None:
        coord = SkyCoord(a.ra * u.deg, a.dec * u.deg)
    else:
        p.error("give --file, --name, or --ra and --dec")

    results = search_images(coord, a.radius, a.collection)
    if len(results) == 0:
        print(f"No images found in {a.collection}. Collections IRSA lists now:")
        try:
            list_collections()
        except Exception as e:                           # noqa: BLE001
            print(f"  (couldn't list collections: {e})")
        print("Put the right name(s) in RUN_SETTINGS['collection'] or use --collection.")
        return
    summarize(results)
    if a.list:
        return

    datas = fetch_selected(results, coord, a.pick, a.cutout)
    if getattr(a, "blink", False):
        print("\n(The blink comparator opens after you close the image window.)")
    if not datas:
        print("Nothing to show: the target fell outside every downloaded image.")
        return
    if getattr(a, "grid", True) and not getattr(a, "no_grid", False) and len(datas) > 1:
        show_grid(datas, coord, subtract_zodi=not a.no_zodi, mask_flags=a.mask_flags,
                  save="spherex_data/grid.png" if a.save else None)
    else:
        for n, d in enumerate(datas, start=1):
            show(d, coord, subtract_zodi=not a.no_zodi, mask_flags=a.mask_flags,
                 index=n, total=len(datas),
                 save=d["path"].replace(".fits", ".png") if a.save else None)
    if getattr(a, "blink", False):
        run_blink(a, results, coord, datas)


def run_blink(a, results, coord, datas):
    mode = getattr(a, "blink_mode", "same_detector")
    size = a.cutout or 0.25
    os.makedirs("spherex_data", exist_ok=True)
    gif = "spherex_data/blink.gif" if getattr(a, "save_gif", True) else None
    if mode == "same_detector":
        bdatas = fetch_same_detector(results, coord, getattr(a, "blink_detector", 3),
                                     getattr(a, "blink_frames", 6), a.cutout)
        same_scale = True
    else:
        bdatas, same_scale = datas, False
    blink(bdatas, coord, size, getattr(a, "blink_interval_ms", 700),
          same_scale=same_scale, subtract_zodi=not a.no_zodi, gif_path=gif)


def fetch_selected(results, coord, pick, cutout, tries_per_detector=4):
    """
    Download the chosen exposures and return them read in.
    pick = "per_detector" -> for each detector 1..6, the first exposure in
           which the target is safely inside the image (skips edge cases)
    pick = "all"          -> every row (careful: can be hundreds)
    pick = [0, 3, ...]    -> those row numbers
    """
    if isinstance(pick, (list, tuple)) and len(pick) == 1 and isinstance(pick[0], str):
        pick = pick[0]                                  # from the command line
    if isinstance(pick, str) and pick not in ("per_detector", "all"):
        pick = [int(x) for x in pick.split(",")]
    if isinstance(pick, (list, tuple)):
        pick = [int(x) for x in pick]

    datas = []
    if pick == "per_detector":
        by_det = {}
        for i in range(len(results)):
            by_det.setdefault(_detector_of_row(results, i), []).append(i)
        for det in sorted(k for k in by_det if k is not None):
            for i in by_det[det][:tries_per_detector]:
                print(f"\nDetector {det}: trying row {i}")
                d = _safe_get(results["access_url"][i], coord, cutout)
                if d is None:
                    continue
                if target_inside(d, coord):
                    datas.append(d)
                    break
                print("  target too close to the edge of this exposure - trying another")
        return datas

    rows = range(len(results)) if pick == "all" else pick
    for i in rows:
        d = _safe_get(results["access_url"][i], coord, cutout)
        if d is not None:
            datas.append(d)
    return datas


def _safe_get(url, coord, cutout):
    """Download + read one exposure; on failure print why and return None
    so the rest of the run keeps going."""
    try:
        return read_spherex(download(url, coord, cutout))
    except Exception as e:                                # noqa: BLE001
        print(f"  !! skipped this exposure: {str(e).splitlines()[0][:120]}")
        return None

if __name__ == "__main__":
    main()
