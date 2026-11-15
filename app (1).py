
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

import hashlib
import math
import os
import re
import traceback

# Matplotlib sin interfaz gráfica para el servidor.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from astropy.visualization import (
    ImageNormalize,
    ZScaleInterval,
    AsinhStretch,
)
from astropy.time import Time

# Importamos el módulo completo para corregir también Blink.
import spherex_fetch as sx

from spherex_fetch import (
    search_images,
    SkyCoord,
    u,
    run_blink,
    download,
    read_spherex,
    describe,
)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "spherex_data")
PREVIEW_DIR = os.path.join(DATA_DIR, "web_previews")

MAX_PREVIEWS = 6

os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =========================================================
# SAFE CONVERSIONS
# =========================================================

def safe_text(value, default=""):

    if value is None or np.ma.is_masked(value):
        return default

    try:
        result = str(value).strip()

        if result.lower() in {
            "", "masked", "--", "nan", "none", "null"
        }:
            return default

        return result

    except Exception:
        return default


def optional_float(value):
    """Convert valid values to float, ignoring None and masked data."""

    if value is None or np.ma.is_masked(value):
        return None

    try:
        result = float(value)
        return result if math.isfinite(result) else None

    except (TypeError, ValueError, OverflowError):
        return None


def query_float(name, default, minimum=None, maximum=None):

    raw = request.args.get(name)

    if raw is None or raw.strip() == "":
        value = float(default)
    else:
        value = optional_float(raw)

    if value is None:
        raise ValueError(
            f"El parámetro '{name}' debe ser un número válido."
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"El parámetro '{name}' debe ser al menos {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"El parámetro '{name}' no puede superar {maximum}."
        )

    return value


# =========================================================
# RESOLVE OBJECT NAMES
# =========================================================

class TargetResolutionError(ValueError):
    pass


def valid_coordinate(coord):

    try:
        coord = SkyCoord(coord).icrs

        ra = optional_float(coord.ra.deg)
        dec = optional_float(coord.dec.deg)

        if ra is None or dec is None:
            return None

        if not (0 <= ra < 360 and -90 <= dec <= 90):
            return None

        return coord

    except Exception:
        return None


def resolve_catalog_name(name):
    """
    Try multiple astronomical name resolvers.

    1. MAST/SANTA using SIMBAD and NED.
    2. CDS Sesame.
    3. JPL Horizons for planets, asteroids and comets.

    Returns:
        coordinate, resolver name, moving-object flag
    """

    errors = []

    # -----------------------------------------------------
    # 1. MAST / SIMBAD / NED
    # -----------------------------------------------------

    try:

        from astroquery.mast import Observations

        print(f"Trying MAST resolver for: {name}")

        coord = Observations.resolve_object(name)

        coord = valid_coordinate(coord)

        if coord is not None:
            return (
                coord,
                "MAST / SIMBAD / NED",
                False
            )

    except Exception as error:

        errors.append(
            "MAST: "
            + str(error).splitlines()[0][:180]
        )

    # -----------------------------------------------------
    # 2. CDS SESAME
    # -----------------------------------------------------

    try:

        print(f"Trying CDS Sesame for: {name}")

        coord = SkyCoord.from_name(name)

        coord = valid_coordinate(coord)

        if coord is not None:
            return (
                coord,
                "CDS Sesame",
                False
            )

    except Exception as error:

        errors.append(
            "CDS Sesame: "
            + str(error).splitlines()[0][:180]
        )

    # -----------------------------------------------------
    # 3. JPL HORIZONS
    # -----------------------------------------------------

    try:

        from astroquery.jplhorizons import Horizons

        print(f"Trying JPL Horizons for: {name}")

        epoch_jd = Time.now().jd

        # First try the default search:
        # planets, moons, spacecraft and then small bodies.
        # If that fails, specifically try the small-body database.
        for id_type in (None, "smallbody"):

            try:

                options = {
                    "id": name,
                    "location": "500@399",
                    "epochs": epoch_jd,
                }

                if id_type is not None:
                    options["id_type"] = id_type

                target = Horizons(**options)

                ephemeris = target.ephemerides()

                if len(ephemeris) == 0:
                    continue

                ra = optional_float(ephemeris["RA"][0])
                dec = optional_float(ephemeris["DEC"][0])

                if ra is None or dec is None:
                    continue

                coord = SkyCoord(
                    ra * u.deg,
                    dec * u.deg,
                    frame="icrs"
                )

                return (
                    coord,
                    "JPL Horizons (posición actual)",
                    True
                )

            except Exception as error:

                errors.append(
                    "JPL Horizons: "
                    + str(error).splitlines()[0][:180]
                )

    except Exception as error:

        errors.append(
            "JPL Horizons: "
            + str(error).splitlines()[0][:180]
        )

    # -----------------------------------------------------
    # NO RESOLVER FOUND THE OBJECT
    # -----------------------------------------------------

    detail = " | ".join(errors[-4:])

    raise TargetResolutionError(
        f"No se pudo identificar '{name}' en los catálogos "
        "consultados. Probá con el nombre oficial, una "
        "designación de catálogo o sus coordenadas. "
        f"Detalles: {detail}"
    )


def resolve_target():
    """
    Resolve a typed object name or use manual RA / Dec.

    Returns:
        coordinate, object name, resolver, moving flag
    """

    name = (request.args.get("name") or "").strip()

    # If the user types a name, it takes priority over RA/Dec.
    if name:

        if len(name) > 200:
            raise TargetResolutionError(
                "El nombre del objeto es demasiado largo."
            )

        coord, resolver, moving = resolve_catalog_name(name)

        print(
            f"\nResolved object: {name}"
            f"\nResolver: {resolver}"
            f"\nRA: {coord.ra.deg:.8f}"
            f"\nDec: {coord.dec.deg:.8f}"
        )

        return coord, name, resolver, moving

    # If there is no name, use manual coordinates.
    ra_text = request.args.get("ra")
    dec_text = request.args.get("dec")

    if not ra_text or not dec_text:

        raise TargetResolutionError(
            "Escribí el nombre de un objeto o completá RA y Dec."
        )

    ra = optional_float(ra_text)
    dec = optional_float(dec_text)

    if ra is None or dec is None:

        raise TargetResolutionError(
            "RA y Dec deben ser números válidos."
        )

    if not (0 <= ra < 360 and -90 <= dec <= 90):

        raise TargetResolutionError(
            "RA debe estar entre 0° y menos de 360° "
            "y Dec entre -90° y 90°."
        )

    coord = SkyCoord(
        ra * u.deg,
        dec * u.deg,
        frame="icrs"
    )

    return coord, "", "Coordenadas manuales", False


def detector_from_band(band):

    match = re.search(r"D(\d)", str(band))

    return int(match.group(1)) if match else None


# =========================================================
# HANDLE MISSING OBSERVATION DATES
# =========================================================

def sorted_rows_by_time(results, rows):
    """Sort observations safely, even when t_min is missing."""

    if "t_min" not in results.colnames:
        return list(rows)

    def time_key(index):

        value = optional_float(results["t_min"][index])

        if value is None:
            return float("inf")

        return value

    return sorted(rows, key=time_key)


def choose_preview_rows(results, maximum=MAX_PREVIEWS):

    rows = sorted_rows_by_time(
        results,
        list(range(len(results)))
    )

    if len(rows) <= maximum:
        return rows

    if maximum <= 1:
        return [rows[0]]

    positions = [
        round(i * (len(rows) - 1) / (maximum - 1))
        for i in range(maximum)
    ]

    return [rows[p] for p in positions]


# =========================================================
# SAFE BLINK OBSERVATION SELECTION
# =========================================================

def safe_fetch_same_detector(
    results,
    coord,
    detector,
    n_frames,
    cutout
):
    """Replacement for the original fetch_same_detector."""

    rows = [
        i for i in range(len(results))
        if sx._detector_of_row(results, i) == detector
    ]

    if not rows:

        print(f"No exposures from detector {detector} here.")

        return []

    # Handle None and masked t_min values.
    rows = sorted_rows_by_time(results, rows)

    count = min(
        len(rows),
        max(1, int(n_frames) * 2)
    )

    positions = np.linspace(
        0,
        len(rows) - 1,
        count
    )

    selected = list(dict.fromkeys(
        int(round(p)) for p in positions
    ))

    data_frames = []

    for position in selected:

        if len(data_frames) >= n_frames:
            break

        row_index = rows[position]

        access_url = safe_text(
            results["access_url"][row_index]
        )

        if not access_url:

            print(
                f"Row {row_index}: missing access_url; skipping"
            )

            continue

        print(
            f"\nBlink frame {len(data_frames) + 1}/{n_frames}: "
            f"detector {detector}, row {row_index}"
        )

        item = sx._safe_get(
            access_url,
            coord,
            cutout
        )

        if item is not None and sx.target_inside(item, coord):

            data_frames.append(item)

        elif item is not None:

            print("Target too close to the edge - skipping")

    return data_frames


# run_blink uses fetch_same_detector internally.
# Replace it with the safe version above.
sx.fetch_same_detector = safe_fetch_same_detector


# =========================================================
# CREATE PNG PREVIEWS FROM REAL FITS DATA
# =========================================================

def create_preview(data, coord, output_path, title):

    image = np.asarray(
        data["image"],
        dtype=float
    ).copy()

    # Remove zodiacal background when available.
    if data.get("zodi") is not None:

        zodi = np.asarray(
            data["zodi"],
            dtype=float
        )

        if zodi.shape == image.shape:
            image -= zodi

    finite = np.isfinite(image)

    if not finite.any():

        raise ValueError(
            "La imagen FITS no contiene píxeles válidos."
        )

    norm = ImageNormalize(
        image[finite],
        interval=ZScaleInterval(),
        stretch=AsinhStretch()
    )

    fig = plt.figure(
        figsize=(6.4, 5.2),
        facecolor="#050510"
    )

    try:

        ax = fig.add_subplot(
            111,
            projection=data["wcs"]
        )

        ax.set_facecolor("#050510")

        ax.imshow(
            image,
            origin="lower",
            cmap="inferno",
            norm=norm
        )

        ax.set_title(
            title,
            color="white",
            fontsize=11,
            pad=12
        )

        ax.coords[0].set_axislabel(
            "Right Ascension",
            color="#cbd5e1"
        )

        ax.coords[1].set_axislabel(
            "Declination",
            color="#cbd5e1"
        )

        ax.tick_params(
            colors="#cbd5e1",
            labelsize=8
        )

        # Mark requested sky coordinates.
        ax.scatter(
            coord.ra.deg,
            coord.dec.deg,
            transform=ax.get_transform("world"),
            s=95,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.6
        )

        fig.text(
            0.5,
            0.015,
            "SPHEREx infrared data · cyan circle = requested coordinates",
            ha="center",
            color="#94a3b8",
            fontsize=8
        )

        fig.tight_layout(
            rect=(0, 0.04, 1, 1)
        )

        fig.savefig(
            output_path,
            dpi=135,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight"
        )

    finally:

        plt.close(fig)


# =========================================================
# DOWNLOAD AND PROCESS ONE OBSERVATION
# =========================================================

def make_preview(results, row_index, coord, cutout):

    row = results[row_index]

    band = (
        safe_text(row["energy_bandpassname"])
        if "energy_bandpassname" in results.colnames
        else "SPHEREx"
    )

    obs_id = (
        safe_text(row["obs_id"])
        if "obs_id" in results.colnames
        else f"row-{row_index}"
    )

    detector = detector_from_band(band)

    access_url = (
        safe_text(row["access_url"])
        if "access_url" in results.colnames
        else ""
    )

    if not access_url:

        raise ValueError(
            "La observación no contiene una URL de descarga."
        )

    # Unique cache key based on target and observation.
    key_text = (
        f"{access_url}|"
        f"{coord.ra.deg:.8f}|"
        f"{coord.dec.deg:.8f}|"
        f"{cutout}"
    )

    filename = (
        hashlib.sha1(key_text.encode("utf-8")).hexdigest()
        + ".png"
    )

    png_path = os.path.join(
        PREVIEW_DIR,
        filename
    )

    # Download real FITS data.
    fits_path = download(
        access_url,
        coord,
        cutout_deg=cutout,
        outdir=DATA_DIR
    )

    data = read_spherex(fits_path)

    info = describe(data)

    if not os.path.exists(png_path):

        create_preview(
            data,
            coord,
            png_path,
            f"SPHEREx · Detector {info.get('detector') or detector or '?'}"
        )

    wavelength = info.get("wl_range")

    wavelength_text = ""

    if wavelength is not None and len(wavelength) >= 2:

        low = optional_float(wavelength[0])
        high = optional_float(wavelength[1])

        if low is not None and high is not None:

            wavelength_text = (
                f"{low:.2f}–{high:.2f} μm"
            )

    return {
        "index": int(row_index),
        "band": band,
        "detector": info.get("detector") or detector,
        "obs_id": obs_id,
        "date": info.get("date", "Fecha desconocida"),
        "wavelength": wavelength_text,
        "image_url": f"/api/preview/{filename}"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/")
def index():
    """Serve the website itself, so it can be opened at http://localhost:5000"""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "message": "Infrapix backend running"
    })


# =========================================================
# SEARCH SPHEREX IMAGES
# =========================================================

@app.route("/api/search")
def api_search():

    try:

        coord, object_name, resolver, is_moving = resolve_target()

        radius = query_float(
            "radius",
            25,
            0.01,
            3600
        )

        cutout = query_float(
            "cutout",
            0.25,
            0.01,
            1.0
        )

        print(
            f"\nSearch target: {object_name or 'RA/Dec'} | "
            f"RA={coord.ra.deg:.8f}°, "
            f"Dec={coord.dec.deg:.8f}°"
        )

        print("Resolver:", resolver)

        # Search actual NASA/IPAC IRSA observations.
        results = search_images(
            coord,
            radius_arcsec=radius
        )

        target_info = {
            "target_name": object_name,
            "target_resolver": resolver,
            "moving_object": is_moving,
            "resolved_ra": float(coord.ra.deg),
            "resolved_dec": float(coord.dec.deg)
        }

        if len(results) == 0:

            return jsonify({
                "success": True,
                "count": 0,
                "shown": 0,
                "results": [],
                **target_info
            })

        previews = []
        preview_errors = []

        selected_rows = choose_preview_rows(results)

        print(
            f"Preparing {len(selected_rows)} "
            "real SPHEREx image previews..."
        )

        for row_index in selected_rows:

            try:

                preview = make_preview(
                    results,
                    row_index,
                    coord,
                    cutout
                )

                previews.append(preview)

                print(
                    f"Preview ready: {preview['band']} "
                    f"(row {row_index})"
                )

            except Exception as error:

                traceback.print_exc()

                preview_errors.append({
                    "index": row_index,
                    "error": str(error).splitlines()[0][:220]
                })

        return jsonify({
            "success": True,
            "count": len(results),
            "shown": len(previews),
            "results": previews,
            "preview_errors": preview_errors,
            **target_info
        })

    except TargetResolutionError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except (ValueError, TypeError) as error:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception as error:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# =========================================================
# SERVE PNG IMAGES TO THE WEBSITE
# =========================================================

@app.route("/api/preview/<path:filename>")
def api_preview(filename):

    if (
        os.path.basename(filename) != filename
        or not filename.endswith(".png")
    ):

        return jsonify({
            "success": False,
            "error": "Nombre de imagen no válido."
        }), 400

    return send_from_directory(
        PREVIEW_DIR,
        filename,
        mimetype="image/png",
        max_age=86400
    )


# =========================================================
# BLINK COMPARATOR
# =========================================================

@app.route("/api/blink")
def api_blink():

    try:

        # Resolve the same name used by image search.
        coord, object_name, resolver, is_moving = resolve_target()

        try:

            detector = int(
                request.args.get("detector", "3")
            )

            frames = int(
                request.args.get("frames", "6")
            )

        except (TypeError, ValueError, OverflowError) as error:

            raise ValueError(
                "Detector y frames deben ser números enteros."
            ) from error

        cutout = query_float(
            "cutout",
            0.25,
            0.01,
            1.0
        )

        if not 1 <= detector <= 6:

            raise ValueError(
                "El detector debe estar entre 1 y 6."
            )

        if not 2 <= frames <= 12:

            raise ValueError(
                "La cantidad de frames debe estar entre 2 y 12."
            )

        print(
            f"\nBlink target: {object_name or 'RA/Dec'} | "
            f"RA={coord.ra.deg:.8f}°, "
            f"Dec={coord.dec.deg:.8f}°"
        )

        print("Resolver:", resolver)

        results = search_images(
            coord,
            radius_arcsec=25
        )

        if len(results) == 0:

            return jsonify({
                "success": False,
                "error": (
                    "No se encontraron imágenes de SPHEREx "
                    "para ese objeto o posición."
                )
            }), 404

        # Parameters required by the original Blink function.
        class Args:
            pass

        args = Args()

        args.cutout = cutout
        args.blink_detector = detector
        args.blink_frames = frames
        args.blink_interval_ms = 700
        args.no_zodi = False
        args.save_gif = True
        args.blink_mode = "same_detector"

        gif_path = os.path.join(
            DATA_DIR,
            "blink.gif"
        )

        # Remove the previous GIF so an old result is not reused.
        if os.path.exists(gif_path):
            os.remove(gif_path)

        # Uses the safe date-selection function installed above.
        try:
            run_blink(
                args,
                results,
                coord,
                []
            )
        finally:
            # Free the matplotlib figures so the server doesn't grow in memory
            # with every Blink request.
            plt.close("all")

        if not os.path.exists(gif_path):

            return jsonify({
                "success": False,
                "error": (
                    f"No se generó el GIF. El detector {detector} "
                    "podría no tener al menos dos observaciones "
                    "descargables para este objeto."
                )
            }), 500

        return send_file(
            gif_path,
            mimetype="image/gif",
            conditional=True
        )

    except TargetResolutionError as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except (ValueError, TypeError) as error:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception as error:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print("\n===========================================")
    print("  Infrapix Backend corriendo")
    print("  http://localhost:5000")
    print("===========================================\n")

    app.run(
        debug=True,
        port=5000,
        use_reloader=False
    )
