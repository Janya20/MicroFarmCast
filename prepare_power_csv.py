"""Convert a NASA POWER hourly point CSV to the MicroFarmCast canonical schema."""
import argparse
from io import StringIO
from pathlib import Path
import re
import pandas as pd


def convert(source, destination):
    lines = Path(source).read_text(encoding="utf-8-sig").splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip() == "-END HEADER-"), None)
    if end is None or end + 1 >= len(lines):
        raise ValueError("Could not find POWER -END HEADER- marker and CSV header")
    metadata = "\n".join(lines[:end + 1])
    location = re.search(r"Location:\s*Latitude\s+([-\d.]+)\s+Longitude\s+([-\d.]+)", metadata)
    elevation = re.search(r"Elevation from MERRA-2:.*=\s*([-\d.]+)\s*meters", metadata)
    if not location:
        raise ValueError("Could not read latitude/longitude from POWER file header")
    lat, lon = map(float, location.groups())
    elev = float(elevation.group(1)) if elevation else float("nan")
    raw = pd.read_csv(StringIO("\n".join(lines[end + 1:])), na_values=[-999, "-999"])
    required = {"YEAR", "MO", "DY", "HR", "T2M", "RH2M", "WS2M", "PRECTOTCORR"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"POWER file is missing expected columns: {missing}")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime({"year": raw.YEAR, "month": raw.MO, "day": raw.DY, "hour": raw.HR}, utc=True),
        "location_id": f"power_{lat:.4f}_{lon:.4f}",
        "latitude": lat,
        "longitude": lon,
        "elevation_m": elev,
        "temperature_c": raw.T2M,
        "relative_humidity_pct": raw.RH2M,
        "rainfall_mm": raw.PRECTOTCORR,
        "wind_speed_m_s": raw.WS2M,
    })
    if "T2MDEW" in raw.columns:
        out["dewpoint_c"] = raw.T2MDEW
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    print(f"Wrote {len(out):,} hourly rows to {destination}")
    print("Source elevation is a regional MERRA-2 grid average, not surveyed farm elevation.")
    print("No soil-moisture label is present in this POWER export.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Original NASA POWER hourly point CSV")
    parser.add_argument("destination", nargs="?", default="data/power_bengaluru_2024.csv")
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
