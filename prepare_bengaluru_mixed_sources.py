"""Align POWER/MERRA-2 weather and ERA5-Land layer-1 soil water by UTC hour."""
import argparse
from pathlib import Path
import pandas as pd


def merge_sources(power_path, era5_path, output_path):
    power = pd.read_csv(power_path)
    soil = pd.read_csv(era5_path)
    power["timestamp"] = pd.to_datetime(power["timestamp"], errors="coerce", utc=True)
    soil["timestamp"] = pd.to_datetime(soil["valid_time"], errors="coerce", utc=True)
    if power["timestamp"].isna().any() or soil["timestamp"].isna().any():
        raise ValueError("Invalid timestamp found; fix timestamps before merging")
    if power["timestamp"].duplicated().any() or soil["timestamp"].duplicated().any():
        raise ValueError("Duplicate hourly timestamps found; resolve duplicates first")
    if soil["swvl1"].isna().any():
        raise ValueError("ERA5-Land swvl1 has missing values; inspect before training")
    # The requested ERA5 point is expected to be the 0.1-degree grid point nearest the POWER point.
    power_lat, power_lon = float(power["latitude"].iloc[0]), float(power["longitude"].iloc[0])
    soil_lat, soil_lon = float(soil["latitude"].iloc[0]), float(soil["longitude"].iloc[0])
    soil["soil_moisture_m3_m3"] = pd.to_numeric(soil["swvl1"], errors="coerce")
    join = power.merge(soil[["timestamp", "soil_moisture_m3_m3"]], on="timestamp", how="outer", indicator=True, validate="one_to_one")
    unmatched = join["_merge"] != "both"
    if unmatched.any():
        sides = join.loc[unmatched, "_merge"].value_counts().to_dict()
        raise ValueError(f"Hourly timestamp coverage differs between inputs: {sides}")
    join = join.drop(columns="_merge").sort_values("timestamp")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    join.to_csv(output_path, index=False)
    print(f"Merged {len(join):,} exact UTC hourly matches -> {output_path}")
    print(f"POWER/MERRA-2 requested point: {power_lat:.4f}, {power_lon:.4f}")
    print(f"ERA5-Land selected grid point: {soil_lat:.4f}, {soil_lon:.4f}")
    print("Sources are close but use different grids; this is a mixed reanalysis proxy, not farm sensor truth.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("power_csv")
    parser.add_argument("era5_soil_csv")
    parser.add_argument("--output", default="data/bengaluru_power_era5_2024.csv")
    args = parser.parse_args()
    merge_sources(args.power_csv, args.era5_soil_csv, args.output)


if __name__ == "__main__":
    main()
