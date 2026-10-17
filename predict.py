"""Run a saved MicroFarmCast model on recent CSV/Parquet history."""
import argparse
from pathlib import Path

import pandas as pd

from microfarmcast.predict import predict_from_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Recent history CSV or Parquet")
    parser.add_argument("--artifacts", default="artifacts_bengaluru_5target")
    parser.add_argument("--output", default="prediction.csv")
    parser.add_argument("--horizon-hours", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.input)
    history = pd.read_parquet(source) if source.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(source)
    result = predict_from_history(history, args.artifacts, args.horizon_hours)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved prediction to {args.output}")


if __name__ == "__main__":
    main()
