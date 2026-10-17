import argparse
import json
from microfarmcast.train import run

parser = argparse.ArgumentParser(description="Train MicroFarmCast Model 1.")
parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
args = parser.parse_args()
print(json.dumps(run(args.config), indent=2, default=str))
