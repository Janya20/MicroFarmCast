import argparse
import json
from microfarmcast.predict import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict one hour ahead with saved MicroFarmCast models")
    parser.add_argument("--input", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(predict(args.input, args.location, args.timestamp, args.artifacts), indent=2))
