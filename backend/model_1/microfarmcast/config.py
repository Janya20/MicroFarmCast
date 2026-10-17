from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_dir"] = str(path.parent)
    for key in ("input_paths",):
        config["data"][key] = [
            str((path.parent / p).resolve()) if not Path(p).is_absolute() else p
            for p in config["data"].get(key, [])
        ]
    output = Path(config["data"]["output_dir"])
    config["data"]["output_dir"] = str(
        output if output.is_absolute() else (path.parent / output).resolve()
    )
    return config
