"""Generate labeled, non-observational hourly data for pipeline smoke runs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(output: str | Path = "data/demo_microclimate.csv", days: int = 45, seed: int = 17) -> Path:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-06-01T00:00:00Z")
    times = pd.date_range(start, periods=days*24, freq="h")
    sites = [("DEMO_A",18.52,73.85,560),("DEMO_B",18.61,73.91,590),("DEMO_C",18.44,73.78,520),
             ("DEMO_D",18.70,73.68,640),("DEMO_E",18.35,73.94,480),("DEMO_F",18.55,74.04,610)]
    rows=[]
    for j,(loc,lat,lon,elev) in enumerate(sites):
        temp_prev=24.0-j*.25; humidity_prev=70+j*.4; soil=.30-j*.006; wind_prev=2.0
        for t in times:
            hour=t.hour
            temp=24 + 4*np.sin(2*np.pi*(hour-8)/24) + 1.1*np.sin(2*np.pi*t.dayofyear/365) - (elev-550)*.006 + rng.normal(0,.45)
            humidity=np.clip(72 - .75*(temp-24) + 4*np.sin(2*np.pi*(hour+2)/24) + rng.normal(0,2),25,99)
            wet=bool(rng.random() < (.12 + .09*(humidity>82)))
            rain=float(rng.gamma(1.5,3.0) if wet else 0.)
            wind=max(.1, .65*wind_prev + .7 + .25*wet + rng.normal(0,.35))
            soil=float(np.clip(.985*soil + rain*.012 - max(temp-25,0)*.0005 + rng.normal(0,.0015),.08,.65))
            # Retain a small physical temporal response so the synthetic demo
            # has realistic persistence, while remaining explicitly invented.
            temp=.55*temp+.45*temp_prev
            humidity=.55*humidity+.45*humidity_prev
            rows.append((t,loc,lat,lon,elev,temp,humidity,rain,wind,soil,True))
            temp_prev,humidity_prev,wind_prev=temp,humidity,wind
    frame=pd.DataFrame(rows,columns=["timestamp","location_id","latitude","longitude","elevation_m","temperature_c","relative_humidity_pct","rainfall_mm","wind_speed_mps","soil_moisture","synthetic_demo"])
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,float_format="%.5f")
    return path


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="data/demo_microclimate.csv"); parser.add_argument("--days",type=int,default=45); parser.add_argument("--seed",type=int,default=17)
    args=parser.parse_args(); print(generate(args.output,args.days,args.seed))
