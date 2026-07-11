"""openSSS scatter estimation runner (subprocess entry for danvas_compare.py).

Usage: python opensss_runner.py <config.pkl>
  config: activity_npy, mumap_npy, lors_npy, out_npy,
          scale ("manual" | "tailfit"), scatter_fraction (float)

Estimates the scatter table from the given activity/mu-map, maps it onto the
continuous LORs, scales it, and saves the per-event vector to out_npy.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import opensss_bridge as bridge

cfg = pickle.load(open(sys.argv[1], "rb"))
activity = np.load(cfg["activity_npy"]).astype(np.float32)
mumap = np.load(cfg["mumap_npy"]).astype(np.float32)
lors = np.load(cfg["lors_npy"]).astype(np.float32)

print(f"openSSS: estimating scatter (activity {activity.shape[0]}^3)", flush=True)
full = bridge.estimate_scatter(activity, mumap)
s = bridge.map_to_lors(full, lors)

if cfg.get("scale") == "tailfit":
    scale, tail_frac = bridge.tail_fit_scale(full, lors, activity)
    if scale is not None:
        s = s * scale
        print(f"openSSS: tail fit ok — tail events {tail_frac:.1%}, "
              f"scatter fraction {s.sum() / len(lors):.1%}", flush=True)
    else:
        s = bridge.scale_to_fraction(s, float(cfg["scatter_fraction"]))
        print(f"openSSS: too few tails ({tail_frac:.1%}); fell back to manual "
              f"fraction {cfg['scatter_fraction']}", flush=True)
else:
    s = bridge.scale_to_fraction(s, float(cfg["scatter_fraction"]))
    print(f"openSSS: manual scatter fraction {cfg['scatter_fraction']}", flush=True)

np.save(cfg["out_npy"], s.astype(np.float32))
print("openSSS: done", flush=True)
