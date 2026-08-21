"""
Realisable time translation: the receiver-range cell of Table 3, and the
transmission-range saturation diagnostic of Section 7.

Standalone. Downloads what it needs and depends on nothing in the notebook,
so a reviewer can run this file alone to check the two numbers the paper's
structural claim rests on.

WHAT THIS COMPUTES
------------------
Section 5.1 separates two realisations of the same group element T_{0,tau}:

  benchmark realisation   shifts sendTime AND rcvTime. Outside the Section 2.1
                          threat model -- the adversary cannot write the receive
                          timestamp. Measured elsewhere: 9 of 95 vehicles, at
                          control, because the claimed position and the receiver's
                          own reference lag together and the distance between them
                          is largely preserved.

  realisable realisation  the vehicle reports the position it occupied tau seconds
                          ago under an honest current timestamp. Both timestamps
                          genuine, latency invariant in fact. Only the claim lags,
                          so the claimed position is displaced by roughly v*tau
                          from where the vehicle actually is.

This file constructs the second on benign trajectories and applies the
receiver-range check of Section 4.4 to it.

Reported in the paper:
  displacement of claimed position   median 39.1 m   (predicted v*tau ~ 40 m)
  receiver-range check               298 of 380 vehicles
  control, same trajectories         19 of 380 (5%, by construction)
  threshold                          322.3 m
  saturation band                    3.8% of benign per-transmission maxima
                                     within 2.8 m below the threshold

CALIBRATION NOTE
----------------
The construction is synthetic on benign trajectories rather than drawn from the
benchmark, so it is calibrated separately and is NOT a like-for-like comparison
with the attacker-vehicle counts beside it in Table 3 (stated in Section 7).
Two controls make it auditable:

  - benign vehicles are truncated to the median attacker trace length (35
    transmissions), so a long benign trace does not get extra chances to cross
    the threshold;
  - the threshold is set at the benign 95th percentile of per-vehicle maximum
    sender-receiver distance, giving a 5% benign vehicle rate by construction.

The script also prints the benign 99th percentile of per-TRANSMISSION maximum
sender-receiver distance. It should land near the 321.2 m the paper reports from
the benchmark subsets; agreement is the evidence that this pipeline and the
notebook's are measuring the same quantity.

Usage:  python realisable_time_translation.py
"""

import gc
import io
import json
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SCENARIO = "InTAS_highway_2"
BASE = "https://zenodo.org/records/19665762/files"
TAU = 2.80          # recovered lag, Section 5.1
TRUNCATE_N = 35     # median attacker trace length, timeDelayAttack
N_BOOT = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def px(v, i):
    """Field i of a 'x,y,z' string; NaN if absent or malformed."""
    try:
        return float(str(v).split(",")[i])
    except Exception:
        return np.nan


def ensure(attack):
    os.makedirs("data", exist_ok=True)
    p = f"data/{SCENARIO}_{attack}.zip"
    if not (os.path.exists(p) and zipfile.is_zipfile(p)):
        if os.path.exists(p):
            os.remove(p)
        urllib.request.urlretrieve(f"{BASE}/{SCENARIO}_{attack}.zip?download=1", p)
    assert zipfile.is_zipfile(p), f"download failed: {attack}"
    return p


def read_pairs(attack):
    """One row per reception, carrying the RECEIVER's own de-noised true position.

    Each log file is written from one receiving vehicle's point of view, so the
    file named <rid> gives vehicle rid's own position at every rcvTime it logged.
    Deduplicating on rcvTime within a file therefore recovers that vehicle's true
    trajectory -- the same recovery Section 5.1 uses to obtain the attack
    magnitudes. pos_noise is subtracted to remove the recorded sensor-error term.
    """
    outer = zipfile.ZipFile(ensure(attack))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(
        [n for n in outer.namelist() if "/Test/" in n and n.endswith(".zip")][0])))
    rows = []
    for name in inner.namelist():
        if not name.endswith(".json"):
            continue
        rid = name.replace("\\", "/").split("/")[-1].replace(".json", "")
        for m in json.loads(inner.read(name)):
            st = m.get("sendTime")
            if st is None:
                continue
            r = m.get("receiver", {})
            rows.append((rid, m["sender_id"], float(st) / 1e9, m["rcvTime"] / 1e9,
                         int(m.get("attacker", 0)),
                         px(r.get("pos"), 0) - px(r.get("pos_noise"), 0),
                         px(r.get("pos"), 1) - px(r.get("pos_noise"), 1)))
    del outer, inner
    gc.collect()
    return pd.DataFrame(
        rows, columns=["rid", "sid", "s_t", "r_t", "from_atk", "tx", "ty"])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build(P):
    """Apply the realisable time translation to every benign reception.

    Returns the benign receptions with four added columns: the falsified claim
    (cx, cy) and the genuine position at the same instant (sx, sy).
    """
    own = {rid: g.drop_duplicates("r_t").sort_values("r_t")[["r_t", "tx", "ty"]].values
           for rid, g in P.groupby("rid")}

    B = P[(P.from_atk == 0) & P.sid.isin(own)].copy()
    cx = np.full(len(B), np.nan)
    cy = np.full(len(B), np.nan)
    sx = np.full(len(B), np.nan)
    sy = np.full(len(B), np.nan)

    for sid, idx in B.groupby("sid").groups.items():
        t_, x_, y_ = own[sid][:, 0], own[sid][:, 1], own[sid][:, 2]
        if len(t_) < 2:
            continue
        pos = B.index.get_indexer(idx)
        t = B.loc[idx, "s_t"].values
        tb = t - TAU
        # Messages whose lagged time precedes the vehicle's own logged track are
        # dropped, not clamped: clamping would manufacture a stationary period at
        # the start of every trajectory and depress the displacement.
        ok = (tb >= t_[0]) & (tb <= t_[-1]) & (t >= t_[0]) & (t <= t_[-1])
        cx[pos[ok]] = np.interp(tb[ok], t_, x_)
        cy[pos[ok]] = np.interp(tb[ok], t_, y_)
        sx[pos[ok]] = np.interp(t[ok], t_, x_)
        sy[pos[ok]] = np.interp(t[ok], t_, y_)

    B = B.assign(cx=cx, cy=cy, sx=sx, sy=sy).dropna(subset=["cx", "sx"])
    B["d_atk"] = np.hypot(B.cx - B.tx, B.cy - B.ty)   # falsified claim to receiver
    B["d_ctl"] = np.hypot(B.sx - B.tx, B.sy - B.ty)   # genuine position to receiver
    B["disp"] = np.hypot(B.cx - B.sx, B.cy - B.sy)    # displacement, ~ v*tau
    return B


def truncate(series, n):
    """Keep each vehicle's first n transmissions, matching attacker trace length."""
    return series.groupby(level=0, group_keys=False).apply(lambda g: g.head(n))


def boot(x, n=N_BOOT, seed=SEED):
    r = np.random.default_rng(seed)
    return np.percentile([x[r.integers(0, len(x), len(x))].mean()
                          for _ in range(n)], [2.5, 97.5])


# ---------------------------------------------------------------------------

def main():
    print("reading constantPositionOffset (benign rows only: this subset")
    print("falsifies position, not time, so its benign timestamps are genuine)\n")
    P = read_pairs("constantPositionOffset")
    B = build(P)
    print(f"benign receptions {len(B):,}   senders {B.sid.nunique()}   "
          f"receivers {B.rid.nunique()}\n")

    # Section 4.4 thresholds the per-TRANSMISSION maximum over receivers.
    tx_atk = B.groupby(["sid", "s_t"]).d_atk.max()
    tx_ctl = B.groupby(["sid", "s_t"]).d_ctl.max()

    print("--- calibration against the paper's benchmark-derived figures ---")
    print(f"benign per-transmission max: median {tx_ctl.median():7.1f} m   "
          f"(paper 217.5 m)")
    print(f"benign 99th percentile     : {np.percentile(tx_ctl, 99):7.1f} m   "
          f"(paper 321.2 m)\n")

    print("--- displacement of the claimed position ---")
    print(f"median {B.disp.median():.1f} m   "
          f"IQR {B.disp.quantile(.25):.1f}-{B.disp.quantile(.75):.1f} m")
    print(f"predicted v*tau at the benign median speed 14.5 m/s: "
          f"{14.5 * TAU:.0f} m\n")

    # Length-match, then calibrate the threshold to a 5% benign vehicle rate.
    ta = truncate(tx_atk.reset_index().set_index("sid").d_atk, TRUNCATE_N)
    tc = truncate(tx_ctl.reset_index().set_index("sid").d_ctl, TRUNCATE_N)
    va, vc = ta.groupby("sid").max(), tc.groupby("sid").max()
    T = np.percentile(vc, 95)

    flagged = (va > T).astype(float).values
    lo, hi = boot(flagged)

    print(f"--- receiver-range check, length-matched to {TRUNCATE_N} transmissions ---")
    print(f"threshold (benign 95th pct of per-vehicle max): {T:.1f} m")
    print(f"  realisable time translation : {int((va > T).sum())}/{len(va)} = "
          f"{(va > T).mean():.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  control, untransformed      : {int((vc > T).sum())}/{len(vc)} = "
          f"{(vc > T).mean():.3f}\n")

    # Section 7: the mechanism is partly physical impossibility, not
    # implausibility, because the simulated transmission range is a hard edge.
    band = ((tx_ctl > T - 2.8) & (tx_ctl <= T)).mean()
    print("--- saturation diagnostic (Section 7 caveat) ---")
    print("benign per-transmission max, upper tail:")
    for q in (90, 95, 97, 98, 99, 99.5):
        print(f"  p{q:<5} {np.percentile(tx_ctl, q):7.1f} m")
    print(f"fraction within 2.8 m below the threshold: {band:.4f}")
    print("  -> a hard edge at the simulated transmission range. Part of what the")
    print("     check detects is a claim placed beyond the distance at which the")
    print("     message was demonstrably received: physical impossibility rather")
    print("     than implausibility. Simulation has a sharp cutoff and deployed")
    print("     radios do not, so this figure and the 58 of 95 in Figure 1 are")
    print("     both upper bounds.")

    os.makedirs("out", exist_ok=True)
    pd.DataFrame({
        "quantity": ["displacement_median_m", "threshold_m", "attack_flagged",
                     "attack_total", "control_flagged", "control_total",
                     "ci_low", "ci_high", "saturation_band_frac"],
        "value": [B.disp.median(), T, int((va > T).sum()), len(va),
                  int((vc > T).sum()), len(vc), lo, hi, band],
    }).to_csv("out/realisable_time_translation.csv", index=False)
    print("\nwrote out/realisable_time_translation.csv")


if __name__ == "__main__":
    main()
