import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from datetime import datetime

DATA_DIR = "pulse-transit-time-ppg/1.1.0/csv"
RESULTS_DIR = "results/mapi_validation"
FS = 500.0  # sampling freq for PPG & IMU (Hz)

##############################
# 1. Signal / beat utilities #
##############################

def preprocess_ppg(ppg_raw, fs=FS):
    """Remove DC and bandpass 0.75–5 Hz (as suggested in dataset notes)."""
    ppg = ppg_raw.astype(float)
    ppg = ppg - np.mean(ppg)
    # 2nd-order Butterworth bandpass
    b, a = signal.butter(2, [0.75, 5.0], btype="bandpass", fs=fs)
    ppg_f = signal.filtfilt(b, a, ppg)
    return ppg_f

def detect_ppg_peaks(ppg_f, fs=FS):
    """Detect systolic peaks, automatically handling signal polarity."""
    std = np.std(ppg_f)
    min_distance = int(0.3 * fs)  # at most ~200 bpm
    # Try positive polarity
    peaks_pos, _ = signal.find_peaks(
        ppg_f,
        distance=min_distance,
        prominence=0.5 * std
    )
    # Try negative polarity
    peaks_neg, _ = signal.find_peaks(
        -ppg_f,
        distance=min_distance,
        prominence=0.5 * std
    )
    if len(peaks_neg) > len(peaks_pos):
        # Invert if negative gives more reasonable beats
        ppg_f = -ppg_f
        peaks = peaks_neg
    else:
        peaks = peaks_pos
    return ppg_f, peaks

def extract_beats(df, fs=FS, ppg_col="pleth_2"):
    """
    From one CSV record, compute beat-wise features:
      amplitude, rise time, period, HR, temperature, accel std, peak times.
    """
    ppg_raw = df[ppg_col].values
    temp = df["temp_1"].values
    ax = df["a_x"].values
    ay = df["a_y"].values
    az = df["a_z"].values

    ppg_f = preprocess_ppg(ppg_raw, fs)
    ppg_f, peaks = detect_ppg_peaks(ppg_f, fs)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    amp_list = []
    t_rise_list = []
    period_list = []
    hr_list = []
    temp_list = []
    std_acc_list = []
    t_peak_list = []

    for i in range(1, len(peaks)):
        peak = peaks[i]
        prev_peak = peaks[i - 1]
        # segment from previous peak to current peak
        if peak <= prev_peak + 2:
            continue
        seg = ppg_f[prev_peak:peak + 1]
        foot_rel = np.argmin(seg)
        foot = prev_peak + foot_rel
        amp = ppg_f[peak] - ppg_f[foot]
        if amp <= 0:
            continue
        t_peak = peak / fs
        t_prev_peak = prev_peak / fs
        period = t_peak - t_prev_peak
        if period <= 0:
            continue
        t_rise = (peak - foot) / fs

        # Mean temp & accel std during this beat
        beat_slice = slice(prev_peak, peak + 1)
        temp_mean = float(np.mean(temp[beat_slice]))
        std_acc = float(np.std(acc_mag[beat_slice]))

        amp_list.append(amp)
        t_rise_list.append(t_rise)
        period_list.append(period)
        hr_list.append(60.0 / period)
        temp_list.append(temp_mean)
        std_acc_list.append(std_acc)
        t_peak_list.append(t_peak)

    beats = {
        "amp": np.array(amp_list),
        "t_rise": np.array(t_rise_list),
        "period": np.array(period_list),
        "hr": np.array(hr_list),
        "temp": np.array(temp_list),
        "std_acc": np.array(std_acc_list),
        "t_peak": np.array(t_peak_list),
    }
    return beats

##############################
# 2. MAPI computation        #
##############################

def compute_baseline(beats):
    """Baseline from (typically) the sit record: use medians."""
    baseline = {
        "amp0": np.median(beats["amp"]),
        "t_rise0": np.median(beats["t_rise"]),
        "hr0": np.median(beats["hr"]),
        "temp0": np.median(beats["temp"]),
        "std_acc0": np.median(beats["std_acc"]),
    }
    return baseline

def compute_mapi(beats, baseline,
                 alpha=0.0774, beta=0.0, gamma=0.0,
                 sigma_ref=0.05, temp_ref=1.0):
    """Your species-agnostic MAPI formula."""
    amp = beats["amp"]
    t_rise = beats["t_rise"]
    hr = beats["hr"]
    temp = beats["temp"]
    std_acc = beats["std_acc"]

    # Relative terms (clip to avoid crazy outliers)
    PI_rel = np.clip(amp / (baseline["amp0"] + 1e-9), 0, 3)
    RT_rel = np.clip(baseline["t_rise0"] / (t_rise + 1e-9), 0, 3)
    HR_rel = np.clip(hr / (baseline["hr0"] + 1e-9), 0, 3)

    MAPI_core = 1.0 \
        + alpha * (PI_rel - 1.0) \
        + beta * (RT_rel - 1.0) \
        + gamma * (HR_rel - 1.0)

    # Weights
    delta_temp = np.abs(temp - baseline["temp0"])
    w_motion = 1.0 / (1.0 + (std_acc / (sigma_ref + 1e-9))**2)
    w_temp = 1.0 / (1.0 + (delta_temp / (temp_ref + 1e-9))**2)

    MAPI = MAPI_core * w_motion * w_temp
    return MAPI

def start_end_mapi(mapi, t_peak, window_sec=20.0):
    """Mean MAPI in early & late windows of a record."""
    if len(mapi) == 0:
        return np.nan, np.nan
    t0 = t_peak[0]
    t_end = t_peak[-1]
    dur = t_end - t0
    w = min(window_sec, dur / 3.0)
    start_mask = (t_peak - t0 <= w)
    end_mask = (t_end - t_peak <= w)
    if not np.any(start_mask) or not np.any(end_mask):
        return np.nan, np.nan
    return float(np.mean(mapi[start_mask])), float(np.mean(mapi[end_mask]))

##############################
# 3. Main analysis           #
##############################

def main():
    info_path = os.path.join(DATA_DIR, "subjects_info.csv")
    info = pd.read_csv(info_path)

    # Build map: subject -> {activity: record_name}
    subjects = {}
    for _, row in info.iterrows():
        rec = row["record"]        # e.g. "s1_walk"
        activity = row["activity"] # sit/walk/run
        subj = rec.split("_")[0]   # e.g. "s1"
        subjects.setdefault(subj, {})[activity] = rec

    # 1) Compute baselines from sit recordings
    baselines = {}
    for subj, recs in subjects.items():
        sit_rec = recs.get("sit", None)
        if sit_rec is None:
            print(f"No sit record for {subj}, skipping baseline.")
            continue
        csv_path = os.path.join(DATA_DIR, f"{sit_rec}.csv")
        if not os.path.exists(csv_path):
            print(f"File not found: {sit_rec}.csv, skipping baseline.")
            continue
        df = pd.read_csv(csv_path)
        beats = extract_beats(df)
        if len(beats["amp"]) < 20:
            print(f"Too few beats in {sit_rec} for baseline.")
            continue
        baselines[subj] = compute_baseline(beats)

    # 2) Loop over all records and gather MAPI/MAP pairs
    mapi_vals = []
    map_vals = []
    delta_mapi_vals = []
    delta_map_vals = []

    for _, row in info.iterrows():
        rec = row["record"]
        activity = row["activity"]
        subj = rec.split("_")[0]

        csv_path = os.path.join(DATA_DIR, f"{rec}.csv")
        if not os.path.exists(csv_path):
            continue

        if subj not in baselines:
            # fall back: compute baseline from this record itself
            df = pd.read_csv(csv_path)
            beats_bl = extract_beats(df)
            if len(beats_bl["amp"]) < 20:
                continue
            baselines[subj] = compute_baseline(beats_bl)

        baseline = baselines[subj]
        df = pd.read_csv(csv_path)
        beats = extract_beats(df)
        if len(beats["amp"]) < 20:
            continue

        mapi = compute_mapi(beats, baseline)
        mapi_start, mapi_end = start_end_mapi(mapi, beats["t_peak"])
        if np.isnan(mapi_start) or np.isnan(mapi_end):
            continue

        # Cuff MAP at start / end
        map_start = (row["bp_sys_start"] + 2 * row["bp_dia_start"]) / 3.0
        map_end   = (row["bp_sys_end"]   + 2 * row["bp_dia_end"])   / 3.0

        # Absolute points
        mapi_vals.extend([mapi_start, mapi_end])
        map_vals.extend([map_start, map_end])

        # Deltas per record
        delta_mapi_vals.append(mapi_end - mapi_start)
        delta_map_vals.append(map_end - map_start)

    mapi_vals = np.array(mapi_vals)
    map_vals = np.array(map_vals)
    delta_mapi_vals = np.array(delta_mapi_vals)
    delta_map_vals = np.array(delta_map_vals)

    # 3) Correlations & significance
    print("\n" + "="*60)
    print("MAPI VALIDATION RESULTS")
    print("="*60)
    print(f"Number of subjects with baselines: {len(baselines)}")
    print(f"Total data points: {len(mapi_vals)}")
    print(f"Total records processed: {len(delta_mapi_vals)}")
    
    print("\nAbsolute MAPI vs MAP:")
    r_abs, p_abs = pearsonr(mapi_vals, map_vals)
    rs_abs, ps_abs = spearmanr(mapi_vals, map_vals)
    print(f"  Pearson r = {r_abs:.3f}, p = {p_abs:.3e}")
    print(f"  Spearman rho = {rs_abs:.3f}, p = {ps_abs:.3e}")

    print("\nDelta MAPI vs Delta MAP:")
    r_d, p_d = pearsonr(delta_mapi_vals, delta_map_vals)
    rs_d, ps_d = spearmanr(delta_mapi_vals, delta_map_vals)
    print(f"  Pearson r = {r_d:.3f}, p = {p_d:.3e}")
    print(f"  Spearman rho = {rs_d:.3f}, p = {ps_d:.3e}")
    print("="*60)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed data
    results_df = pd.DataFrame({
        'mapi': mapi_vals,
        'map': map_vals
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, f"absolute_values_{timestamp}.csv"), index=False)
    
    delta_df = pd.DataFrame({
        'delta_mapi': delta_mapi_vals,
        'delta_map': delta_map_vals
    })
    delta_df.to_csv(os.path.join(RESULTS_DIR, f"delta_values_{timestamp}.csv"), index=False)
    
    # Save summary report
    report_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write("MAPI VALIDATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of subjects: {len(baselines)}\n")
        f.write(f"Total data points: {len(mapi_vals)}\n")
        f.write(f"Total records: {len(delta_mapi_vals)}\n\n")
        
        f.write("Absolute MAPI vs MAP:\n")
        f.write(f"  Pearson r = {r_abs:.3f}, p = {p_abs:.3e}\n")
        f.write(f"  Spearman rho = {rs_abs:.3f}, p = {ps_abs:.3e}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if p_abs < 0.05 else 'No'}\n\n")
        
        f.write("Delta MAPI vs Delta MAP:\n")
        f.write(f"  Pearson r = {r_d:.3f}, p = {p_d:.3e}\n")
        f.write(f"  Spearman rho = {rs_d:.3f}, p = {ps_d:.3e}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if p_d < 0.05 else 'No'}\n\n")
        
        f.write("CONCLUSIONS:\n")
        f.write("-" * 60 + "\n")
        if p_abs < 0.05:
            f.write(f"- Absolute values show significant correlation (r={r_abs:.3f})\n")
        else:
            f.write(f"- Absolute values show no significant correlation (p={p_abs:.3e})\n")
        
        if p_d < 0.05:
            f.write(f"- Delta values show significant correlation (r={r_d:.3f})\n")
        else:
            f.write(f"- Delta values show no significant correlation (p={p_d:.3e})\n")
        
        if len(baselines) < 10:
            f.write(f"\n⚠ WARNING: Small sample size (n={len(baselines)}). Results may not be reliable.\n")
            f.write("  Recommend collecting data from at least 20 subjects.\n")
    
    print(f"\n✅ Results saved to {RESULTS_DIR}/")
    print(f"   - Summary: summary_{timestamp}.txt")
    print(f"   - Absolute values: absolute_values_{timestamp}.csv")
    print(f"   - Delta values: delta_values_{timestamp}.csv")

if __name__ == "__main__":
    main()

