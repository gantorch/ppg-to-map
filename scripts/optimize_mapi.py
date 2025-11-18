import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from datetime import datetime

# Import functions from mapi_validation
sys.path.insert(0, os.path.dirname(__file__))
from mapi_validation import (
    preprocess_ppg, detect_ppg_peaks, extract_beats, 
    compute_baseline, start_end_mapi, DATA_DIR, FS
)

RESULTS_DIR = "results/mapi_optimization"

def compute_mapi_with_params(beats, baseline, alpha, beta, gamma, 
                              sigma_ref=0.05, temp_ref=1.0):
    """Compute MAPI with custom parameters."""
    amp = beats["amp"]
    t_rise = beats["t_rise"]
    hr = beats["hr"]
    temp = beats["temp"]
    std_acc = beats["std_acc"]

    # Relative terms
    PI_rel = np.clip(amp / (baseline["amp0"] + 1e-9), 0, 3)
    RT_rel = np.clip(baseline["t_rise0"] / (t_rise + 1e-9), 0, 3)
    HR_rel = np.clip(hr / (baseline["hr0"] + 1e-9), 0, 3)

    MAPI_core = 1.0 + alpha * (PI_rel - 1.0) + beta * (RT_rel - 1.0) + gamma * (HR_rel - 1.0)

    # Weights
    delta_temp = np.abs(temp - baseline["temp0"])
    w_motion = 1.0 / (1.0 + (std_acc / (sigma_ref + 1e-9))**2)
    w_temp = 1.0 / (1.0 + (delta_temp / (temp_ref + 1e-9))**2)

    MAPI = MAPI_core * w_motion * w_temp
    return MAPI

def evaluate_params(params, data_dict, optimize_for='absolute'):
    """
    Evaluate parameter set by computing correlation with MAP.
    Returns negative correlation (for minimization).
    """
    alpha, beta, gamma = params
    
    mapi_vals = []
    map_vals = []
    delta_mapi_vals = []
    delta_map_vals = []
    
    for rec_data in data_dict['records']:
        beats = rec_data['beats']
        baseline = rec_data['baseline']
        row = rec_data['row']
        
        mapi = compute_mapi_with_params(beats, baseline, alpha, beta, gamma)
        mapi_start, mapi_end = start_end_mapi(mapi, beats["t_peak"])
        
        if np.isnan(mapi_start) or np.isnan(mapi_end):
            continue
        
        map_start = (row["bp_sys_start"] + 2 * row["bp_dia_start"]) / 3.0
        map_end = (row["bp_sys_end"] + 2 * row["bp_dia_end"]) / 3.0
        
        mapi_vals.extend([mapi_start, mapi_end])
        map_vals.extend([map_start, map_end])
        delta_mapi_vals.append(mapi_end - mapi_start)
        delta_map_vals.append(map_end - map_start)
    
    if len(mapi_vals) < 4:
        return 1e6  # Penalty for insufficient data
    
    mapi_vals = np.array(mapi_vals)
    map_vals = np.array(map_vals)
    
    if optimize_for == 'absolute':
        r, p = pearsonr(mapi_vals, map_vals)
    else:  # delta
        delta_mapi_vals = np.array(delta_mapi_vals)
        delta_map_vals = np.array(delta_map_vals)
        r, p = pearsonr(delta_mapi_vals, delta_map_vals)
    
    # Return negative correlation (we want to maximize correlation)
    return -abs(r)

def load_data():
    """Load all data once for optimization."""
    info_path = os.path.join(DATA_DIR, "subjects_info.csv")
    info = pd.read_csv(info_path)
    
    # Build subjects map
    subjects = {}
    for _, row in info.iterrows():
        rec = row["record"]
        activity = row["activity"]
        subj = rec.split("_")[0]
        subjects.setdefault(subj, {})[activity] = rec
    
    # Compute baselines
    baselines = {}
    for subj, recs in subjects.items():
        sit_rec = recs.get("sit", None)
        if sit_rec is None:
            continue
        csv_path = os.path.join(DATA_DIR, f"{sit_rec}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        beats = extract_beats(df)
        if len(beats["amp"]) < 20:
            continue
        baselines[subj] = compute_baseline(beats)
    
    # Load all records
    records = []
    for _, row in info.iterrows():
        rec = row["record"]
        subj = rec.split("_")[0]
        
        csv_path = os.path.join(DATA_DIR, f"{rec}.csv")
        if not os.path.exists(csv_path):
            continue
        
        if subj not in baselines:
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
        
        records.append({
            'beats': beats,
            'baseline': baseline,
            'row': row,
            'record': rec
        })
    
    return {'records': records, 'baselines': baselines}

def main():
    print("="*60)
    print("MAPI PARAMETER OPTIMIZATION")
    print("="*60)
    
    print("\nLoading data...")
    data_dict = load_data()
    print(f"Loaded {len(data_dict['records'])} records from {len(data_dict['baselines'])} subjects")
    
    # Initial parameters
    initial_params = [0.6, 0.3, 0.1]  # alpha, beta, gamma
    
    print("\nOptimizing for absolute MAPI vs MAP correlation...")
    result_abs = minimize(
        evaluate_params,
        initial_params,
        args=(data_dict, 'absolute'),
        method='Nelder-Mead',
        bounds=[(0, 2), (0, 2), (0, 2)],
        options={'maxiter': 200, 'disp': True}
    )
    
    print("\nOptimizing for delta MAPI vs delta MAP correlation...")
    result_delta = minimize(
        evaluate_params,
        initial_params,
        args=(data_dict, 'delta'),
        method='Nelder-Mead',
        bounds=[(0, 2), (0, 2), (0, 2)],
        options={'maxiter': 200, 'disp': True}
    )
    
    # Results
    alpha_abs, beta_abs, gamma_abs = result_abs.x
    alpha_delta, beta_delta, gamma_delta = result_delta.x
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print("\nOptimized for Absolute Correlation:")
    print(f"  α = {alpha_abs:.4f}")
    print(f"  β = {beta_abs:.4f}")
    print(f"  γ = {gamma_abs:.4f}")
    print(f"  Best correlation: {-result_abs.fun:.4f}")
    
    print("\nOptimized for Delta Correlation:")
    print(f"  α = {alpha_delta:.4f}")
    print(f"  β = {beta_delta:.4f}")
    print(f"  γ = {gamma_delta:.4f}")
    print(f"  Best correlation: {-result_delta.fun:.4f}")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_path = os.path.join(RESULTS_DIR, f"optimization_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write("MAPI PARAMETER OPTIMIZATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of records: {len(data_dict['records'])}\n")
        f.write(f"Number of subjects: {len(data_dict['baselines'])}\n\n")
        
        f.write("Initial Parameters:\n")
        f.write(f"  α = {initial_params[0]:.4f}\n")
        f.write(f"  β = {initial_params[1]:.4f}\n")
        f.write(f"  γ = {initial_params[2]:.4f}\n\n")
        
        f.write("Optimized for Absolute Correlation:\n")
        f.write(f"  α = {alpha_abs:.4f}\n")
        f.write(f"  β = {beta_abs:.4f}\n")
        f.write(f"  γ = {gamma_abs:.4f}\n")
        f.write(f"  Best |r| = {-result_abs.fun:.4f}\n\n")
        
        f.write("Optimized for Delta Correlation:\n")
        f.write(f"  α = {alpha_delta:.4f}\n")
        f.write(f"  β = {beta_delta:.4f}\n")
        f.write(f"  γ = {gamma_delta:.4f}\n")
        f.write(f"  Best |r| = {-result_delta.fun:.4f}\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write("-" * 60 + "\n")
        if -result_abs.fun > -result_delta.fun:
            f.write("Use absolute-optimized parameters (stronger correlation)\n")
            f.write(f"Update mapi_validation.py with:\n")
            f.write(f"  alpha={alpha_abs:.4f}, beta={beta_abs:.4f}, gamma={gamma_abs:.4f}\n")
        else:
            f.write("Use delta-optimized parameters (stronger correlation)\n")
            f.write(f"Update mapi_validation.py with:\n")
            f.write(f"  alpha={alpha_delta:.4f}, beta={beta_delta:.4f}, gamma={gamma_delta:.4f}\n")
    
    print(f"\n✅ Results saved to {report_path}")
    print("\nTo use optimized parameters, update scripts/mapi_validation.py")
    print(f"and set: alpha={alpha_abs:.4f}, beta={beta_abs:.4f}, gamma={gamma_abs:.4f}")

if __name__ == "__main__":
    main()

