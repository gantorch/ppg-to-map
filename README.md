# PPG to MAP Prediction

mean arterial pressure (map) from ppg signals. 

## Structure

```
ppg-to-map/
├── data/
│   ├── raw/              # Original dataset (WFDB format)
│   ├── processed/        # Cleaned/transformed data
│   └── external/         # Third-party datasets
├── scripts/
│   ├── download_data.py    # Dataset download
│   └── mapi_validation.py  # MAPI validation analysis
├── src/                  # Source code
├── notebooks/            # Jupyter notebooks
├── ppgtest/              # Virtual environment
├── requirements.txt
└── README.md
```

## Setup

```bash
# Virtual environment
python3 -m venv ppgtest
source ppgtest/bin/activate  # macOS/Linux
# ppgtest\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

**Source:** [PhysioNet - Pulse Transit Time PPG](https://physionet.org/files/pulse-transit-time-ppg/1.1.0/)

- Format: WFDB (`.dat`, `.hea`, `.atr`)
- 22 subjects (s1-s22)
- 3 activities: run, sit, walk
- Full dataset: ~400 MB

### Download

```bash
python scripts/download_data.py
```

Default: 2 subjects, "sit" only (~13 MB)

To download more, edit `scripts/download_data.py`:

```python
SUBJECTS = [1, 2, 3, 4, 5]  # subjects 1-22
ACTIVITIES = ['sit', 'walk', 'run']
```

## Usage

### Loading WFDB Data

```python
import wfdb
import pandas as pd

# Load record
record = wfdb.rdrecord('data/raw/s1_sit')

# To DataFrame
df = pd.DataFrame(record.p_signal, columns=record.sig_name)
df['time'] = np.arange(len(df)) / record.fs
```

### MAPI Validation

Run the MAPI (Mean Arterial Pressure Index) validation analysis:

```bash
python scripts/mapi_validation.py
```

Requires CSV files in `pulse-transit-time-ppg/1.1.0/csv/` directory.

## MAPI Formula

Current formula for estimating Mean Arterial Pressure Index:

\[
MAPI_{core} = 1 + \alpha \cdot (PI_{rel} - 1) + \beta \cdot (RT_{rel} - 1) + \gamma \cdot (HR_{rel} - 1)
\]

\[
MAPI = MAPI_{core} \cdot w_{motion} \cdot w_{temp}
\]

Where:
- \( PI_{rel} = \frac{A}{A_0} \) (pulse amplitude relative to baseline)
- \( RT_{rel} = \frac{RT_0}{RT} \) (rise time relative to baseline)
- \( HR_{rel} = \frac{HR}{HR_0} \) (heart rate relative to baseline)
- \( w_{motion} = \frac{1}{1 + (\sigma_{acc} / \sigma_{ref})^2} \) (motion artifact weight)
- \( w_{temp} = \frac{1}{1 + (\Delta T / T_{ref})^2} \) (temperature change weight)

**Current Parameters (Optimized 2025-11-18):**
- \( \alpha = 0.0774 \) (pulse amplitude weight)
- \( \beta = 0.0 \) (rise time weight)
- \( \gamma = 0.0 \) (heart rate weight)
- \( \sigma_{ref} = 0.05 \) (reference motion std)
- \( T_{ref} = 1.0 \) (reference temperature change)

**Simplified Formula:**
\[
MAPI = [1 + 0.0774 \cdot (PI_{rel} - 1)] \cdot w_{motion} \cdot w_{temp}
\]

Note: Optimization found that only pulse amplitude contributes to MAP prediction in this dataset.

## Key Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `wfdb` - WFDB file reading

Full list in `requirements.txt`
