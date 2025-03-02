# Coherence-Monitor for KAGRA

This repository contains Python scripts for analyzing coherence between KAGRA’s K1 strain data and witness channels. The scripts identify significant witness channels for noise subtraction applications, such as DeepClean.

This work builds on the original codebase developed by [Siddharth Soni (MIT)](https://git.ligo.org/siddharth.soni/coherence-monitor) and [Christina Reissel (MIT)](https://git.ligo.org/christina.reissel/coherence-monitor) for LIGO GW data analysis. It extends their coherence analysis framework by customizing it for KAGRA (K1).
## Overview

The workflow consists of three main scripts:
1. **`utils.py`**: Utility functions for data retrieval, coherence calculation, and plotting, tailored for K1.
2. **`run_coherence.py`**: Computes coherence between strain and witness channels, saving results and generating a summary plot with parallel processing optimizations.
3. **`select_witnesses.py`**: Analyzes coherence data to select high-coherence witness channels in a specified frequency band, producing a DeepClean-compatible channel list.

These scripts process KAGRA data in GPS time segments, using `gwpy` for GW analysis and `plotly` for visualization.

## Prerequisites

### Software Requirements
- **Conda Environment**: Use the `igwn-py38` environment for gravitational wave tools:
  ```bash
  conda env create -n igwn-py38 -f https://git.ligo.org/lscsoft/igwn-environments/-/raw/main/environments/igwn-py38.yaml
  conda activate igwn-py38
  ```
  
### Environment Variables
- **`KAGRA_DATA_DIR`**: Directory with KAGRA `.gwf` files.
- **`CHANNEL_DIR`**: Directory with K1 channel list CSVs.
  Set these variables:
  ```bash
  export KAGRA_DATA_DIR="/path/to/kagra/data"
  export CHANNEL_DIR="/path/to/k1/channels"
  ```
  
## Usage

### Step 1: Compute coherence (`run_coherence.py`)
Calculates coherence between K1 strain (`K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ`) and witness channels.

#### Command
```bash
python run_coherence.py --savedir <output_dir> --time <gps_time> --dur <duration> --ifo K1
```
- `--savedir`: Output directory (e.g., `K1_automatic`).
- `--time`: GPS start time (e.g., `1368986010`).
- `--dur`: Duration in seconds (default: `900`).
- `--ifo`: Interferometer (must be `K1`).

#### Example
```bash
python run_coherence.py --savedir K1_automatic --time 1368986010 --dur 900 --ifo K1
```

#### Outputs
- **Directory**: `<savedir>/<gps_time>/` (e.g., `K1_automatic/1368986010/`).
- **Coherence Files**: `<channel>_<starttime>_<endtime>.csv` (e.g., `K1_PEM_VOLT_AS_TABLE_GND_OUT_DQ_1368986010_1368986910.csv`).
- **Zero-Power Channels**: `zero_power_channels_<gps_time>.csv` (lists channels with zero PSD issues).
- **Plot**: `plots/scatter_coh_<gps_time>_<duration>s.html` (max coherence vs. frequency).

### Step 2: Select Witness Channels (`select_witnesses.py`)
Selects witness channels with coherence > 0.2 in a specified frequency band.

#### Command
```bash
python select_witnesses.py --ifo K1 --time <gps_time> --savedir <coherence_dir> --lowfreq <low_freq> --highfreq <high_freq>
```
- `--ifo`: Interferometer (must be `K1`).
- `--time`: GPS start time (optional if inferred from `savedir`).
- `--savedir`: Directory from `run_coherence.py` (e.g., `K1_automatic/1368986010/`).
- `--lowfreq`, `--highfreq`: Frequency range in Hz (e.g., `55`, `65`).

#### Example
```bash
python select_witnesses.py --ifo K1 --time 1368986010 --savedir K1_automatic/1368986010 --lowfreq 55 --highfreq 65
```

#### Outputs
- **Plots**: `<savedir>/plots/channels_coh_<gps_time>_a.html` and `_b.html` (top and second-top coherent channels).
- **CSV**: `max_corr_output_<gps_time>.csv` (frequency and coherence data).
- **Text File**: `max_corr_output_<gps_time>.txt`.
- **Channel List**: `chanlist_O4_<lowfreq>Hz-<highfreq>Hz.ini` (DeepClean input).

### Example Shell Script
```bash
#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate igwn-py38

IFO="K1"
TIME=1368986010
DURATION=900
LOW_FREQ=55.0
HIGH_FREQ=65.0

python run_coherence.py --savedir "${IFO}_automatic" --time "$TIME" --dur "$DURATION" --ifo "$IFO"
python select_witnesses.py --ifo "$IFO" --time "$TIME" --savedir "${IFO}_automatic/${TIME}/" --lowfreq "$LOW_FREQ" --highfreq "$HIGH_FREQ"
```

## Interpreting Outputs

### Coherence Files
- **Format**: Two columns (`frequency`, `coherence`).
- **Purpose**: Raw coherence spectra per witness channel.

### Zero-Power Channels
- **File**: `zero_power_channels_<gps_time>.csv`.
- **Content**: Channels with zero power spectral density (e.g., flat data).
- **Use**: Identify problematic channels for investigation.

### Maximum Coherence Plot
- **File**: `scatter_coh_<gps_time>_<duration>s.html`.
- **Details**: Frequency vs. max coherence, colored by channel group.
- **Use**: Overview of coherence peaks.

### Coherence Channel Plots
- **Files**: `channels_coh_<gps_time>_a.html`, `_b.html`.
- **Details**: Top two coherent channels per frequency bin in the range.
- **Use**: Visualize specific frequency band coherence.

### Maximum Coherence CSV
- **File**: `max_corr_output_<gps_time>.csv`.
- **Columns**: `frequency`, `channel1`, `corr1`, `channel2`, `corr2`, `group1`, `group2`.
- **Use**: Detailed coherence analysis.

### Channel List
- **File**: `chanlist_O4_<lowfreq>Hz-<highfreq>Hz.ini`.
- **Format**: Strain channel followed by witness channels (e.g., `K1:OMC-TRANS_DC_A_OUT_DQ`).
- **Use**: Input for DeepClean noise subtraction.

## Troubleshooting

- **Missing Data**: Check `$KAGRA_DATA_DIR` and channel CSV paths.
- **Zero-Power Warnings**: Review `zero_power_channels_<gps_time>.csv` for affected channels.
- **Empty Results**: Confirm `run_coherence.py` ran successfully and `savedir` matches.
- **Parallel Issues**: Adjust `processes=4` in `run_coherence.py` if resource limits are exceeded.
