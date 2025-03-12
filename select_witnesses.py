# #!/usr/bin/env python
# '''
# Script to select witness channels for DeepClean based on coherence analysis.
# Generates plots, saves coherence data, and creates a channel list from KAGRA (K1) data.
# '''

import argparse
import os
import pandas as pd
from utils import find_max_corr_channel, plot_max_corr_chan

parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
parser.add_argument('--lowfreq', type=float, required=True, help='Lower frequency boundary')
parser.add_argument('--highfreq', type=float, required=True, help='Upper frequency boundary')
parser.add_argument('--savedir', default=os.getcwd(), type=str, help='Output directory')
parser.add_argument('--ifo', default='K1', type=str, help='Interferometer (default: K1)')
parser.add_argument('--time', type=float, default=None, help='GPS start time')
parser.add_argument('--dur', type=float, default=900.0, help='Duration in seconds')
args = parser.parse_args()

# Validate frequency range
if args.lowfreq >= args.highfreq:
    raise ValueError(f"Lower frequency ({args.lowfreq}) must be less than upper frequency ({args.highfreq})")

savedir_abs = os.path.abspath(args.savedir)

# Determine start time
try:
    time_val = args.time or int(
        os.path.basename(savedir_abs) if os.path.basename(savedir_abs).isdigit()
        else os.path.basename(os.path.dirname(savedir_abs))
    )
except ValueError:
    raise ValueError(f"Could not infer a valid numeric time from savedir '{args.savedir}' or --time")
if not isinstance(time_val, (int, float)):
    raise ValueError(f"Time value '{time_val}' is not numeric")
print(f"Using time: {time_val}")

# Check if savedir is writable
if not os.access(savedir_abs, os.W_OK):
    raise PermissionError(f"Cannot write to directory '{savedir_abs}'")

print("Generating coherence plots...")
vals = plot_max_corr_chan(
    args.savedir,
    fft=12,
    ifo=args.ifo,
    flow=args.lowfreq,
    fhigh=args.highfreq,
    duration=args.dur,
)
if vals.empty:
    raise ValueError(f"No coherence data found in '{args.savedir}'")

# Display a preview of the top coherence values
print("Preview of coherence data:")
print(vals[['frequency', 'corr1', 'corr2']].head())

# Save coherence data to CSV
output_csv = os.path.join(savedir_abs, f"max_corr_output_{time_val}.csv")
vals.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

# Save coherence data as a text file
output_txt = os.path.join(savedir_abs, f"max_corr_output_{time_val}.txt")
with open(output_txt, "w") as f:
    f.write(vals.to_string())
print(f"Saved to {output_txt}")

print("Selecting witness channels...")
# Filter by frequency and coherence threshold
vals_selc = vals[
    (vals['frequency'] > args.lowfreq) &
    (vals['frequency'] < args.highfreq) &
    ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
]

# Get unique channels sorted by coherence
channels1 = vals_selc.sort_values('corr1', ascending=False).drop_duplicates('channel1')['channel1'].tolist()
channels2 = vals_selc.sort_values('corr2', ascending=False).drop_duplicates('channel2')['channel2'].tolist()

# Combine channels, prioritizing higher coherence (channels1 over channels2)
channels = channels1 + [c for c in channels2 if c not in channels1]
if not channels:
    print("Warning: No channels selected based on the coherence threshold of 0.2")

# Write selected channels to .ini file
chanlist_file = f'chanlist_O4_{args.lowfreq}Hz-{args.highfreq}Hz.ini'
with open(chanlist_file, 'w') as f:
    f.write(f"{args.ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n")
    for c in channels:
        new_c = c.replace('_', '-', 1)  # Replace first underscore with hyphen
        f.write(f"{new_c}\n")
print(f"Channels written to {chanlist_file}")