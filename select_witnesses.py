#!/usr/bin/env python
import argparse
import os
import pandas as pd
from utils import find_max_corr_channel, plot_max_corr_chan

parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
parser.add_argument('--lowfreq', type=float, required=True,
                    help='Lower frequency boundary')
parser.add_argument('--highfreq', type=float, required=True,
                    help='Upper frequency boundary')
parser.add_argument('--savedir', default=os.getcwd(), type=str,
                    help='Output directory')
parser.add_argument('--ifo', default='K1', type=str,
                    help='Interferometer (default: K1)')
parser.add_argument('--time', type=float, default=None, help='GPS start time')
parser.add_argument('--dur', type=float, default=900.0,
                    help='Duration in seconds')
args = parser.parse_args()

savedir_abs = os.path.abspath(args.savedir)
time_val = args.time or int(
    os.path.basename(savedir_abs) if os.path.basename(savedir_abs).isdigit()
    else os.path.basename(os.path.dirname(savedir_abs))
)
if not isinstance(time_val, (int, float)):
    raise ValueError(f"Could not determine time from --time or savedir '{args.savedir}'")
print(f"Using time: {time_val}")

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
    raise ValueError("No coherence data found.")

print("Preview of coherence data:")
print(vals[['frequency', 'corr1', 'corr2']].head())

output_csv = os.path.join(args.savedir, f"max_corr_output_{time_val}.csv")
vals.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")

output_txt = os.path.join(args.savedir, f"max_corr_output_{time_val}.txt")
with open(output_txt, "w") as f:
    f.write(vals.to_string())
print(f"Saved to {output_txt}")

print("Selecting witness channels...")
vals_selc = vals[
    (vals['frequency'] > args.lowfreq) &
    (vals['frequency'] < args.highfreq) &
    ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
]
channels = list(set(vals_selc['channel1'].tolist() + vals_selc['channel2'].tolist()))

chanlist_file = os.path.join(args.savedir,
                             f'chanlist_O4_{args.lowfreq}Hz-{args.highfreq}Hz.ini')
os.makedirs(os.path.dirname(chanlist_file), exist_ok=True)
with open(chanlist_file, 'w') as f:
    f.write(f"{args.ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n")
    for c in channels:
        new_c = c.replace('_', '-', 1)
        f.write(f"{new_c}\n")
print(f"Channels written to {chanlist_file}")