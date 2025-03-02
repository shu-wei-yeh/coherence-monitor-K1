# #!/usr/bin/env python
# import argparse
# import os
# import pandas as pd
# import random
# from utils import (
#     get_times, calc_coherence, run_coherence, get_max_corr,
#     get_frame_files, get_strain_data, find_max_corr_channel, plot_max_corr_chan, give_group_v2
# )
# from gwpy.timeseries import TimeSeries
# from datetime import datetime, timedelta

# # Parse command-line arguments.
# parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
# parser.add_argument('--lowfreq', type=float, help='Lower boundary for frequency in DeepClean', required=True)
# parser.add_argument('--highfreq', type=float, help='Upper boundary for frequency in DeepClean', required=True)
# parser.add_argument('--savedir', default=os.getcwd(), type=str, help='Output directory where data is saved')
# parser.add_argument('--ifo', default='K1', type=str, help='Interferometer (default: K1)')
# parser.add_argument('--time', type=float, help='GPS start time', default=None)
# parser.add_argument('--dur', type=float, default=1024.0, help='Duration of data in seconds')
# args = parser.parse_args()

# # Robustly extract a numeric time from the savedir.
# savedir_abs = os.path.abspath(args.savedir)
# time_str = os.path.basename(os.path.normpath(savedir_abs))
# if not time_str.isdigit():
#     time_str = os.path.basename(os.path.dirname(savedir_abs))
# if not time_str.isdigit():
#     raise ValueError(f"Could not extract a numeric time from savedir '{args.savedir}'")
# time_val = int(time_str)
# print("Current time is:", time_val)

# print("Start loading CSV files and generating plots...")
# # Generate the plot of maximum coherence channels.
# # FFT is set to 10 and duration (for plot titles) is taken from args.dur.
# vals = plot_max_corr_chan(args.savedir, fft=10, ifo=args.ifo, flow=args.lowfreq, fhigh=args.highfreq, duration=args.dur)

# # Print the first few rows to check the details, especially 'corr1' and 'corr2' columns.
# print("Preview of 'corr1' and 'corr2' columns:")
# print(vals[['corr1', 'corr2']].head())

# # Save the entire DataFrame to a CSV file for further inspection with added spacing.
# output_csv = os.path.join(args.savedir, "max_corr_output.csv")
# vals.to_csv(output_csv, index=False, sep="\t", lineterminator="\n\n")
# print(f"Maximum correlation output saved to {output_csv}")

# # Save a pretty-printed version of the DataFrame to a text file.
# pretty_output = vals.to_string()
# output_txt = os.path.join(args.savedir, "max_corr_output.txt")
# with open(output_txt, "w") as f:
#     f.write(pretty_output)
# print(f"Pretty output saved to {output_txt}")

# print("Selecting witness channels to be copied to DeepClean")
# # Select rows based on frequency and coherence thresholds.
# vals_selc = vals.loc[
#     (vals['frequency'] > args.lowfreq) &
#     (vals['frequency'] < args.highfreq) &
#     ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
# ]

# # Obtain unique witness channels from both 'channel1' and 'channel2' columns.
# channels1 = vals_selc.sort_values('corr1', ascending=False)\
#                       .drop_duplicates(subset=['channel1'])['channel1'].tolist()
# channels2 = vals_selc.sort_values('corr2', ascending=False)\
#                       .drop_duplicates(subset=['channel2'])['channel2'].tolist()

# channels = channels1.copy()
# for c in channels2:
#     if c not in channels:
#         channels.append(c)

# # Write the channel list to a file. The first line is the strain channel.
# with open('chanlist_O4.ini', 'w') as f:
#     f.write(f"{args.ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n")
#     for c in channels:
#         # Replace the first underscore with a dash.
#         new_c = c.replace('_', '-', 1)
#         f.write(new_c + '\n')

# print("Channels written to chanlist_O4.ini")

#!/usr/bin/env python
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

savedir_abs = os.path.abspath(args.savedir)
time_val = args.time or int(os.path.basename(savedir_abs) if os.path.basename(savedir_abs).isdigit() else
                            os.path.basename(os.path.dirname(savedir_abs)))
if not isinstance(time_val, (int, float)):
    raise ValueError(f"Could not determine time from --time or savedir '{args.savedir}'")
print(f"Using time: {time_val}")

print("Generating coherence plots...")
vals = plot_max_corr_chan(args.savedir, fft=12, ifo=args.ifo, flow=args.lowfreq, fhigh=args.highfreq, duration=args.dur)
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
vals_selc = vals[(vals['frequency'] > args.lowfreq) & (vals['frequency'] < args.highfreq) &
                 ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))]
channels = list(set(vals_selc['channel1'].tolist() + vals_selc['channel2'].tolist()))

chanlist_file = os.path.join(args.savedir, f'chanlist_O4_{args.lowfreq}Hz-{args.highfreq}Hz.ini')
os.makedirs(os.path.dirname(chanlist_file), exist_ok=True)
with open(chanlist_file, 'w') as f:
    f.write(f"{args.ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n")
    for c in channels:
        new_c = c.replace('_', '-', 1)
        f.write(f"{new_c}\n")
print(f"Channels written to {chanlist_file}")