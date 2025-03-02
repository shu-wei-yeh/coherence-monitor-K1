## V1
#!/usr/bin/env python
from utils import (
    get_observing_segs, get_times, calc_coherence, run_coherence,
    get_max_corr, get_frame_files, get_strain_data, find_max_corr_channel, plot_max_corr_chan
)
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lowfreq', type=float, help='Lower boundary for frequency in DeepClean')
parser.add_argument('--highfreq', type=float, help='Upper boundary for frequency in DeepClean')
parser.add_argument('--savedir', default=os.curdir, type=str, help='Output directory in which data is saved')
parser.add_argument('--ifo', default='L1', type=str, help='Interferometer')
args = parser.parse_args()

def give_group(a):
    parts = a.split('_')
    return parts[1] if len(parts) > 1 else a

# ---- Extract numeric time from savedir robustly ----
# First, try the basename of savedir.
time_str = os.path.basename(os.path.normpath(args.savedir))
if not time_str.isdigit():
    # If not numeric, try the parent's basename.
    parent_dir = os.path.dirname(os.path.normpath(args.savedir))
    time_str = os.path.basename(parent_dir)
if not time_str.isdigit():
    raise ValueError(
        f"Could not extract numeric time from savedir {args.savedir}: "
        f"extracted string '{time_str}' is not numeric. "
        "Please ensure that the savedir path ends with a numeric time folder (e.g., '/.../123456789/')."
    )
time_ = int(time_str)
print('Current time is:', time_)

# ---- Load CSV files and generate the plot ----
print('Start loading the *.csv files and generating the plot')
try:
    vals = plot_max_corr_chan(args.savedir, 10, args.ifo)
except Exception as e:
    print(f"An error occurred during plotting: {e}")
    vals = pd.DataFrame()

if vals.empty or 'frequency' not in vals.columns:
    raise ValueError("Plotting did not return expected data with a 'frequency' column.")

# ---- Filter and sort channels for DeepClean ----
print('Print sorted list of witness channels to be copied to DeepClean')
# Determine which column names to use for the coherence values.
if 'corr1' in vals.columns and 'corr2' in vals.columns:
    frequency_col = 'frequency'
    corr1_col = 'corr1'
    corr2_col = 'corr2'
elif 'coherence' in vals.columns and 'Coherence' in vals.columns:
    frequency_col = 'frequency'
    corr1_col = 'coherence'
    corr2_col = 'Coherence'
else:
    raise ValueError("Expected coherence columns not found in the data.")

vals_selc = vals.loc[
    (vals[frequency_col] > args.lowfreq) &
    (vals[frequency_col] < args.highfreq) &
    ((vals[corr1_col] > 0.2) | (vals[corr2_col] > 0.2))
]

# Build unique lists of channels from the two coherence columns.
channels1 = vals_selc.sort_values([corr1_col], ascending=False) \
                     .drop_duplicates(['channel1'])['channel1'] \
                     .to_list()
channels2 = vals_selc.sort_values([corr2_col], ascending=False) \
                     .drop_duplicates(['channel2'])['channel2'] \
                     .to_list()

channels = list(channels1)
for c in channels2:
    if c not in channels:
        channels.append(c)

# ---- Write channel list to file ----
with open('chanlist_o4.ini','w') as f:
    f.write(args.ifo + ':CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n')
    for c in channels:
        # Remove any file extension from the channel name.
        f.write(c.split('.')[0] + '\n')

print("Channel list saved to 'chanlist_o4.ini'")

from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data,find_max_corr_channel,plot_max_corr_chan

from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lowfreq', type=float, help='lower boundary for frequncy in DeepClean')
parser.add_argument('--highfreq', type=float, help='upper boundary for frequncy in DeepClean')
parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory in which data is saved')
parser.add_argument('--ifo', default='L1', type=str, help='Interferometer')
args = parser.parse_args()

def give_group(a):
    group = a.split('_')[1]
    return group

time_ = int(args.savedir.split("/")[-2])
print('Current time is:', time_)

print('Start loading the *.csv files and doing the plot')
vals = plot_max_corr_chan(args.savedir, 10, args.ifo)

print('Print sorted list of witness channels to be copied to DeepClean')
vals_selc = vals.loc[(vals['frequency']>args.lowfreq) & (vals['frequency']<args.highfreq) & ((vals['coherence']>0.2) | (vals['Coherence'] > 0.2))]

channels = []
channels1 = vals_selc.sort_values(['coherence'], ascending=False).drop_duplicates(['channel1'])['channel1'].to_list()
channels2 = vals_selc.sort_values(['Coherence'], ascending=False).drop_duplicates(['channel2'])['channel2'].to_list()

channels = [c for c in channels1]
for c in channels2:
    if c not in channels1:
        channels.append(c)

with open('chanlist_o4.ini','w') as f:
    f.write(args.ifo+':CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n')
    for c in channels:
        f.write(c.split('.')[0]+'\n')



#!/usr/bin/env python
from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data, find_max_corr_channel, plot_max_corr_chan
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
parser.add_argument('--lowfreq', type=float, help='Lower boundary for frequency in DeepClean', required=True)
parser.add_argument('--highfreq', type=float, help='Upper boundary for frequency in DeepClean', required=True)
parser.add_argument('--savedir', default=os.getcwd(), type=str, help='Output directory where data is saved')
parser.add_argument('--ifo', default='K1', type=str, help='Interferometer (default: K1)')
args = parser.parse_args()

# A simple grouping function for channel names.
def give_group(a):
    # Assumes channel names like "K1-PEM-VOLT_AS_TABLE_GND_OUT_DQ"
    group = a.split('_')[1]
    return group

# Determine the timestamp from the savedir.
# We assume that the savedir (or its parent) is named with the GPS time.
savedir_abs = os.path.abspath(args.savedir)
try:
    # Try the basename of savedir.
    time_val = int(os.path.basename(savedir_abs))
except Exception:
    # Otherwise, try the parent folder.
    time_val = int(os.path.basename(os.path.dirname(savedir_abs)))
print("Current time is:", time_val)

print("Start loading CSV files and generating plots...")
# Plot the maximum coherence channels using the CSV files in savedir.
# The default FFT is 10 and the default duration for titles is 256 seconds.
vals = plot_max_corr_chan(args.savedir, fft=10, ifo=args.ifo, flow=args.lowfreq, fhigh=args.highfreq, duration=900)

print("Selecting witness channels to be copied to DeepClean")
# Select rows based on frequency and coherence thresholds.
vals_selc = vals.loc[
    (vals['frequency'] > args.lowfreq) &
    (vals['frequency'] < args.highfreq) &
    ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
]

# Obtain a sorted list of witness channels from both coherence columns.
channels1 = vals_selc.sort_values(['corr1'], ascending=False).drop_duplicates(['channel1'])['channel1'].to_list()
channels2 = vals_selc.sort_values(['corr2'], ascending=False).drop_duplicates(['channel2'])['channel2'].to_list()

channels = channels1.copy()
for c in channels2:
    if c not in channels1:
        channels.append(c)

# Write the channel list to a file. The first line is the strain channel.
with open('chanlist_o4.ini','w') as f:
    f.write(args.ifo + ':CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n')
    for c in channels:
        # Replace the first underscore with a dash.
        new_c = c.replace('_', '-', 1)
        f.write(new_c + '\n')

print("Channels written to chanlist_o4.ini")

# V2
#!/usr/bin/env python
from utils import get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data, find_max_corr_channel, plot_max_corr_chan
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
parser.add_argument('--lowfreq', type=float, help='Lower boundary for frequency in DeepClean', required=True)
parser.add_argument('--highfreq', type=float, help='Upper boundary for frequency in DeepClean', required=True)
parser.add_argument('--savedir', default=os.getcwd(), type=str, help='Output directory where data is saved')
parser.add_argument('--ifo', default='K1', type=str, help='Interferometer (default: K1)')
args = parser.parse_args()

# A simple grouping function for channel names.
def give_group(a):
    # Assumes channel names like "K1-PEM-VOLT_AS_TABLE_GND_OUT_DQ"
    group = a.split('_')[2]
    return group

# Determine the timestamp from the savedir.
# We assume that the savedir (or its parent) is named with the GPS time.
savedir_abs = os.path.abspath(args.savedir)
try:
    # Try the basename of savedir.
    time_val = int(os.path.basename(savedir_abs))
except Exception:
    # Otherwise, try the parent folder.
    time_val = int(os.path.basename(os.path.dirname(savedir_abs)))
print("Current time is:", time_val)

print("Start loading CSV files and generating plots...")
# Plot the maximum coherence channels using the CSV files in savedir.
# The default FFT is 10 and the default duration for titles is 256 seconds.
vals = plot_max_corr_chan(args.savedir, fft=10, ifo=args.ifo, flow=args.lowfreq, fhigh=args.highfreq, duration=900)

print("Selecting witness channels to be copied to DeepClean")
# Select rows based on frequency and coherence thresholds.
vals_selc = vals.loc[
    (vals['frequency'] > args.lowfreq) &
    (vals['frequency'] < args.highfreq) &
    ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
]

# Obtain a sorted list of witness channels from both coherence columns.
channels1 = vals_selc.sort_values(['corr1'], ascending=False).drop_duplicates(['channel1'])['channel1'].to_list()
channels2 = vals_selc.sort_values(['corr2'], ascending=False).drop_duplicates(['channel2'])['channel2'].to_list()

channels = channels1.copy()
for c in channels2:
    if c not in channels1:
        channels.append(c)

# Write the channel list to a file. The first line is the strain channel.
with open('chanlist_O4.ini','w') as f:
    f.write(args.ifo + ':CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n')
    for c in channels:
        # Replace the first underscore with a dash.
        new_c = c.replace('_', '-', 1)
        f.write(new_c + '\n')

print("Channels written to chanlist_O4.ini")




#!/usr/bin/env python
import argparse
import os
import pandas as pd
import random
from utils import (
    get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr,
    get_frame_files, get_strain_data, find_max_corr_channel, plot_max_corr_chan, give_group_v2
)
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
parser.add_argument('--lowfreq', type=float, help='Lower boundary for frequency in DeepClean', required=True)
parser.add_argument('--highfreq', type=float, help='Upper boundary for frequency in DeepClean', required=True)
parser.add_argument('--savedir', default=os.getcwd(), type=str, help='Output directory where data is saved')
parser.add_argument('--ifo', default='K1', type=str, help='Interferometer (default: K1)')
parser.add_argument('--dur', type=float, default=1024.0, help='Duration of data in seconds')
args = parser.parse_args()

# Robustly extract a numeric time from the savedir.
savedir_abs = os.path.abspath(args.savedir)
time_str = os.path.basename(os.path.normpath(savedir_abs))
if not time_str.isdigit():
    time_str = os.path.basename(os.path.dirname(savedir_abs))
if not time_str.isdigit():
    raise ValueError(f"Could not extract a numeric time from savedir '{args.savedir}'")
time_val = int(time_str)
print("Current time is:", time_val)

print("Start loading CSV files and generating plots...")
# Generate the plot of maximum coherence channels.
# FFT is set to 10, and we use a duration of 900 seconds for the plot title.
vals = plot_max_corr_chan(args.savedir, fft=10, ifo=args.ifo, flow=args.lowfreq, fhigh=args.highfreq, duration=args.dur)

print("Selecting witness channels to be copied to DeepClean")
# Select rows based on frequency and coherence thresholds.
vals_selc = vals.loc[
    (vals['frequency'] > args.lowfreq) &
    (vals['frequency'] < args.highfreq) &
    ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
]

# Obtain unique witness channels from both 'channel1' and 'channel2' columns.
channels1 = vals_selc.sort_values('corr1', ascending=False).drop_duplicates(subset=['channel1'])['channel1'].tolist()
channels2 = vals_selc.sort_values('corr2', ascending=False).drop_duplicates(subset=['channel2'])['channel2'].tolist()

channels = channels1.copy()
for c in channels2:
    if c not in channels:
        channels.append(c)

# Write the channel list to a file. The first line is the strain channel.
with open('chanlist_O4.ini','w') as f:
    f.write(f"{args.ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n")
    for c in channels:
        # Replace the first underscore with a dash.
        new_c = c.replace('_', '-', 1)
        f.write(new_c + '\n')

print("Channels written to chanlist_O4.ini")

## V3
#!/usr/bin/env python
import argparse
import os
import pandas as pd
import random
from utils import (
    get_observing_segs, get_times, calc_coherence, run_coherence, get_max_corr,
    get_frame_files, get_strain_data, find_max_corr_channel, plot_max_corr_chan, give_group_v2
)
from gwpy.timeseries import TimeSeries
from datetime import datetime, timedelta

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Select witness channels for DeepClean")
parser.add_argument('--lowfreq', type=float, help='Lower boundary for frequency in DeepClean', required=True)
parser.add_argument('--highfreq', type=float, help='Upper boundary for frequency in DeepClean', required=True)
parser.add_argument('--savedir', default=os.getcwd(), type=str, help='Output directory where data is saved')
parser.add_argument('--ifo', default='K1', type=str, help='Interferometer (default: K1)')
parser.add_argument('--dur', type=float, default=1024.0, help='Duration of data in seconds')
args = parser.parse_args()

# Robustly extract a numeric time from the savedir.
savedir_abs = os.path.abspath(args.savedir)
time_str = os.path.basename(os.path.normpath(savedir_abs))
if not time_str.isdigit():
    time_str = os.path.basename(os.path.dirname(savedir_abs))
if not time_str.isdigit():
    raise ValueError(f"Could not extract a numeric time from savedir '{args.savedir}'")
time_val = int(time_str)
print("Current time is:", time_val)

print("Start loading CSV files and generating plots...")
# Generate the plot of maximum coherence channels.
# FFT is set to 10 and duration (for plot titles) is taken from args.dur.
vals = plot_max_corr_chan(args.savedir, fft=10, ifo=args.ifo, flow=args.lowfreq, fhigh=args.highfreq, duration=args.dur)

print("Selecting witness channels to be copied to DeepClean")
# Select rows based on frequency and coherence thresholds.
vals_selc = vals.loc[
    (vals['frequency'] > args.lowfreq) &
    (vals['frequency'] < args.highfreq) &
    ((vals['corr1'] > 0.2) | (vals['corr2'] > 0.2))
]

# Obtain unique witness channels from both 'channel1' and 'channel2' columns.
channels1 = vals_selc.sort_values('corr1', ascending=False)\
                      .drop_duplicates(subset=['channel1'])['channel1'].tolist()
channels2 = vals_selc.sort_values('corr2', ascending=False)\
                      .drop_duplicates(subset=['channel2'])['channel2'].tolist()

channels = channels1.copy()
for c in channels2:
    if c not in channels:
        channels.append(c)

# Write the channel list to a file. The first line is the strain channel.
with open('chanlist_O4.ini', 'w') as f:
    f.write(f"{args.ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ\n")
    for c in channels:
        # Replace the first underscore with a dash.
        new_c = c.replace('_', '-', 1)
        f.write(new_c + '\n')

print("Channels written to chanlist_O4.ini")
