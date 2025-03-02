# #!/usr/bin/env python
# from utils import get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
# from gwpy.timeseries import TimeSeries
# from gwpy.time import to_gps
# from gwpy.segments import Segment, SegmentList
# from datetime import datetime
# import multiprocessing
# import pandas as pd
# import random
# import argparse
# import os
# import time
# import plotly.express as px
# import plotly

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Coherence Processing Script")
#     parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
#     parser.add_argument('--time', type=float, help='GPS start time', default=None)
#     parser.add_argument('--dur', type=float, default=1024.0, help='Duration of data in seconds')
#     parser.add_argument('--ifo', type=str, help='Interferometer: H1, L1, or K1')
#     parser.add_argument('--savedir', default=os.curdir, type=str, help='Output directory to save data')
#     args = parser.parse_args()

#     # Set parameters from command-line arguments.
#     t1_str = args.date
#     ifo = args.ifo
#     savedir = args.savedir
#     duration = args.dur

#     # Define the data segment using either a date or a GPS time.
#     if args.date is not None:
#         date_obj = datetime.strptime(t1_str, '%Y-%m-%d')
#         start_gps = to_gps(date_obj)
#         end_gps = start_gps + duration
#         seg_list = SegmentList([Segment(start_gps, end_gps)])
#     elif args.time is not None:
#         start_gps = args.time
#         end_gps = start_gps + duration
#         seg_list = SegmentList([Segment(start_gps, end_gps)])
#     else:
#         raise Exception("Either a date or a GPS time must be provided!")

#     # Generate time stamps from the segment list.
#     times_segs = get_times(seg_list, duration=duration)

#     # Load the list of voltage, OMC, and mic channels.
#     channel_path = os.path.join('/home/shu-wei.yeh/coherence-monitor/channel_files', ifo)

#     asc_chans = pd.read_csv(
#         os.path.join(channel_path, 'asc_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     cal_chans = pd.read_csv(
#         os.path.join(channel_path, 'cal_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     imc_chans = pd.read_csv(
#         os.path.join(channel_path, 'imc_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     lsc_chans = pd.read_csv(
#         os.path.join(channel_path, 'lsc_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     mic_chans = pd.read_csv(
#         os.path.join(channel_path, 'mic_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     omc_chans = pd.read_csv(
#         os.path.join(channel_path, 'omc_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     pem_chans = pd.read_csv(
#         os.path.join(channel_path, 'pem_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     psl_chans = pd.read_csv(
#         os.path.join(channel_path, 'psl_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     related_chans = pd.read_csv(
#         os.path.join(channel_path, 'related_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     tms_chans = pd.read_csv(
#         os.path.join(channel_path, 'tms_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     vis_chans = pd.read_csv(
#         os.path.join(channel_path, 'vis_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     volt_chans = pd.read_csv(
#         os.path.join(channel_path, 'volt_channels.csv'),
#         header=None, 
#         names=['channel']
#     )

#     # Randomly choose one GPS start time from the available times.
#     time_ = random.choice(times_segs)
#     print("Chosen GPS start time: {}".format(time_))

#     # Get the strain data for the interval [time_, time_+duration).
#     ht_data = get_strain_data(time_, time_ + duration, duration, ifo=ifo)
#     if ht_data is None:
#         raise RuntimeError("Failed to load h(t) data.")
#     print("Strain data loaded successfully.")

#     # Define a simple grouping function.
#     def give_group(a):
#         # For example, group by the third component of the channel name.
#         return a.split('_')[2]

#     # Define helper functions to run coherence for different channel sets.

#     def get_coherence_asc(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for OMC channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for ASC channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_cal(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for mic channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for CAL channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return

#     def get_coherence_imc(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for mic channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for IMC channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return

#     def get_coherence_lsc(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for OMC channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for LSC channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_mic(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for mic channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for MIC channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return

#     def get_coherence_omc(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for mic channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for OMC channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return

#     def get_coherence_pem(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for voltage channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for PEM channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_psl(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for voltage channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for PSL channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_related(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for voltage channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for Related channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_tms(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for voltage channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for TMS channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_vis(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for voltage channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for VIS channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    
#     def get_coherence_volt(channel_list, ifo, t0, strain_data, savedir, duration):
#         """
#         Retrieve witness frame files and run coherence calculation for voltage channels.
#         """
#         files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
#         print("Got {} witness file(s) for VOLT channels".format(len(files_)))
#         if not files_:
#             raise ValueError("No frame files found.")
#         run_coherence(
#             channel_list=channel_list,
#             frame_files=files_,
#             starttime=t0,
#             endtime=t0 + duration,
#             ifo=ifo,
#             strain_data=strain_data,
#             savedir=savedir
#         )
#         return
    

#     # Create processes for each channel type.
#     p1 = multiprocessing.Process(
#         target=get_coherence_asc,
#         args=(asc_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p2 = multiprocessing.Process(
#         target=get_coherence_cal,
#         args=(cal_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p3 = multiprocessing.Process(
#         target=get_coherence_imc,
#         args=(imc_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p4 = multiprocessing.Process(
#         target=get_coherence_lsc,
#         args=(lsc_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p5 = multiprocessing.Process(
#         target=get_coherence_mic,
#         args=(mic_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p6 = multiprocessing.Process(
#         target=get_coherence_omc,
#         args=(omc_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p7 = multiprocessing.Process(
#         target=get_coherence_pem,
#         args=(pem_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p8 = multiprocessing.Process(
#         target=get_coherence_psl,
#         args=(psl_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p9 = multiprocessing.Process(
#         target=get_coherence_related,
#         args=(related_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p10 = multiprocessing.Process(
#         target=get_coherence_tms,
#         args=(tms_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p11 = multiprocessing.Process(
#         target=get_coherence_vis,
#         args=(vis_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )
#     p12 = multiprocessing.Process(
#         target=get_coherence_volt,
#         args=(volt_chans['channel'], ifo, time_, ht_data, savedir, duration)
#     )

#     processes = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    
#     tic = time.time()
#     for p in processes:
#         p.start()
#     for p in processes:
#         p.join()
#     tac = time.time()
#     print("Coherence processing took {:.2f} seconds".format(tac - tic))

#     # Assume that run_coherence saves CSV files in a subdirectory named with the GPS time.
#     output_dir = os.path.join(savedir, f'{int(time_)}')

#     vals = get_max_corr(output_dir, save=True)
#     if vals.empty:
#         raise ValueError("No maximum correlation data found.")
#     vals['group'] = vals['channel'].apply(give_group)

#     # Create a scatter plot of maximum correlation vs. frequency.
#     fig = px.scatter(
#         vals, 
#         x="frequency", 
#         y="max_correlation",
#         hover_data=['channel'], 
#         color="group",
#         labels={"max_correlation": "Max Correlation", "frequency": "Frequency [Hz]"}
#     )

#     # Update layout: increase title font and legend font size.
#     fig.update_layout(
#         title=dict(
#             text="Max Coherence during {} -- {}".format(str(time_), str(time_ + duration)),
#             font=dict(family="Courier New, monospace", size=28, color="Blue")
#         ),
#         legend=dict(
#             font=dict(size=20)
#         )
#     )

#     # Update traces: increase the scatter point marker size.
#     fig.update_traces(marker=dict(size=20, opacity=0.8))

#     # Save the plot to an HTML file.
#     plot_filename = os.path.join('plots', f'scatter_coh_{int(time_)}_{duration}s.html')
#     os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
#     plotly.offline.plot(fig, filename=plot_filename)

#     print("Scatter plot saved to {}".format(plot_filename))

#!/usr/bin/env python
from utils import get_times, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from gwpy.time import to_gps
from gwpy.segments import Segment, SegmentList
import multiprocessing as mp
import pandas as pd
import argparse
import os
import time
import plotly.express as px
import plotly

CHANNEL_DIR = os.getenv('CHANNEL_DIR', '/home/shu-wei.yeh/coherence-monitor/channel_files')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coherence Processing Script")
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--time', type=float, help='GPS start time', default=None)
    parser.add_argument('--dur', type=float, default=900.0, help='Duration in seconds')
    parser.add_argument('--ifo', type=str, default='K1', help='Interferometer (default: K1)')
    parser.add_argument('--savedir', default=os.curdir, type=str, help='Output directory')
    args = parser.parse_args()

    if args.date:
        from datetime import datetime
        start_gps = to_gps(datetime.strptime(args.date, '%Y-%m-%d'))
        end_gps = start_gps + args.dur
    elif args.time:
        start_gps = args.time
        end_gps = start_gps + args.dur
    else:
        raise ValueError("Either --date or --time must be provided.")
    
    seg_list = SegmentList([Segment(start_gps, end_gps)])
    time_ = get_times(seg_list, duration=args.dur)[0]
    print(f"Processing GPS time: {time_}")

    channel_path = os.path.join(CHANNEL_DIR, args.ifo)
    # channel_types = ['asc', 'cal', 'imc', 'lsc', 'mic', 'omc', 'pem', 'psl', 'related', 'tms', 'vis', 'volt']
    channel_types = ['mic', 'omc', 'volt']
    channels = {}
    for ct in channel_types:
        try:
            channels[ct] = pd.read_csv(os.path.join(channel_path, f'{ct}_channels.csv'),
                                       header=None, names=['channel'])
        except FileNotFoundError:
            print(f"Warning: {ct}_channels.csv not found.")
            channels[ct] = pd.DataFrame(columns=['channel'])

    ht_data = get_strain_data(time_, time_ + args.dur, args.dur, ifo=args.ifo)
    if ht_data is None:
        raise RuntimeError("Failed to load strain data.")

    def process_coherence(channel_type, channel_list, ifo, t0, strain_data, savedir, dur):
        files_ = get_frame_files(t0, t0 + dur, dur, ifo=ifo)
        if not files_:
            print(f"No frame files for {channel_type} channels.")
            return
        print(f"Processing {channel_type} with {len(files_)} files.")
        run_coherence(channel_list=channel_list, frame_files=files_, starttime=t0,
                      endtime=t0 + dur, ifo=ifo, strain_data=strain_data, savedir=savedir)

    with mp.Pool(processes=4) as pool:
        pool.starmap(process_coherence, [(ct, channels[ct]['channel'].tolist(), args.ifo, time_, ht_data, args.savedir, args.dur)
                                         for ct in channel_types])
    
    output_dir = os.path.join(args.savedir, f'{int(time_)}')
    vals = get_max_corr(output_dir, save=True)
    if vals.empty:
        raise ValueError("No coherence data found.")
    
    def give_group(a):
        return a.split('_')[2] if len(a.split('_')) > 2 else 'UNKNOWN'
    
    vals['group'] = vals['channel'].apply(give_group)
    fig = px.scatter(vals, x="frequency", y="max_correlation", hover_data=['channel'], color="group",
                     labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"})
    fig.update_layout(title=dict(text=f"Max Coherence {time_} -- {time_ + args.dur}",
                                 font=dict(family="Courier New, monospace", size=28, color="Blue")),
                      legend=dict(font=dict(size=20)))
    fig.update_traces(marker=dict(size=20, opacity=0.8))
    
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plotly.offline.plot(fig, filename=os.path.join(plot_dir, f'scatter_coh_{int(time_)}_{args.dur}s.html'))
    print(f"Plot saved to {plot_dir}")






##
from gwpy.time import to_gps, from_gps
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.frequencyseries import FrequencySeries
import gwdatafind
import numpy as np
import pandas as pd
import os
import glob
import plotly.express as px
import plotly
import warnings

# Configurable base directories
DATA_DIR = os.getenv('KAGRA_DATA_DIR', '/data/KAGRA/raw')

def give_group_v2(a):
    """Extract group from channel string, e.g., 'K1:PEM-VOLT_AS_TABLE_GND_OUT_DQ' -> 'PEM'."""
    try:
        return a.split(':')[1].split('_')[0]
    except (IndexError, AttributeError):
        print(f"Warning: Malformed channel string '{a}'. Returning 'UNKNOWN'.")
        return 'UNKNOWN'

def get_strain_data(starttime, endtime, duration, ifo='K1', source_gwf=None, base_dir=None):
    """Retrieve strain data for K1 over the given time interval."""
    try:
        if ifo != 'K1':
            raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
        base_dir = base_dir or os.path.join(DATA_DIR, 'science')
        
        if source_gwf is None:
            j = int(starttime)
            computed_endtime = j + duration
            first_file = int(j - (j % 32))
            last_file = int(computed_endtime - (computed_endtime % 32))
            block = int((j - (j % 100000)) / 100000)
            strain_channel_files = [
                f"{base_dir}/{block}/K-K1_R-{i}-32.gwf"
                for i in range(first_file, last_file + 1, 32)
            ]
            existing_files = [f for f in strain_channel_files if os.path.exists(f)]
            if not existing_files:
                raise ValueError("No GWF files found for K1.")
            source_gwf = existing_files
            print(f"Selected strain data files: {source_gwf}")
        
        strain_channel = 'K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ'
        ht = TimeSeries.read(source=source_gwf, channel=strain_channel, start=starttime, end=endtime)
        return ht
    except Exception as e:
        print(f"Error in get_strain_data: {e}")
        return None

def get_frame_files(starttime, endtime, duration, ifo, directory=None):
    """Retrieve witness channel files for K1."""
    try:
        if ifo != 'K1':
            raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
        directory = directory or os.path.join(DATA_DIR, 'full')
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        j = int(starttime)
        computed_endtime = j + duration
        first_file = int(j - (j % 32))
        last_file = int(computed_endtime - (computed_endtime % 32))
        block = int((j - (j % 100000)) / 100000)
        witness_channel_files = [
            f"{directory}/{block}/K-K1_C-{i}-32.gwf"
            for i in range(first_file, last_file + 1, 32)
        ]
        files = sorted([f for f in witness_channel_files if os.path.exists(f)])
        return files
    except Exception as e:
        print(f"Error in get_frame_files: {e}")
        return []

def get_unsafe_channels(ifo):
    """Load unsafe channels from a CSV file."""
    valid_ifos = ['K1']
    if ifo not in valid_ifos:
        raise ValueError(f"Unsupported interferometer: {ifo}. Must be one of {valid_ifos}.")
    path = os.path.join('channel_files', ifo, f'{ifo}_unsafe_channels.csv')
    try:
        df = pd.read_csv(path)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Warning: Unsafe channels file at {path} not found or empty.")
        return pd.DataFrame(columns=['channel'])
    except Exception as e:
        raise RuntimeError(f"Error in get_unsafe_channels: {e}")

def get_times(seglist, duration=3600):
    """Generate time steps from a segment list."""
    if duration <= 0:
        raise ValueError("Duration must be positive.")
    times = [np.arange(i.start, i.end, duration) for i in seglist]
    return [item for sublist in times for item in sublist]

# def calc_coherence(channel2, frame_files, start_time, end_time, fft, overlap, window, strain_data, channel1=None):
#     """Compute coherence between strain and witness channels."""
#     t1, t2 = to_gps(start_time), to_gps(end_time)
#     if not isinstance(strain_data, TimeSeries):
#         raise ValueError("`strain_data` must be a TimeSeries object.")
    
#     try:
#         ts2 = TimeSeriesDict.read(frame_files, channels=channel2, start=t1, end=t2)
#     except Exception as e:
#         raise RuntimeError(f"Failed to read time series from {frame_files}: {e}")
    
#     coh = {}
#     for chan_name, witness_ts in ts2.items():
#         witness_sample_rate = witness_ts.sample_rate
#         # Only resample if rates differ
#         if strain_data.sample_rate != witness_sample_rate:
#             ts1_resampled = strain_data.resample(witness_sample_rate)
#         else:
#             ts1_resampled = strain_data
#         coherence = ts1_resampled.coherence(witness_ts, fftlength=fft, overlap=overlap, window='hann')
#         coh[chan_name] = coherence
    
#     for fs in coh.values():
#         inf_indices = np.where(np.isinf(fs.value))[0]
#         for i in inf_indices:
#             fs.value[i] = 1e-20 if i < 2 else (fs.value[i-2] + fs.value[i-1]) / 2
    
#     return coh

# def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, ifo, fft=12, overlap=6):
#     """Calculate and save coherence between strain and witness channels."""
#     if not channel_list:
#         raise ValueError("Channel list cannot be empty.")
    
#     t1, t2 = to_gps(starttime), to_gps(endtime)
#     outdir = os.path.join(savedir, f'{t1}')
#     os.makedirs(outdir, exist_ok=True)
    
#     if ifo != 'K1':
#         raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
#     strain_channel = f'{ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ'
    
#     try:
#         print(f"Calculating coherence for {len(channel_list)} channels...")
#         coherence_dict = calc_coherence(channel2=channel_list, frame_files=frame_files,
#                                         start_time=starttime, end_time=endtime, fft=fft,
#                                         overlap=overlap, window='hann', strain_data=strain_data)
#         for chan, coh in coherence_dict.items():
#             sanitized_channel = chan.replace(':', '_').replace('-', '_')
#             output_file = os.path.join(outdir, f"{sanitized_channel}_{t1}_{t2}.csv")
#             coh.write(output_file)
#             print(f"Saved coherence for {chan} to {output_file}")
#     except Exception as e:
#         print(f"Failed to calculate or save coherence: {e}")

def calc_coherence(channel2, frame_files, start_time, end_time, fft, overlap, window, strain_data, channel1=None):
    """Compute coherence between strain and witness channels, tracking zero-power warnings."""
    t1, t2 = to_gps(start_time), to_gps(end_time)
    if not isinstance(strain_data, TimeSeries):
        raise ValueError("`strain_data` must be a TimeSeries object.")
    
    try:
        ts2 = TimeSeriesDict.read(frame_files, channels=channel2, start=t1, end=t2)
    except Exception as e:
        raise RuntimeError(f"Failed to read time series from {frame_files}: {e}")
    
    coh = {}
    zero_power_channels = []  # List to store channels with zero-power warnings
    
    for chan_name, witness_ts in ts2.items():
        witness_sample_rate = witness_ts.sample_rate
        if strain_data.sample_rate != witness_sample_rate:
            ts1_resampled = strain_data.resample(witness_sample_rate)
        else:
            ts1_resampled = strain_data
        
        # Capture divide-by-zero warnings during coherence calculation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Ensure warnings are captured
            coherence = ts1_resampled.coherence(witness_ts, fftlength=fft, overlap=overlap, window='hann')
            if any("divide by zero" in str(warning.message) for warning in w):
                print(f"Warning: Zero power encountered for channel {chan_name}")
                zero_power_channels.append(chan_name)
        
        coh[chan_name] = coherence
    
    # Fix infinite values in coherence
    for fs in coh.values():
        inf_indices = np.where(np.isinf(fs.value))[0]
        for i in inf_indices:
            fs.value[i] = 1e-20 if i < 2 else (fs.value[i-2] + fs.value[i-1]) / 2
    
    return coh, zero_power_channels

def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, ifo, fft=12, overlap=6):
    """Calculate and save coherence, logging channels with zero-power issues."""
    if not channel_list:
        raise ValueError("Channel list cannot be empty.")
    
    t1, t2 = to_gps(starttime), to_gps(endtime)
    outdir = os.path.join(savedir, f'{t1}')
    os.makedirs(outdir, exist_ok=True)
    
    if ifo != 'K1':
        raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
    strain_channel = f'{ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ'
    
    try:
        print(f"Calculating coherence for {len(channel_list)} channels...")
        coherence_dict, zero_power_channels = calc_coherence(
            channel2=channel_list, frame_files=frame_files, start_time=starttime, end_time=endtime,
            fft=fft, overlap=overlap, window='hann', strain_data=strain_data
        )
        
        # Save coherence data
        for chan, coh in coherence_dict.items():
            sanitized_channel = chan.replace(':', '_').replace('-', '_')
            output_file = os.path.join(outdir, f"{sanitized_channel}_{t1}_{t2}.csv")
            coh.write(output_file)
            print(f"Saved coherence for {chan} to {output_file}")
        
        # Save channels with zero-power warnings to a CSV
        if zero_power_channels:
            zero_power_df = pd.DataFrame(zero_power_channels, columns=['channel'])
            zero_power_file = os.path.join(outdir, f"zero_power_channels_{t1}.csv")
            zero_power_df.to_csv(zero_power_file, index=False)
            print(f"Saved {len(zero_power_channels)} channels with zero-power issues to {zero_power_file}")
    
    except Exception as e:
        print(f"Failed to calculate or save coherence: {e}")

def get_max_corr(output_dir, save=False):
    """Process coherence CSV files to find max correlation."""
    output_dir = os.path.abspath(output_dir)
    files = glob.glob(os.path.join(output_dir, '*.csv'))
    # Exclude summary files
    files = [f for f in files if 'max_corr_output' not in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(f"No coherence CSV files found in {output_dir}")
    
    vals = []
    for file_path in files:
        try:
            base = os.path.basename(file_path)
            chan_name = base.split('DQ')[0] + 'DQ'
            fs = FrequencySeries.read(file_path)
            if len(fs.frequencies) < 2:
                raise ValueError("Insufficient frequency points.")
            n_diff = fs.frequencies.value[1] - fs.frequencies.value[0]
            ind1, ind2 = int(1 / n_diff), int(200 / n_diff)
            fs_sub = fs[ind1:ind2]
            max_value = fs_sub.max().value
            max_value_frequency = fs_sub.frequencies[fs_sub.argmax()].value
            if save and max_value > 0:
                vals.append((chan_name, max_value, max_value_frequency))
        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")
    
    return pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency']) if save else pd.DataFrame()

def combine_csv(dir_path, ifo):
    """Combine coherence CSV files, filtering unsafe channels."""
    dir_path = os.path.abspath(dir_path)
    all_files = glob.glob(os.path.join(dir_path, "*.csv"))
    # Exclude summary files
    all_files = [f for f in all_files if 'max_corr_output' not in os.path.basename(f)]
    if not all_files:
        raise FileNotFoundError(f"No coherence CSV files found in {dir_path}")
    
    chan_removes = get_unsafe_channels(ifo=ifo)['channel']
    chan_removes = [chan.replace(':', '_').replace('-', '_') for chan in chan_removes]
    filtered_files = [f for f in all_files if not any(os.path.basename(f).startswith(chan) for chan in chan_removes)]
    if not filtered_files:
        raise ValueError("No valid CSV files after filtering.")
    
    combined_data = []
    column_names = []
    for file_path in filtered_files:
        try:
            base_name = os.path.basename(file_path).split('_14')[0]
            column_freq = f"{base_name}_freq"
            column_corr = f"{base_name}_corr"
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] != 2:
                print(f"Warning: {file_path} has {df.shape[1]} columns; expected 2. Skipping.")
                continue
            combined_data.append(df)
            column_names.extend([column_freq, column_corr])
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if not combined_data:
        raise ValueError("No valid coherence files processed.")
    
    combined_df = pd.concat(combined_data, axis=1, ignore_index=True)
    if combined_df.shape[1] != len(column_names):
        raise ValueError("Mismatch between columns and names.")
    combined_df.columns = column_names
    return combined_df

def find_max_corr_channel(path, ifo, fft=12):
    """
    Find, for each frequency bin, the channels with the highest and second-highest coherence.
    
    Parameters:
      - path (str): Directory containing CSV files.
      - ifo (str): Interferometer identifier.
      - fft (float): FFT length used to compute frequency bins.
    
    Returns:
      DataFrame: A DataFrame with columns ['frequency', 'channel1', 'corr1', 'channel2', 'corr2'].
    """
    frame_df = combine_csv(path, ifo)
    if frame_df.empty:
        raise ValueError(f"No valid data found in the combined CSV files at {path}.")
    
    max_vals = []
    for i in range(len(frame_df)):
        try:
            corr_values = frame_df.iloc[i, 1::2]
            corr_sorted = corr_values.sort_values(ascending=False)
            top_columns = corr_sorted.index[:2]
            # Reconstruct clean channel names by removing '_corr' and timestamps
            chan_names = []
            for col in top_columns:
                # Remove '_corr' suffix
                base_name = col.replace('_corr', '')
                # Split by '_' and take parts up to the timestamp (last two elements are timestamps)
                parts = base_name.split('_')
                # Find the index where numeric timestamps likely start (assuming 10-digit GPS times)
                timestamp_idx = next((i for i, p in enumerate(parts) if p.isdigit() and len(p) >= 9), len(parts))
                clean_parts = parts[:timestamp_idx]
                # Replace the first '_' with ':' to restore the original format
                clean_name = ':'.join(clean_parts[:2]) + '_' + '_'.join(clean_parts[2:])
                chan_names.append(clean_name)
            max_corr_vals = corr_sorted.iloc[:2].tolist()
            freq = i / fft
            max_vals.append((freq, chan_names[0], max_corr_vals[0], chan_names[1], max_corr_vals[1]))
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    df_max = pd.DataFrame(max_vals, columns=['frequency', 'channel1', 'corr1', 'channel2', 'corr2'])
    return df_max

def plot_max_corr_chan(path, fft, ifo, duration, flow=0, fhigh=200):
    """
    Plot the highest and second-highest coherence channels across frequency bins.
    
    Parameters:
      - path (str): Directory containing CSV files.
      - fft (float): FFT length (used in frequency calculation).
      - ifo (str): Interferometer identifier.
      - flow (float): Lower frequency bound for plotting.
      - fhigh (float): Upper frequency bound for plotting.
      - duration (int): Duration used in plot titles.
    
    Returns:
      DataFrame: The DataFrame of maximum correlation channel data.
    """
    try:
        # Assume the directory name is a timestamp.
        time_ = int(os.path.basename(os.path.normpath(path)))
        vals = find_max_corr_channel(path=path, fft=fft, ifo=ifo)
        print("Data acquired; generating plots...")
        # Filter rows by frequency.
        vals = vals[(vals['frequency'] >= flow) & (vals['frequency'] <= fhigh)]
        # vals = vals.iloc[flow*fft:fhigh*fft+1]

        # Apply a grouping function to channels (assumes give_group_v2 is defined).
        vals['group1'] = vals['channel1'].apply(give_group_v2)
        vals['group2'] = vals['channel2'].apply(give_group_v2)
        
        # Plot highest coherence channel.
        fig1 = px.scatter(
            vals,
            x="frequency",
            y="corr1",
            hover_data=['channel1'],
            color="group1",
            labels={"corr1": "Max Coherence", "frequency": "Frequency [Hz]"}
        )
        fig1.update_layout(
            title={
                "text": f"Highest Coherence Channel at Each Frequency ({time_} to {time_ + duration})",
                "font": {"family": "Courier New, monospace", "size": 28, "color": "RebeccaPurple"}
            },
            font_size=28
        )
        fig1.update_traces(marker=dict(size=20, opacity=0.8))
        
        # Plot second-highest coherence channel.
        fig2 = px.scatter(
            vals,
            x="frequency",
            y="corr2",
            hover_data=['channel2'],
            color="group2",
            labels={"corr2": "Second Max Coherence", "frequency": "Frequency [Hz]"}
        )
        fig2.update_layout(
            title={
                "text": f"Second Highest Coherence Channel at Each Frequency ({time_} to {time_ + duration})",
                "font": {"family": "Courier New, monospace", "size": 28, "color": "RebeccaPurple"}
            },
            font_size=28
        )
        fig2.update_traces(marker=dict(size=20, opacity=0.8))
        
        # Create a subdirectory for plots.
        plot_dir = os.path.join(path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plotly.offline.plot(fig1, filename=os.path.join(plot_dir, f'channels_coh_{time_}_a.html'))
        plotly.offline.plot(fig2, filename=os.path.join(plot_dir, f'channels_coh_{time_}_b.html'))
        print("Plots saved successfully.")
        return vals
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        return pd.DataFrame()