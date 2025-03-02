## V1
#!/usr/bin/env python
from utils import get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from gwpy.time import to_gps
from gwpy.segments import Segment, SegmentList
from datetime import datetime
import multiprocessing
import pandas as pd
import random
import argparse
import os
import time
import plotly.express as px
import plotly

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coherence Processing Script")
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--time', type=float, help='GPS start time', default=None)
    parser.add_argument('--dur', type=float, default=1024.0, help='Duration of data in seconds')
    parser.add_argument('--ifo', type=str, help='Interferometer: H1, L1, or K1')
    parser.add_argument('--savedir', default=os.curdir, type=str, help='Output directory to save data')
    args = parser.parse_args()

    # Set parameters from command-line arguments.
    t1_str = args.date
    ifo = args.ifo
    savedir = args.savedir
    duration = args.dur

    # Define the data segment using either a date or a GPS time.
    if args.date is not None:
        date_obj = datetime.strptime(t1_str, '%Y-%m-%d')
        start_gps = to_gps(date_obj)
        end_gps = start_gps + duration
        seg_list = SegmentList([Segment(start_gps, end_gps)])
    elif args.time is not None:
        start_gps = args.time
        end_gps = start_gps + duration
        seg_list = SegmentList([Segment(start_gps, end_gps)])
    else:
        raise Exception("Either a date or a GPS time must be provided!")

    # Generate time stamps from the segment list.
    times_segs = get_times(seg_list, duration=duration)

    # Load the list of voltage channels.
    channel_path = os.path.join('/home/shu-wei.yeh/coherence-monitor/channel_files', ifo)

    volt_chans = pd.read_csv(
        os.path.join(channel_path, 'volt_channels.csv'),
        header=None, 
        names=['channel']
        )

    omc_chans = pd.read_csv(
        os.path.join(channel_path, 'omc_channels.csv'),
        header=None, 
        names=['channel']
        )
    
    mic_chans = pd.read_csv(
        os.path.join(channel_path, 'mic_channels.csv'),
        header=None, 
        names=['channel']
        )

    # Randomly choose one GPS start time from the available times.
    time_ = random.choice(times_segs)
    print("Chosen GPS start time: {}".format(time_))

    # Get the strain data for the interval [time_, time_+duration).
    ht_data = get_strain_data(time_, time_ + duration, duration, ifo=ifo)
    if ht_data is None:
        raise RuntimeError("Failed to load h(t) data.")
    print("Strain data loaded successfully.")

    # Define a simple grouping function.
    def give_group(a):
        # For example, group by the second component of the channel name.
        return a.split('_')[2]

    def get_coherence_volt(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation.
        """
        # Get witness frame files from the specified directory over [t0, t0+duration)
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s)".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")

        # Pass the full list of files (not just the first file) to run_coherence.
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return
    
    def get_coherence_omc(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation.
        """
        # Get witness frame files from the specified directory over [t0, t0+duration)
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s)".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")

        # Pass the full list of files (not just the first file) to run_coherence.
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return

    def get_coherence_mic(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation.
        """
        # Get witness frame files from the specified directory over [t0, t0+duration)
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s)".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")

        # Pass the full list of files (not just the first file) to run_coherence.
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return
    
    # Run the coherence computation in a separate process.
    p1 = multiprocessing.Process(
        target=get_coherence_volt,
        args=(volt_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )

    p2 = multiprocessing.Process(
        target=get_coherence_omc,
        args=(omc_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )

    p3 = multiprocessing.Process(
        target=get_coherence_mic,
        args=(mic_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )

    tic = time.time()
    p1.start()
    p1.join()
    tac = time.time()
    print("Coherence processing took {:.2f} seconds".format(tac - tic))

    # Assume that run_coherence saves CSV files in a subdirectory named with the GPS time.
    output_dir = os.path.join(savedir, f'{int(time_)}')
    vals = get_max_corr(output_dir, save=True)
    if vals.empty:
        raise ValueError("No maximum correlation data found.")
    vals['group'] = vals['channel'].apply(give_group)

    # Create a scatter plot of maximum correlation vs. frequency.
    fig = px.scatter(
        vals, 
        x="frequency", 
        y="max_correlation",
        hover_data=['channel'], 
        color="group",
        labels={"max_correlation": "Max Correlation", "frequency": "Frequency [Hz]"}
    )

    # Update layout: increase title font and legend font size.
    fig.update_layout(
        title=dict(
            text="Max Coherence during {} -- {}".format(str(time_), str(time_ + duration)),
            font=dict(family="Courier New, monospace", size=28, color="Blue")
        ),
        legend=dict(
            font=dict(size=20)  # Increased legend font size
        )
    )

    # Update traces: increase the scatter point marker size.
    fig.update_traces(marker=dict(size=20, opacity=0.8))  # Increased marker size

    # Save the plot to an HTML file.
    plot_filename = os.path.join('plots', f'scatter_coh_{int(time_)}_{duration}s.html')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plotly.offline.plot(fig, filename=plot_filename)

print("Scatter plot saved to {}".format(plot_filename))


##V2

from utils import get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from gwpy.time import to_gps  # ensure this is imported for time conversion
from gwpy.segments import Segment, SegmentList
from datetime import datetime, timedelta
import multiprocessing
import pandas as pd
import random
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=str, help='YYYY-MM-DD')
    parser.add_argument('--time', type=float, help='GPS time', default=None)
    parser.add_argument('--dur', type=float, default=1024.0, help='duration of data in secs')
    parser.add_argument('--ifo', type=str, help='H1, L1, or K1')
    parser.add_argument('--savedir', default=os.curdir, type=str, help='output directory to save data')
    args = parser.parse_args()

    t1 = args.date
    ifo = args.ifo
    savedir = args.savedir

    if args.date is not None:
        # Parse date string to datetime object.
        date_obj = datetime.strptime(t1, '%Y-%m-%d')
        # For manual seg_list, convert date_obj to GPS time.
        start_gps = to_gps(date_obj)
        # Here we define the segment to last 900 seconds.
        end_gps = start_gps + 900
        # Manually create seg_list.
        seg_list = SegmentList([Segment(start_gps, end_gps)])
    elif args.time is not None:
        start_gps = args.time
        end_gps = start_gps + 900
        seg_list = SegmentList([Segment(start_gps, end_gps)])
    else:
        raise Exception("Either date or GPS time needs to be defined!")

    # Generate time stamps from the manual seg_list.
    times_segs = get_times(seg_list, duration=1024)

    channel_path = f'/home/shu-wei.yeh/coherence-monitor/channel_files/{ifo}/'
    volt_chans = pd.read_csv(os.path.join(channel_path, 'volt_channels.csv'), header=None, names=['channel'])

    # Pick a random time stamp from our manual segment.
    time_ = random.choice(times_segs)
    print("Time is {}".format(time_))

    # Retrieve strain data for the chosen time window.
    ht_data = get_strain_data(time_, time_ + 900, ifo=ifo)
    print("Got h(t) data")

    def give_group(a):
        group = a.split('_')[1]
        return group

    # def get_coherence_volt(channel_list=volt_chans['channel'], ifo=ifo, t0=time_, strain_data=ht_data):
    #     files_ = get_frame_files(t0, t0 + 900, ifo=ifo, directory='/data/KAGRA/raw/full/')
    #     print("Got {} files".format(len(files_)))
    #     if not files_:
    #         raise ValueError("No frame files found.")
    #     # Pass only one file (e.g., the first file) to run_coherence:
    #     run_coherence(channel_list=channel_list, frame_files=files_[0], starttime=t0, 
    #                 endtime=t0+900, ifo=ifo, strain_data=ht_data, savedir=savedir)
    #     return

    def get_coherence_volt(channel_list, ifo, t0, strain_data, savedir):
        # Get frame files that fall within the requested time window.
        files_found = get_frame_files(t0, t0+900, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} frame files".format(len(files_found)))
        if not files_found:
            raise ValueError("No frame files found.")
        # Select the first file (or add logic to choose the best file)
        frame_file = files_found[0]
        run_coherence(channel_list=channel_list, frame_files=frame_file, starttime=t0, 
                    endtime=t0+900, ifo=ifo, strain_data=strain_data, savedir=savedir)


    p1 = multiprocessing.Process(target=get_coherence_volt)

    import time
    tic = time.time()
    p1.start()
    p1.join()
    tac = time.time()
    print("Coherence processing took {:.2f} seconds".format(tac - tic))

    file_path = os.path.join(savedir, f'{int(time_)}', '')
    vals = get_max_corr(file_path, save=True)
    vals['group'] = vals['channel'].apply(give_group)

    import plotly.express as px
    import plotly

    fig = px.scatter(vals, x="frequency", y="max_correlation", 
                     hover_data=['channel'], color="group", 
                     labels={"max_correlation": "Max Correlation", "frequency": "Frequency [Hz]"})
    fig.update_layout(
        title=dict(text="Max Coherence during {} -- {}".format(str(time_), str(time_ + 900)), 
                   font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))
    )
    # Note: plotly.offline.plot produces an HTML file by default.
    plotly.offline.plot(fig, filename=f'plots/scatter_coh_{int(time_)}.html')



## V3
#!/usr/bin/env python
from utils import get_times, calc_coherence, run_coherence, get_max_corr, get_frame_files, get_strain_data
from gwpy.timeseries import TimeSeries
from gwpy.time import to_gps
from gwpy.segments import Segment, SegmentList
from datetime import datetime
import multiprocessing
import pandas as pd
import random
import argparse
import os
import time
import plotly.express as px
import plotly

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coherence Processing Script")
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--time', type=float, help='GPS start time', default=None)
    parser.add_argument('--dur', type=float, default=1024.0, help='Duration of data in seconds')
    parser.add_argument('--ifo', type=str, help='Interferometer: H1, L1, or K1')
    parser.add_argument('--savedir', default=os.curdir, type=str, help='Output directory to save data')
    args = parser.parse_args()

    # Set parameters from command-line arguments.
    t1_str = args.date
    ifo = args.ifo
    savedir = args.savedir
    duration = args.dur

    # Define the data segment using either a date or a GPS time.
    if args.date is not None:
        date_obj = datetime.strptime(t1_str, '%Y-%m-%d')
        start_gps = to_gps(date_obj)
        end_gps = start_gps + duration
        seg_list = SegmentList([Segment(start_gps, end_gps)])
    elif args.time is not None:
        start_gps = args.time
        end_gps = start_gps + duration
        seg_list = SegmentList([Segment(start_gps, end_gps)])
    else:
        raise Exception("Either a date or a GPS time must be provided!")

    # Generate time stamps from the segment list.
    times_segs = get_times(seg_list, duration=duration)

    # Load the list of voltage, OMC, and mic channels.
    channel_path = os.path.join('/home/shu-wei.yeh/coherence-monitor/channel_files', ifo)

    volt_chans = pd.read_csv(
        os.path.join(channel_path, 'volt_channels.csv'),
        header=None, 
        names=['channel']
    )

    omc_chans = pd.read_csv(
        os.path.join(channel_path, 'omc_channels.csv'),
        header=None, 
        names=['channel']
    )
    
    mic_chans = pd.read_csv(
        os.path.join(channel_path, 'mic_channels.csv'),
        header=None, 
        names=['channel']
    )

    related_chans = pd.read_csv(
        os.path.join(channel_path, 'related_channels.csv'),
        header=None, 
        names=['channel']
    )

    # Randomly choose one GPS start time from the available times.
    time_ = random.choice(times_segs)
    print("Chosen GPS start time: {}".format(time_))

    # Get the strain data for the interval [time_, time_+duration).
    ht_data = get_strain_data(time_, time_ + duration, duration, ifo=ifo)
    if ht_data is None:
        raise RuntimeError("Failed to load h(t) data.")
    print("Strain data loaded successfully.")

    # Define a simple grouping function.
    def give_group(a):
        # For example, group by the third component of the channel name.
        return a.split('_')[2]

    # Define helper functions to run coherence for different channel sets.
    def get_coherence_volt(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation for voltage channels.
        """
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s) for VOLT channels".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return
    
    def get_coherence_omc(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation for OMC channels.
        """
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s) for OMC channels".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return
    
    def get_coherence_mic(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation for mic channels.
        """
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s) for MIC channels".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return

    def get_coherence_related(channel_list, ifo, t0, strain_data, savedir, duration):
        """
        Retrieve witness frame files and run coherence calculation for mic channels.
        """
        files_ = get_frame_files(t0, t0 + duration, duration, ifo=ifo, directory='/data/KAGRA/raw/full/')
        print("Got {} witness file(s) for Related channels".format(len(files_)))
        if not files_:
            raise ValueError("No frame files found.")
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + duration,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir
        )
        return

    # Create processes for each channel type.
    p1 = multiprocessing.Process(
        target=get_coherence_volt,
        args=(volt_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )
    p2 = multiprocessing.Process(
        target=get_coherence_omc,
        args=(omc_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )
    p3 = multiprocessing.Process(
        target=get_coherence_mic,
        args=(mic_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )
    p4 = multiprocessing.Process(
        target=get_coherence_related,
        args=(related_chans['channel'], ifo, time_, ht_data, savedir, duration)
    )

    # processes = [p1, p2, p3]
    processes = [p1, p2, p3, p4]
    
    tic = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    tac = time.time()
    print("Coherence processing took {:.2f} seconds".format(tac - tic))

    # Assume that run_coherence saves CSV files in a subdirectory named with the GPS time.
    output_dir = os.path.join(savedir, f'{int(time_)}')

    vals = get_max_corr(output_dir, save=True)
    if vals.empty:
        raise ValueError("No maximum correlation data found.")
    vals['group'] = vals['channel'].apply(give_group)

    # Create a scatter plot of maximum correlation vs. frequency.
    fig = px.scatter(
        vals, 
        x="frequency", 
        y="max_correlation",
        hover_data=['channel'], 
        color="group",
        labels={"max_correlation": "Max Correlation", "frequency": "Frequency [Hz]"}
    )

    # Update layout: increase title font and legend font size.
    fig.update_layout(
        title=dict(
            text="Max Coherence during {} -- {}".format(str(time_), str(time_ + duration)),
            font=dict(family="Courier New, monospace", size=28, color="Blue")
        ),
        legend=dict(
            font=dict(size=20)
        )
    )

    # Update traces: increase the scatter point marker size.
    fig.update_traces(marker=dict(size=20, opacity=0.8))

    # Save the plot to an HTML file.
    plot_filename = os.path.join('plots', f'scatter_coh_{int(time_)}_{duration}s.html')
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plotly.offline.plot(fig, filename=plot_filename)

    print("Scatter plot saved to {}".format(plot_filename))