#!/usr/bin/env python
import os
import argparse
import time

import pandas as pd
import plotly.express as px
import plotly.offline
import multiprocessing as mp

from gwpy.time import to_gps
from gwpy.segments import Segment, SegmentList
from utils import (get_times, run_coherence, get_max_corr, get_frame_files,
                   get_strain_data)

# Configurable channel directory
CHANNEL_DIR = os.getenv('CHANNEL_DIR',
                        '/home/shu-wei.yeh/coherence-monitor/channel_files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coherence Processing Script")
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD format')
    parser.add_argument('--time', type=float, help='GPS start time', default=None)
    parser.add_argument('--dur', type=float, default=900.0,
                        help='Duration in seconds')
    parser.add_argument('--ifo', type=str, default='K1',
                        help='Interferometer (default: K1)')
    parser.add_argument('--savedir', default=os.curdir, type=str,
                        help='Output directory')
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

    channel_types = [
        'asc', 'cal', 'imc', 'lsc', 'mic', 'omc', 'pem', 'psl', 'related',
        'tms', 'vis', 'volt'
    ]
    
    channels = {}
    for ct in channel_types:
        try:
            channels[ct] = pd.read_csv(
                os.path.join(channel_path, f'{ct}_channels.csv'),
                header=None,
                names=['channel'],
            )
        except FileNotFoundError:
            print(f"Warning: {ct}_channels.csv not found.")
            channels[ct] = pd.DataFrame(columns=['channel'])

    ht_data = get_strain_data(time_, time_ + args.dur, args.dur, ifo=args.ifo)
    if ht_data is None:
        raise RuntimeError("Failed to load strain data.")

    def process_coherence(channel_type, channel_list, ifo, t0, strain_data,
                          savedir, dur):
        files_ = get_frame_files(t0, t0 + dur, dur, ifo=ifo)
        if not files_:
            print(f"No frame files for {channel_type} channels.")
            return
        print(f"Processing {channel_type} with {len(files_)} files.")
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + dur,
            ifo=ifo,
            strain_data=strain_data,
            savedir=savedir,
        )

    with mp.Pool(processes=4) as pool:
        pool.starmap(
            process_coherence,
            [(ct, channels[ct]['channel'].tolist(), args.ifo, time_, ht_data,
              args.savedir, args.dur) for ct in channel_types]
        )

    output_dir = os.path.join(args.savedir, f'{int(time_)}')
    vals = get_max_corr(output_dir, save=True)
    if vals.empty:
        raise ValueError("No coherence data found.")

    def give_group(a):
        return a.split('_')[2] if len(a.split('_')) > 2 else 'UNKNOWN'

    vals['group'] = vals['channel'].apply(give_group)
    fig = px.scatter(
        vals,
        x="frequency",
        y="max_correlation",
        hover_data=['channel'],
        color="group",
        labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"},
    )
    fig.update_layout(
        title=dict(
            text=f"Max Coherence {time_} -- {time_ + args.dur}",
            font=dict(family="Courier New, monospace", size=28, color="Blue"),
        ),
        legend=dict(font=dict(size=20)),
    )
    fig.update_traces(marker=dict(size=20, opacity=0.8))

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plotly.offline.plot(fig, filename=os.path.join(plot_dir,
                        f'scatter_coh_{int(time_)}_{args.dur}s.html'))
    print(f"Plot saved to {plot_dir}")