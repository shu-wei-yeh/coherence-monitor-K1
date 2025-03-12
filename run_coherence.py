# ##!/usr/bin/env python
# '''
# Script to compute coherence between K1 strain and witness channels.
# Processes data in parallel, saves coherence results, and generates visualizations.
# '''

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
                   get_strain_data, give_group_v2, create_coherence_plot)

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
    channel_types = ["mic", "omc", "related", "volt"]

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
        """Process coherence for a single group independently."""
        files_ = get_frame_files(t0, t0 + dur, dur, ifo=ifo)
        if not files_:
            print(f"No frame files for {channel_type} channels.")
            return
        print(f"Processing {channel_type} with {len(files_)} files.")
        
        # Use a copy of strain_data to ensure no cross-group modification
        strain_copy = strain_data.copy()
        run_coherence(
            channel_list=channel_list,
            frame_files=files_,
            starttime=t0,
            endtime=t0 + dur,
            ifo=ifo,
            strain_data=strain_copy,
            savedir=savedir,
        )
        outdir = os.path.join(savedir, str(int(t0)))
        coherence_data = []
        for chan in channel_list:
            sanitized_channel = chan.replace(':', '_').replace('-', '_')
            csv_file = os.path.join(outdir, f"{sanitized_channel}_{int(t0)}_{int(t0 + dur)}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, header=None, names=['Frequency [Hz]', 'Coherence'])
                df['Channel'] = chan
                coherence_data.append(df)
        
        if coherence_data:
            combined_df = pd.concat(coherence_data, ignore_index=True)
            line_plot_dir = os.path.join(savedir, 'line_plots')
            create_coherence_plot(
                df=combined_df,
                group_name=channel_type,
                segment_start=int(t0),
                segment_end=int(t0 + dur),
                output_dir=line_plot_dir,
                freq_low=200,
                freq_high=400,
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

    vals['group'] = vals['channel'].apply(give_group_v2)

    fig = px.scatter(
        vals,
        x="frequency",
        y="max_correlation",
        hover_data=['channel'],
        color="group",
        labels={"max_correlation": "Max Coherence", "frequency": "Frequency [Hz]"},
    )
    fig.update_layout(
        title={
            "text": f"Max Coherence from ({time_} -- {time_ + args.dur})",
            "font": {"family": "Courier New, monospace", "size": 28,
                     "color": "RebeccaPurple"},
        },
        font_size=28,
    )
    fig.update_traces(marker=dict(size=20, opacity=0.8))

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f'scatter_coh_{int(time_)}_{args.dur}s.html'
    plotly.offline.plot(fig, filename=os.path.join(plot_dir, plot_filename))
    print(f"Scatter plot saved to {plot_dir}")