import os
import glob
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline

from gwpy.time import to_gps, from_gps
from gwpy.segments import DataQualityFlag, SegmentList
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.frequencyseries import FrequencySeries

# Configurable base directory for KAGRA data
DATA_DIR = os.getenv('KAGRA_DATA_DIR', '/data/KAGRA/raw')

# Frequency range constants for coherence analysis (adjustable here only)
RESTRICT_FREQ_LOW = 1    # Lower frequency bound in Hz
RESTRICT_FREQ_HIGH = 200 # Upper frequency bound in Hz


def give_group_v2(a):
    """Extract group from channel string.

    For example, 'K1:PEM-VOLT_AS_TABLE_GND_OUT_DQ' returns 'PEM'.
    """
    try:
        return a.split(':')[1].split('_')[0]
    except (IndexError, AttributeError):
        print(f"Warning: Malformed channel string '{a}'. Returning 'UNKNOWN'.")
        return 'UNKNOWN'


def get_strain_data(starttime, endtime, duration, ifo='K1', source_gwf=None,
                    base_dir=None):
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
        ht = TimeSeries.read(
            source=source_gwf,
            channel=strain_channel,
            start=starttime,
            end=endtime,
        )
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


def calc_coherence(channel2, frame_files, start_time, end_time, fft, overlap,
                   window, strain_data, channel1=None):
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

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            coherence = ts1_resampled.coherence(
                witness_ts,
                fftlength=fft,
                overlap=overlap,
                window='hann',
            )
            if any("divide by zero" in str(warning.message) for warning in w):
                print(f"Warning: Zero power encountered for channel {chan_name}")
                zero_power_channels.append(chan_name)

        coh[chan_name] = coherence

    for fs in coh.values():
        inf_indices = np.where(np.isinf(fs.value))[0]
        for i in inf_indices:
            fs.value[i] = 1e-20 if i < 2 else (fs.value[i - 2] + fs.value[i - 1]) / 2

    return coh, zero_power_channels


def run_coherence(channel_list, frame_files, starttime, endtime, strain_data,
                  savedir, ifo, fft=10, overlap=5):
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
            channel2=channel_list,
            frame_files=frame_files,
            start_time=starttime,
            end_time=endtime,
            fft=fft,
            overlap=overlap,
            window='hann',
            strain_data=strain_data,
        )

        for chan, coh in coherence_dict.items():
            sanitized_channel = chan.replace(':', '_').replace('-', '_')
            output_file = os.path.join(outdir, f"{sanitized_channel}_{t1}_{t2}.csv")
            coh.write(output_file)
            print(f"Saved coherence for {chan} to {output_file}")

        if zero_power_channels:
            zero_power_df = pd.DataFrame(zero_power_channels, columns=['channel'])
            zero_power_file = os.path.join(outdir, f"zero_power_channels_{t1}.csv")
            zero_power_df.to_csv(zero_power_file, index=False)
            print(f"Saved {len(zero_power_channels)} channels with zero-power "
                  f"issues to {zero_power_file}")
    except Exception as e:
        print(f"Failed to calculate or save coherence: {e}")


def get_max_corr(output_dir, restrict_freq_low=RESTRICT_FREQ_LOW,
                 restrict_freq_high=RESTRICT_FREQ_HIGH, save=False):
    """Find maximum coherence values from CSV files within a frequency range.

    Args:
        output_dir (str): Directory with coherence CSV files.
        restrict_freq_low (float): Lower frequency bound (default: RESTRICT_FREQ_LOW).
        restrict_freq_high (float): Upper frequency bound (default: RESTRICT_FREQ_HIGH).
        save (bool): If True, return DataFrame with results (default: False).

    Returns:
        pd.DataFrame: Columns ['channel', 'max_correlation', 'frequency'] if save=True, else empty.
    """
    output_dir = os.path.abspath(output_dir)
    files = glob.glob(os.path.join(output_dir, '*.csv'))
    files = [
        f for f in files
        if 'max_corr_output' not in os.path.basename(f) and
           'zero_power_channels' not in os.path.basename(f)
    ]
    if not files:
        raise FileNotFoundError(f"No coherence CSV files found in {output_dir}")

    if restrict_freq_low >= restrict_freq_high:
        raise ValueError("restrict_freq_low must be less than restrict_freq_high")

    vals = []
    for file_path in files:
        try:
            base = os.path.basename(file_path)
            chan_name = base.split('DQ')[0] + 'DQ' # Reconstruct channel name
            fs = FrequencySeries.read(file_path)
            if len(fs.frequencies) < 2:
                raise ValueError("Insufficient frequency points.")
            # Calculate frequency bin indices
            n_diff = fs.frequencies.value[1] - fs.frequencies.value[0] # Slice frequency range
            ind1, ind2 = int(restrict_freq_low / n_diff), int(restrict_freq_high / n_diff)
            if ind1 >= ind2 or ind1 < 0 or ind2 > len(fs.frequencies):
                print(f"Warning: Frequency range {restrict_freq_low}-{restrict_freq_high} Hz "
                      f"out of bounds for {file_path}. Skipping.")
                continue
            fs_sub = fs[ind1:ind2]
            max_value = fs_sub.max().value
            max_value_frequency = fs_sub.frequencies[fs_sub.argmax()].value
            if save and max_value > 0:
                vals.append((chan_name, max_value, max_value_frequency))
        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")

    return pd.DataFrame(
        vals,
        columns=['channel', 'max_correlation', 'frequency']
    ) if save else pd.DataFrame()


def combine_csv(dir_path, ifo):
    """Combine coherence CSV files into a single DataFrame, filtering unsafe channels.

    Args:
        dir_path (str): Directory containing coherence CSV files.

    Returns:
        pd.DataFrame: Combined data with frequency and coherence columns.
    """
    dir_path = os.path.abspath(dir_path)
    all_files = glob.glob(os.path.join(dir_path, "*.csv"))
    all_files = [f for f in all_files if 'max_corr_output' not in os.path.basename(f)]
    if not all_files:
        raise FileNotFoundError(f"No coherence CSV files found in {dir_path}")
    
    # Filter out unsafe channels
    chan_removes = get_unsafe_channels(ifo=ifo)['channel']
    chan_removes = [chan.replace(':', '_').replace('-', '_') for chan in chan_removes]
    filtered_files = [
        f for f in all_files if not any(os.path.basename(f).startswith(chan) for chan in chan_removes)
    ]
    if not filtered_files:
        raise ValueError("No valid CSV files after filtering.")

    combined_data = []
    column_names = []
    for file_path in filtered_files:
        try:
            base_name = os.path.basename(file_path).split('_14')[0] # Split at timestamp
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


def find_max_corr_channel(path, ifo, fft=10):
    """Identify the top two most coherent channels per frequency bin.

    Args:
        fft (float): FFT length used in coherence calculation (default: 12).

    Returns:
        pd.DataFrame: Columns ['frequency', 'channel1', 'corr1', 'channel2', 'corr2'].
    """
    frame_df = combine_csv(path, ifo)
    if frame_df.empty:
        raise ValueError(f"No valid data found in the combined CSV files at {path}.")

    max_vals = []
    for i in range(len(frame_df)):
        try:
            # Extract coherence values (every 2nd column)
            corr_values = frame_df.iloc[i, 1::2]
            corr_sorted = corr_values.sort_values(ascending=False)
            top_columns = corr_sorted.index[:2] # Top two channels
            chan_names = []
            for col in top_columns:
                base_name = col.replace('_corr', '')
                parts = base_name.split('_')
                # Find timestamp and reconstruct channel name
                timestamp_idx = next(
                    (i for i, p in enumerate(parts) if p.isdigit() and len(p) >= 9),
                    len(parts)
                )
                clean_parts = parts[:timestamp_idx]
                clean_name = ':'.join(clean_parts[:2]) + '_' + '_'.join(clean_parts[2:])
                chan_names.append(clean_name)
            max_corr_vals = corr_sorted.iloc[:2].tolist()
            freq = i / fft # Convert bin index to frequency
            max_vals.append((freq, chan_names[0], max_corr_vals[0],
                            chan_names[1], max_corr_vals[1]))
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    # Create DataFrame with top two channels per frequency
    df_max = pd.DataFrame(
        max_vals,
        columns=['frequency', 'channel1', 'corr1', 'channel2', 'corr2']
    )
    return df_max


def plot_max_corr_chan(path, fft, ifo, duration, flow=0, fhigh=200):
    """Generate scatter plots of the top two coherent channels across frequency bins.

    Args:
        path (str): Directory with coherence data.
        fft (float): FFT length in seconds.
        duration (float): Time segment duration in seconds.
        flow (float): Lower frequency bound for plotting (default: 0).
        fhigh (float): Upper frequency bound for plotting (default: 200).

    Returns:
        pd.DataFrame: Data used for plotting or empty if failed.
    """
    try:
        time_ = int(os.path.basename(os.path.normpath(path)))
        vals = find_max_corr_channel(path=path, fft=fft, ifo=ifo)
        print("Data acquired; generating plots...")
        vals = vals[(vals['frequency'] >= flow) & (vals['frequency'] <= fhigh)]

        # Add group names for coloring
        vals['group1'] = vals['channel1'].apply(give_group_v2)
        vals['group2'] = vals['channel2'].apply(give_group_v2)

        # Plot highest coherence channels
        fig1 = px.scatter(
            vals,
            x="frequency",
            y="corr1",
            hover_data=['channel1'],
            color="group1",
            labels={"corr1": "Max Coherence", "frequency": "Frequency [Hz]"},
        )
        fig1.update_layout(
            title={
                "text": f"Highest Coherence Channel at Each Frequency "
                        f"({time_} to {time_ + duration})",
                "font": {"family": "Courier New, monospace", "size": 28,
                         "color": "RebeccaPurple"},
            },
            font_size=28,
        )
        fig1.update_traces(marker=dict(size=20, opacity=0.8))

        # Plot 2nd-highest coherence channels
        fig2 = px.scatter(
            vals,
            x="frequency",
            y="corr2",
            hover_data=['channel2'],
            color="group2",
            labels={"corr2": "Second Max Coherence", "frequency": "Frequency [Hz]"},
        )
        fig2.update_layout(
            title={
                "text": f"Second Highest Coherence Channel at Each Frequency "
                        f"({time_} to {time_ + duration})",
                "font": {"family": "Courier New, monospace", "size": 28,
                         "color": "RebeccaPurple"},
            },
            font_size=28,
        )
        fig2.update_traces(marker=dict(size=20, opacity=0.8))

        plot_dir = os.path.join(path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plotly.offline.plot(fig1, filename=os.path.join(plot_dir,
                            f'channels_coh_{time_}_a.html'))
        plotly.offline.plot(fig2, filename=os.path.join(plot_dir,
                            f'channels_coh_{time_}_b.html'))
        print("Plots saved successfully.")
        return vals
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        return pd.DataFrame()


def create_coherence_plot(df, group_name, segment_start, segment_end, output_dir,
                          freq_low=RESTRICT_FREQ_LOW, freq_high=RESTRICT_FREQ_HIGH,
                          linewidth=2):
    """Create a line plot of coherence for a group of channels.

    Args:
        df (pd.DataFrame): DataFrame with 'Frequency [Hz]', 'Coherence', and 'Channel' columns.
        group_name (str): Name of the channel group (e.g., 'PEM', 'OMC').
        segment_start (int): Start GPS time of the segment.
        segment_end (int): End GPS time of the segment.
        output_dir (str): Base directory to save the plot.
        freq_low (float): Lower frequency bound for x-axis (default: RESTRICT_FREQ_LOW).
        freq_high (float): Upper frequency bound for x-axis (default: RESTRICT_FREQ_HIGH).
        linewidth (int): Width of plot lines (default: 2).
    """
    if df.empty:
        print(f"No data to plot for {group_name}, segment {segment_start}-{segment_end}")
        return

    fig = px.line(
        df,
        x='Frequency [Hz]',
        y='Coherence',
        color='Channel',
        title=f'Coherence between strain and {group_name} channels '
              f'({segment_start} to {segment_end})',
        labels={'Frequency [Hz]': 'Frequency [Hz]', 'Coherence': 'Coherence'}
    )
    # Update line width for all traces
    for trace in fig.data:
        trace.line.width = linewidth

    fig.update_layout(
        yaxis_range=[0, 1],
        xaxis_range=[freq_low, freq_high],
        font=dict(size=18),
        legend_title=dict(font=dict(size=18))
    )

    # Create output directory structure
    plot_dir = os.path.join(output_dir, str(segment_start), group_name)
    os.makedirs(plot_dir, exist_ok=True)
    output_file = os.path.join(plot_dir,
                               f"coh_plot_{group_name}_{freq_low}Hz-{freq_high}Hz_"
                               f"{segment_start}_{segment_end}.html")
    fig.write_html(output_file)
    print(f"Saved plot to {output_file}")