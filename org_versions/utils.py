# from gwpy.time import to_gps, from_gps
# from gwpy.segments import DataQualityFlag, SegmentList
# from gwpy.timeseries import TimeSeries, TimeSeriesDict
# from gwpy.frequencyseries import FrequencySeries
# import gwdatafind
# import numpy as np
# import pandas as pd
# import os
# import glob
# import plotly.express as px
# import plotly

# def give_group_v2(a):
#     # For a channel string like "K1:PEM-VOLT_AS_TABLE_GND_OUT_DQ", return "PEM"
#     group = a.split(':')[1].split('_')[0]
#     return group

# def get_strain_data(starttime, endtime, duration, ifo='K1', source_gwf=None):
#     """
#     Retrieve strain data for the interferometer K1 over the given time interval.
#     """
#     try:
#         if ifo != 'K1':
#             raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
        
#         if source_gwf is None:
#             # Use the provided starttime and duration.
#             j = int(starttime)
#             computed_endtime = j + duration  
#             # Files are assumed to start every 32 seconds.
#             first_file = int(j - (j % 32))
#             last_file = int(computed_endtime - (computed_endtime % 32))
#             # Compute a block directory (adjust if necessary)
#             block = int((j - (j % 100000)) / 100000)
#             # Build a list of expected strain data file names.
#             strain_channel_files = [
#                 f"/data/KAGRA/raw/science/{block}/K-K1_R-{i}-32.gwf"
#                 for i in range(first_file, last_file + 1, 32)
#             ]
#             # Filter for files that exist.
#             existing_files = [f for f in strain_channel_files if os.path.exists(f)]
#             if len(existing_files) == 0:
#                 raise ValueError("No GWF files found for K1.")
#             source_gwf = existing_files
#             print(f"Selected strain data files: {source_gwf}")
        
#         # Read the strain data using TimeSeries.read (a single channel)
#         strain_channel = 'K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ'
#         ht = TimeSeries.read(
#             source=source_gwf,
#             channel=strain_channel,
#             start=starttime,
#             end=endtime
#         )
#         return ht
#     except Exception as e:
#         print(f"An error occurred in get_strain_data: {e}")
#         return None

# def get_frame_files(starttime, endtime, duration, ifo, directory=None):
#     """
#     Retrieve frame (witness channel) files for the interferometer K1 over the given time interval.
#     """
#     try:
#         if ifo != 'K1':
#             raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
#         if directory is None:
#             raise ValueError("A directory must be specified for 'K1' witness channel data.")
#         if not os.path.isdir(directory):
#             raise FileNotFoundError(f"Directory not found: {directory}")
        
#         j = int(starttime)
#         computed_endtime = j + duration
#         first_file = int(j - (j % 32))
#         last_file = int(computed_endtime - (computed_endtime % 32))
#         block = int((j - (j % 100000)) / 100000)
#         # Build a list of expected witness channel file names.
#         witness_channel_files = [
#             f"/data/KAGRA/raw/full/{block}/K-K1_C-{i}-32.gwf"
#             for i in range(first_file, last_file + 1, 32)
#         ]
#         # Filter the list by checking file existence.
#         files = sorted([f for f in witness_channel_files if os.path.exists(f)])
#         return files
#     except Exception as e:
#         print(f"An error occurred in get_frame_files: {e}")
#         return []
    
# def get_unsafe_channels(ifo):
#     """
#     Load the unsafe channels for a given interferometer (K1) from a CSV file.

#     Parameters:
#       - ifo (str): Interferometer identifier (must be 'K1').

#     Returns:
#       - pd.DataFrame: DataFrame containing the unsafe channels. If the file is not found
#                       or is empty, returns an empty DataFrame.
#     """
#     valid_ifos = ['K1']
#     if ifo not in valid_ifos:
#         raise ValueError(f"Unsupported interferometer: {ifo}. Must be one of {valid_ifos}.")
#     path = os.path.join('channel_files', ifo, f'{ifo}_unsafe_channels.csv')
#     try:
#         df = pd.read_csv(path)
#         return df
#     except FileNotFoundError:
#         # Instead of raising an error, log a warning and return an empty DataFrame.
#         print(f"Warning: Unsafe channels file not found at {path}. Proceeding without filtering unsafe channels.")
#         return pd.DataFrame(columns=['channel'])
#     except pd.errors.EmptyDataError:
#         print(f"Warning: Unsafe channels file is empty at {path}. Proceeding without filtering unsafe channels.")
#         return pd.DataFrame(columns=['channel'])
#     except Exception as e:
#         raise RuntimeError(f"An unexpected error occurred in get_unsafe_channels: {e}")

# def get_times(seglist, duration=3600):
#     times = [np.arange(i.start, i.end, duration) for i in seglist]
#     times_flat = [item for sublist in times for item in sublist]
#     return times_flat

# def calc_coherence(channel2, frame_files, start_time, end_time, fft, overlap, window, strain_data, channel1=None):
#     """
#     Compute the coherence between strain data and all witness (frame) channels.

#     Parameters:
#       - channel2 (list or str): The witness channel(s) to be read.
#       - frame_files (str or list): The file path(s) for the witness channel data.
#       - start_time (float): Start time (GPS seconds).
#       - end_time (float): End time (GPS seconds).
#       - fft (float): The FFT length (in seconds) for the coherence calculation.
#       - overlap (float): The overlap (in seconds) between FFT segments.
#       - strain_data (TimeSeries): The strain data as a TimeSeries object.
#       - channel1: (Optional) Not used in this implementation.

#     Returns:
#       - dict: A dictionary where keys are witness channel names and values are the corresponding coherence FrequencySeries.
#     """
#     # Convert start and end times to GPS-compatible times.
#     t1 = to_gps(start_time)
#     t2 = to_gps(end_time)

#     if not isinstance(strain_data, TimeSeries):
#         raise ValueError("The parameter `strain_data` must be a TimeSeries object.")

#     try:
#         # Read the witness (frame) data into a TimeSeriesDict.
#         ts2 = TimeSeriesDict.read(
#             frame_files,
#             channels=channel2,
#             start=t1,
#             end=t2
#         )
#     except FileNotFoundError:
#         raise FileNotFoundError(f"Frame file(s) not found: {frame_files}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to read time series from {frame_files}: {e}")

#     coh = {}
#     # Loop over each witness channel in the TimeSeriesDict.
#     for chan_name, witness_ts in ts2.items():
#         # Get the sample rate for the current witness channel.
#         witness_sample_rate = witness_ts.sample_rate

#         # Resample the strain data to match the witness channel's sample rate.
#         ts1_resampled = strain_data.resample(witness_sample_rate)

#         # Compute the coherence between the resampled strain data and the witness channel.
#         coherence = ts1_resampled.coherence(
#             witness_ts,
#             fftlength=fft,
#             overlap=overlap,
#             window='hann'
#         )
#         coh[chan_name] = coherence
    
#         # Loop over each FrequencySeries in the dictionary and fix infinite values.
#     for key, fs in coh.items():
#         # Find indices where the coherence value is infinite.
#         inf_indices = np.where(np.isinf(fs.value))[0]
#         for i in inf_indices:
#             try:
#                 # Replace infinite values with the average of the two previous values.
#                 fs.value[i] = (fs.value[i - 2] + fs.value[i - 1]) / 2
#             except IndexError:
#                 # If there is an IndexError (e.g., at the beginning), set to a small value.
#                 fs.value[i] = 1e-20

#     return coh

# def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, ifo, fft=12, overlap=6):
#     """
#     Calculate and save the coherence between strain data and witness channels.
    
#     This function calculates the coherence between the given strain data and all witness channels 
#     specified in `channel_list` (using the frame files). The coherence for each witness channel is 
#     saved as a CSV file in a subdirectory of `savedir` based on the start time.
    
#     Parameters:
#       - channel_list (list): A list of witness channel names.
#       - frame_files (str or list): The file path(s) for the witness channel data.
#       - starttime (float): Start time in GPS seconds.
#       - endtime (float): End time in GPS seconds.
#       - strain_data (TimeSeries): The strain data as a TimeSeries object.
#       - savedir (str): The base directory to save output files.
#       - ifo (str): The interferometer identifier (must be 'K1').
#       - fft (float, optional): The FFT length (in seconds) for the coherence calculation.
#       - overlap (float, optional): The overlap (in seconds) between FFT segments.
#     """
#     # Convert start and end times to GPS times for naming and output.
#     t1, t2 = to_gps(starttime), to_gps(endtime)
    
#     # Create an output directory based on the start time.
#     outdir = os.path.join(savedir, f'{t1}')
#     if not os.path.exists(outdir):
#         print(f"Creating the output directory: {outdir}")
#         os.makedirs(outdir)
    
#     # Determine the strain channel name.
#     if ifo == 'K1':
#         strain_channel = f'{ifo}:CAL-CS_PROC_DARM_STRAIN_DBL_DQ'
#     else:
#         raise ValueError(f"Unsupported interferometer: {ifo}. Must be 'K1'.")
    
#     try:
#         print(f"Calculating coherence for witness channels: {', '.join(channel_list)} ...")
#         # Call the updated calc_coherence with the full channel list.
#         # Note: We pass the original starttime and endtime so that calc_coherence
#         # can perform its own conversion.
#         coherence_dict = calc_coherence(
#             channel2=channel_list,
#             frame_files=frame_files,
#             start_time=starttime,
#             end_time=endtime,
#             fft=fft,
#             overlap=overlap,
#             window='hann',
#             strain_data=strain_data,
#             channel1=None
#         )
        
#         # Iterate over each channel's coherence result and save the output.
#         for chan, coh in coherence_dict.items():
#             # Sanitize the channel name to create a valid filename.
#             sanitized_channel = chan.replace(':', '_').replace('-', '_')
#             output_file = os.path.join(outdir, f"{sanitized_channel}_{t1}_{t2}.csv")
#             coh.write(output_file)
#             print(f"Saved coherence data for {chan} to {output_file}")
            
#     except Exception as e:
#         print(f"Failed to calculate or save coherence: {e}")

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

def calc_coherence(channel2, frame_files, start_time, end_time, fft, overlap, window, strain_data, channel1=None):
    """Compute coherence between strain and witness channels."""
    t1, t2 = to_gps(start_time), to_gps(end_time)
    if not isinstance(strain_data, TimeSeries):
        raise ValueError("`strain_data` must be a TimeSeries object.")
    
    try:
        ts2 = TimeSeriesDict.read(frame_files, channels=channel2, start=t1, end=t2)
    except Exception as e:
        raise RuntimeError(f"Failed to read time series from {frame_files}: {e}")
    
    coh = {}
    for chan_name, witness_ts in ts2.items():
        witness_sample_rate = witness_ts.sample_rate
        # Only resample if rates differ
        if strain_data.sample_rate != witness_sample_rate:
            ts1_resampled = strain_data.resample(witness_sample_rate)
        else:
            ts1_resampled = strain_data
        coherence = ts1_resampled.coherence(witness_ts, fftlength=fft, overlap=overlap, window='hann')
        coh[chan_name] = coherence
    
    for fs in coh.values():
        inf_indices = np.where(np.isinf(fs.value))[0]
        for i in inf_indices:
            fs.value[i] = 1e-20 if i < 2 else (fs.value[i-2] + fs.value[i-1]) / 2
    
    return coh

def run_coherence(channel_list, frame_files, starttime, endtime, strain_data, savedir, ifo, fft=12, overlap=6):
    """Calculate and save coherence between strain and witness channels."""
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
        coherence_dict = calc_coherence(channel2=channel_list, frame_files=frame_files,
                                        start_time=starttime, end_time=endtime, fft=fft,
                                        overlap=overlap, window='hann', strain_data=strain_data)
        for chan, coh in coherence_dict.items():
            sanitized_channel = chan.replace(':', '_').replace('-', '_')
            output_file = os.path.join(outdir, f"{sanitized_channel}_{t1}_{t2}.csv")
            coh.write(output_file)
            print(f"Saved coherence for {chan} to {output_file}")
    except Exception as e:
        print(f"Failed to calculate or save coherence: {e}")

# def get_max_corr(output_dir, save=False):
#     """
#     Process all CSV files in `output_dir` to compute the maximum correlation value
#     (and its corresponding frequency) over a default frequency band from 1 Hz to 200 Hz.
    
#     Parameters:
#       output_dir (str): Directory containing CSV files.
#       save (bool): If True, return a DataFrame of channels with max correlation > 0.
    
#     Returns:
#       DataFrame: If save is True, a DataFrame with columns ['channel', 'max_correlation', 'frequency'];
#                  otherwise, an empty DataFrame.
#     """
#     # Use an absolute path for consistency.
#     output_dir = os.path.abspath(output_dir)
#     files = glob.glob(os.path.join(output_dir, '*.csv'))
#     if not files:
#         raise FileNotFoundError(f"No CSV files found in directory: {output_dir}")
    
#     vals = []
#     for file_path in files:
#         try:
#             # Extract a channel name by splitting on 'DQ'
#             base = os.path.basename(file_path)
#             chan_name = base.split('DQ')[0] + 'DQ'
#             # chan_name = base.split('/')[-1].split('DQ')[0] + 'DQ'
            
#             # Read the FrequencySeries saved in the CSV file.
#             fs = FrequencySeries.read(file_path)
            
#             # Check that there are at least two frequency points to determine the spacing.
#             if len(fs.frequencies) < 2:
#                 raise ValueError("Insufficient frequency points in FrequencySeries.")
            
#             # Compute the frequency difference using the first two frequency points.
#             n1, n2 = fs.frequencies.value[0], fs.frequencies.value[1]
#             n_diff = n2 - n1
            
#             # For the default frequency band of 1 Hz to 200 Hz.
#             # We test around 1 Hz to 400 Hz for KAGRA's data.
#             ind1, ind2 = int(1 / n_diff), int(200 / n_diff)
#             fs_sub = fs[ind1:ind2]
            
#             max_value = fs_sub.max().value
#             max_value_frequency = fs_sub.frequencies[fs_sub.argmax()].value
            
#             if save and max_value > 0:
#                 vals.append((chan_name, max_value, max_value_frequency))
#         except Exception as e:
#             print(f"Failed to process file {file_path}: {e}")
    
#     if save:
#         df_vals = pd.DataFrame(vals, columns=['channel', 'max_correlation', 'frequency'])
#         return df_vals[df_vals['max_correlation'] > 0]
    
#     return pd.DataFrame()

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

# def combine_csv(dir_path, ifo):
#     """
#     Combine CSV files in the given directory after filtering out unsafe channels.
    
#     Parameters:
#       - dir_path (str): Directory containing CSV files.
#       - ifo (str): Interferometer identifier (used to filter unsafe channels).
    
#     Returns:
#       DataFrame: A combined DataFrame whose columns are renamed based on each file.
#     """
#     # Use absolute path for consistency.
#     dir_path = os.path.abspath(dir_path)
#     all_files = glob.glob(os.path.join(dir_path, "*.csv"))
#     if not all_files:
#         raise FileNotFoundError(f"No CSV files found in directory: {dir_path}")
    
#     # Get the list of unsafe channels and sanitize their names.
#     chan_removes = get_unsafe_channels(ifo=ifo)['channel']
#     chan_removes = [chan.replace(':', '_').replace('-', '_') for chan in chan_removes]
    
#     # Filter out files whose basenames start with any unsafe channel.
#     filtered_files = [
#         file for file in all_files
#         if not any(os.path.basename(file).startswith(chan) for chan in chan_removes)
#     ]
#     if not filtered_files:
#         raise ValueError("No valid CSV files found after filtering unsafe channels.")
    
#     combined_data = []
#     column_names = []
#     for file_path in filtered_files:
#         try:
#             # Use part of the filename to generate column names.
#             base_name = os.path.basename(file_path).split('_14')[0]
#             column_freq = f"{base_name}_freq"
#             column_corr = f"{base_name}_corr"
#             df = pd.read_csv(file_path, header=None)
            
#             # Adjust if the file does not have exactly 2 columns.
#             if df.shape[1] == 1:
#                 print(f"Warning: {file_path} has {df.shape[1]} column; expected 2. Adding a column of NaNs.")
#                 df[1] = pd.NA
#             elif df.shape[1] > 2:
#                 print(f"Warning: {file_path} has {df.shape[1]} columns; expected 2. Using only the first 2 columns.")
#                 df = df.iloc[:, :2]
                
#             combined_data.append(df)
#             column_names.extend([column_freq, column_corr])
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")
    
#     if not combined_data:
#         raise ValueError("No CSV files could be processed.")
    
#     # Concatenate dataframes side by side.
#     combined_df = pd.concat(combined_data, axis=1, ignore_index=True)
#     if combined_df.shape[1] != len(column_names):
#         raise ValueError("Mismatch between the number of columns and column names.")
#     combined_df.columns = column_names
#     return combined_df

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


# def find_max_corr_channel(path, ifo, fft=12):
#     """
#     Find, for each frequency bin, the channels with the highest and second-highest coherence.
    
#     Parameters:
#       - path (str): Directory containing CSV files.
#       - ifo (str): Interferometer identifier.
#       - fft (float): FFT length used to compute frequency bins.
    
#     Returns:
#       DataFrame: A DataFrame with columns ['frequency', 'channel1', 'corr1', 'channel2', 'corr2'].
#     """
#     frame_df = combine_csv(path, ifo)
#     if frame_df.empty:
#         raise ValueError(f"No valid data found in the combined CSV files at {path}.")
    
#     max_vals = []
#     # Loop over each row (each frequency bin).
#     for i in range(len(frame_df)):
#         try:
#             # Assume that every second column (starting with index 1) holds coherence values.
#             corr_values = frame_df.iloc[i, 1::2]
#             # Sort descending by coherence.
#             corr_sorted = corr_values.sort_values(ascending=False)
#             # Get the top two channel column indices.
#             top_columns = corr_sorted.index[:2]
#             # Reconstruct channel names from the column names.
#             chan_names = [
#                 col.replace('_corr', '').replace(f'{ifo}_', f'{ifo}:')
#                 for col in top_columns
#             ]
#             max_corr_vals = corr_sorted.iloc[:2].tolist()
#             # Compute the frequency (row index divided by the FFT length).
#             freq = i / fft
#             max_vals.append((freq, chan_names[0], max_corr_vals[0],
#                              chan_names[1], max_corr_vals[1]))
#         except Exception as e:
#             print(f"Error processing row {i}: {e}")
#             continue
#     df_max = pd.DataFrame(max_vals, columns=['frequency', 'channel1', 'corr1', 'channel2', 'corr2'])
#     return df_max

# def find_max_corr_channel(path, ifo, fft=12):
#     """
#     Find, for each frequency bin, the channels with the highest and second-highest coherence.
    
#     Parameters:
#       - path (str): Directory containing CSV files.
#       - ifo (str): Interferometer identifier.
#       - fft (float): FFT length used to compute frequency bins.
    
#     Returns:
#       DataFrame: A DataFrame with columns ['frequency', 'channel1', 'corr1', 'channel2', 'corr2'].
#     """
#     frame_df = combine_csv(path, ifo)
#     if frame_df.empty:
#         raise ValueError(f"No valid data found in the combined CSV files at {path}.")
    
#     max_vals = []
#     for i in range(len(frame_df)):
#         try:
#             corr_values = frame_df.iloc[i, 1::2]
#             corr_sorted = corr_values.sort_values(ascending=False)
#             top_columns = corr_sorted.index[:2]
#             # Reconstruct channel names and remove timestamp suffix
#             chan_names = [
#                 col.replace('_corr', '').replace(f'{ifo}_', f'{ifo}:').split('_')[0] + '_' + '_'.join(col.split('_')[1:-2])
#                 for col in top_columns
#             ]
#             max_corr_vals = corr_sorted.iloc[:2].tolist()
#             freq = i / fft
#             max_vals.append((freq, chan_names[0], max_corr_vals[0], chan_names[1], max_corr_vals[1]))
#         except Exception as e:
#             print(f"Error processing row {i}: {e}")
#             continue
#     df_max = pd.DataFrame(max_vals, columns=['frequency', 'channel1', 'corr1', 'channel2', 'corr2'])
#     return df_max

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