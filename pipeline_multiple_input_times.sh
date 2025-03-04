#!/bin/bash

########################## ONLY CHANGE INFO HERE ###################################
IFO="K1"
# Define array of times to process
TIMES=(1368986010 1368987010 1368988010)  # Add your desired times here
DURATION=900
LOW_FREQ=175.0
HIGH_FREQ=185.0

# Paths to Python scripts (adjust as needed)
RUN_COHERENCE_SCRIPT="./run_coherence.py"
SELECT_WITNESSES_SCRIPT="./select_witnesses.py"

####################################################################################

# Initialize Conda environment
# Source the Conda setup script (adjust path based on your system)
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate igwn-py38 || {
    echo "Error: Failed to activate conda environment 'igwn-py38'."
    exit 1
}

# Check if Python scripts exist
for script in "$RUN_COHERENCE_SCRIPT" "$SELECT_WITNESSES_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: Script '$script' not found."
        exit 1
    fi
done

# Process each time in the array
for TIME in "${TIMES[@]}"; do
    OUTTIME=$((TIME + DURATION))
    echo "Processing TIME=${TIME} with frequency range ${LOW_FREQ}-${HIGH_FREQ} Hz..."

    # Run coherence calculation
    echo "Running run_coherence.py for TIME=${TIME}..."
    python "$RUN_COHERENCE_SCRIPT" \
        --savedir "${IFO}_automatic" \
        --time "$TIME" \
        --dur "$DURATION" \
        --ifo "$IFO" \
        --lowfreq "$LOW_FREQ" \
        --highfreq "$HIGH_FREQ" || {
        echo "Error: run_coherence.py failed for TIME=${TIME}."
        continue  # Skip to next time if this one fails
    }

    # Run witness selection
    echo "Running select_witnesses.py for TIME=${TIME}..."
    python "$SELECT_WITNESSES_SCRIPT" \
        --ifo "$IFO" \
        --time "$TIME" \
        --savedir "${IFO}_automatic/${TIME}/" \
        --lowfreq "$LOW_FREQ" \
        --highfreq "$HIGH_FREQ" || {
        echo "Error: select_witnesses.py failed for TIME=${TIME}."
        continue
    }

    echo "Processing completed for TIME=${TIME}."
done

echo "All times have been processed."