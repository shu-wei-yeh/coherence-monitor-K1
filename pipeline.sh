#!/bin/bash

########################## ONLY CHANGE INFO HERE ###################################
IFO="K1"
TIME=1368986010
DURATION=900
OUTTIME=$((TIME + DURATION))
LOW_FREQ=175.0
HIGH_FREQ=185.0

# Paths to Python scripts (adjust as needed)
RUN_COHERENCE_SCRIPT="./run_coherence.py"
SELECT_WITNESSES_SCRIPT="./select_witnesses.py"

####################################################################################

# Initialize Conda environment
# Source the Conda setup script (adjust path based on your system)
# CONDA_BASE=$(conda info --base)
# source "${CONDA_BASE}/etc/profile.d/conda.sh"
# conda activate igwn-py38 || {
#     echo "Error: Failed to activate conda environment 'igwn-py38'."
#     exit 1
# }

# Check if Python scripts exist
for script in "$RUN_COHERENCE_SCRIPT" "$SELECT_WITNESSES_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: Script '$script' not found."
        exit 1
    fi
done

# Run coherence calculation
echo "Running run_coherence.py for TIME=${TIME}..."
python "$RUN_COHERENCE_SCRIPT" \
    --savedir "${IFO}_automatic" \
    --time "$TIME" \
    --dur "$DURATION" \
    --ifo "$IFO" || {
    echo "Error: run_coherence.py failed."
    exit 1
}

# Run witness selection
echo "Running select_witnesses.py for TIME=${TIME}..."
python "$SELECT_WITNESSES_SCRIPT" \
    --ifo "$IFO" \
    --time "$TIME" \
    --savedir "${IFO}_automatic/${TIME}/" \
    --lowfreq "$LOW_FREQ" \
    --highfreq "$HIGH_FREQ" || {
    echo "Error: select_witnesses.py failed."
    exit 1
}

echo "Processing completed for TIME=${TIME}."