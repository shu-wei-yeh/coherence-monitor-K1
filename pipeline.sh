#!/bin/bash

########################## ONLY CHANGE INFO HERE ###################################
IFO="K1"
TIME=1370928898
DURATION=900
OUTTIME=$((TIME + DURATION))

# Define multiple frequency ranges as an array of "low high" pairs
FREQUENCY_RANGES=(
    # "30 40"
    # "55 65"    
    # "115 125"  
    # "175 185" 
    # "100 200"
    "200 300"
    "300 400"
    "" 
)

# Paths to Python scripts (adjust as needed)
RUN_COHERENCE_SCRIPT="./run_coherence.py"
SELECT_WITNESSES_SCRIPT="./select_witnesses.py"

####################################################################################

# Check if Python scripts exist
for script in "$RUN_COHERENCE_SCRIPT" "$SELECT_WITNESSES_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "Error: Script '$script' not found."
        exit 1
    fi
done

# Run coherence calculation (only once)
echo "Running run_coherence.py for TIME=${TIME}..."
python "$RUN_COHERENCE_SCRIPT" \
    --savedir "${IFO}_automatic" \
    --time "$TIME" \
    --dur "$DURATION" \
    --ifo "$IFO" || {
    echo "Error: run_coherence.py failed."
    exit 1
}

# Process each frequency range
for range in "${FREQUENCY_RANGES[@]}"; do
    # Split the range into LOW_FREQ and HIGH_FREQ
    read -r LOW_FREQ HIGH_FREQ <<< "$range"

    # Validate frequency range
    if (( $(echo "$LOW_FREQ >= $HIGH_FREQ" | bc -l) )); then
        echo "Error: LOW_FREQ ($LOW_FREQ) must be less than HIGH_FREQ ($HIGH_FREQ) for range '$range'."
        continue
    fi

    # Create a unique suffix for output files
    FREQ_SUFFIX="${LOW_FREQ}Hz-${HIGH_FREQ}Hz"

    echo "Running select_witnesses.py for TIME=${TIME} with frequency range ${LOW_FREQ}-${HIGH_FREQ} Hz..."
    python "$SELECT_WITNESSES_SCRIPT" \
        --ifo "$IFO" \
        --time "$TIME" \
        --savedir "${IFO}_automatic/${TIME}/" \
        --dur "$DURATION" \
        --lowfreq "$LOW_FREQ" \
        --highfreq "$HIGH_FREQ" || {
        echo "Error: select_witnesses.py failed for range ${LOW_FREQ}-${HIGH_FREQ} Hz."
        continue
    }

    # Rename output files to include frequency range (optional, if script doesn't support natively)
    for ext in "csv" "txt"; do
        orig_file="${IFO}_automatic/${TIME}/max_corr_output_${TIME}.${ext}"
        new_file="${IFO}_automatic/${TIME}/max_corr_output_${TIME}_${FREQ_SUFFIX}.${ext}"
        if [[ -f "$orig_file" ]]; then
            mv "$orig_file" "$new_file"
            echo "Renamed $orig_file to $new_file"
        fi
    done

    echo "Completed processing for frequency range ${LOW_FREQ}-${HIGH_FREQ} Hz."
done

echo "Processing completed for TIME=${TIME} across all frequency ranges."