#!/bin/bash

########################## ONLY CHANGE INFO here ###################################
IFO=K1
TIME=1378402218
OUTTIME=1378403243
LOW_FREQ=10.0
HIGH_FREQ=15.0

####################################################################################

cd coherence-monitor
python run_coherence.py --savedir ${IFO}_automatic --time ${TIME} --dur 1024 --ifo ${IFO}

# python select_witnesses.py --ifo ${IFO} --savedir ${IFO}_automatic/${TIME}/ --lowfreq ${LOW_FREQ} --highfreq ${HIGH_FREQ}
# mv chanlist_o4.ini ../chanlist_o4.ini
# cd -

# source /home/muhammed.saleem/anaconda3/etc/profile.d/conda.sh
# conda activate dc-prod-py36

# dc-prod-train  --save-dataset True --load-dataset False --fs 1024 --chanslist ./chanlist_o4.ini --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl ${LOW_FREQ} --filt-fh ${HIGH_FREQ} --filt-order 8 --device cuda --train-frac 0.9 --batch-size 32 --max-epochs 40 --num-workers 4 --lr ${LR} --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir outdir_o4_${IFO}_${LOW_FREQ}_${HIGH_FREQ} --train-t0 ${TIME} --train-duration 1024

# dc-prod-clean --save-dataset True --fs 1024 --out-dir outdir_o4_${IFO}_${LOW_FREQ}_${HIGH_FREQ}/ --out-channel ${IFO}:GDS-CALIB_STRAIN_DC --chanslist ./chanlist_o4.ini --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cuda --train-dir outdir_o4_${IFO}_${LOW_FREQ}_${HIGH_FREQ}/ --out-file H-${IFO}_HOFT_DC-${OUTTIME}-4096.gwf --clean-t0 ${OUTTIME} --clean-duration 4096

# python plotting.py --time ${OUTTIME} --ifo ${IFO} --savedir outdir_o4_${IFO}_${LOW_FREQ}_${HIGH_FREQ}/ --lowfreq ${LOW_FREQ} --highfreq ${HIGH_FREQ}

# mkdir ${IFO}_${TIME}_${LOW_FREQ}Hz_${HIGH_FREQ}Hz
# mv *.png ${IFO}_${TIME}_${LOW_FREQ}Hz_${HIGH_FREQ}Hz/.
# mv chanlist_o4.ini ${IFO}_${TIME}_${LOW_FREQ}Hz_${HIGH_FREQ}Hz/.
# cp coherence-monitor/plots/channels_coh_${TIME}_*.png.html ${IFO}_${TIME}_${LOW_FREQ}Hz_${HIGH_FREQ}Hz/.