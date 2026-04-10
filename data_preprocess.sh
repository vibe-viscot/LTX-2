cd /data/ckpt_zsqiao/codes/LTX-2/packages/ltx-trainer

python3 /data/ckpt_zsqiao/codes/LTX-2/packages/ltx-trainer/scripts/process_dataset.py \
  /data/ckpt_zsqiao/codes/omni_ref_v0.1/train/metadata.csv \
  --resolution-buckets "544x960x121;960x544x121;768x768x121" \
  --model-path /yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /data/ckpt_zsqiao/codes/LTX-2/preprocessed \
  --video-column video \
  --caption-column caption \
  --keyframe-column keyframes \
  --batch-size 1 \
  --no-with-audio