cd /datasets/codes_zsqiao/LTX-2/packages/ltx-trainer

python3 /datasets/codes_zsqiao/LTX-2/packages/ltx-trainer/scripts/process_dataset.py \
  /datasets/data_zsqiao/human_ref_v0.2/metadata.csv \
  --resolution-buckets "704x1280x121;1280x704x121;576x1024x121;1024x576x121;704x960x121;544x960x121;960x544x121;704x1440x121;1440x704x121" \
  --model-path /models/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /models/gemma-3-12b-it-qat-q4_0-unquantized \
  --output-dir /datasets/codes_zsqiao/LTX-2/preprocessed \
  --video-column video \
  --caption-column caption \
  --keyframe-column keyframes \
  --batch-size 1 \
  --no-with-audio