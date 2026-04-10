cd /data/ckpt_zsqiao/codes/LTX-2/packages/ltx-trainer

python3 scripts/decode_keyframes.py \
    --model-path "/yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors" \
    --keyframes-dir "/data/ckpt_zsqiao/codes/LTX-2/preprocessed/keyframes" \
    --output-dir "/data/ckpt_zsqiao/codes/LTX-2/preprocessed/decoded_keyframes"