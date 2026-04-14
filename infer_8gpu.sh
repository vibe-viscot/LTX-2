#!/bin/bash
cd /data/ckpt_zsqiao/codes/LTX-2

python infer_8gpu.py \
    --csv /data/ckpt_zsqiao/codes/LTX-2/example_infer.csv \
    --num-gpus 8 \
    --output-dir ./ltx_outputs \
    --checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /yke/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors \
    --distilled-lora-strength 0.8 \
    --lora /data/ckpt_zsqiao/sft_lora_ltx2.3_keyframe/checkpoints/lora_weights_step_02000.safetensors \
    --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --num-inference-steps 50 \
    --height 1536 --width 1024 \
    --frame-rate 24 \
    --num-frames 121 \
    --seed 42
