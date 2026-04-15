#!/bin/bash
cd /data/ckpt_zsqiao/codes/LTX-2

# two-stages
python infer_8gpu.py \
    --mode two_stages \
    --csv /data/ckpt_zsqiao/codes/LTX-2/example_infer.csv \
    --num-gpus 8 \
    --output-dir ./ltx_outputs \
    --checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /yke/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
    --distilled-lora-strength 0.8 \
    --lora /data/ckpt_zsqiao/sft_lora_ltx2.3_keyframe_0415/checkpoints/lora_weights_step_01000.safetensors \
    --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --num-inference-steps 50 \
    --height 1536 --width 1024 \
    --frame-rate 24 --num-frames 121 \
    --seed 42

# 步数蒸馏
# python infer_8gpu.py \
#     --mode distilled \
#     --csv /data/ckpt_zsqiao/codes/LTX-2/example_infer.csv \
#     --num-gpus 8 \
#     --output-dir ./ltx_outputs_distilled \
#     --distilled-checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
#     --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
#     --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
#     --height 1024 --width 1536 \
#     --frame-rate 24 --num-frames 121 \
#     --seed 42