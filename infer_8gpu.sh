#!/bin/bash
# cd /datasets/codes_zsqiao/LTX-2

# single-stage
python /datasets/codes_zsqiao/LTX-2/infer_8gpu.py \
    --mode single \
    --csv /datasets/codes_zsqiao/LTX-2/example_infer.csv \
    --num-gpus 8 \
    --output-dir ./ltx_outputs \
    --checkpoint-path /models/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --lora /datasets/temp_zsqiao_annotate/lora_weights_step_01800.safetensors \
    --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized \
    --num-inference-steps 50 \
    --height 1088 --width 1920 \
    --frame-rate 24 --num-frames 121 \
    --seed 42

# 步数蒸馏
# python infer_8gpu.py \
#     --mode distilled \
#     --csv /datasets/codes_zsqiao/LTX-2/example_infer.csv \
#     --num-gpus 8 \
#     --output-dir ./ltx_outputs_distilled \
#     --distilled-checkpoint-path /models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
#     --spatial-upsampler-path /models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
#     --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized \
#     --height 1024 --width 1536 \
#     --frame-rate 24 --num-frames 121 \
#     --seed 42