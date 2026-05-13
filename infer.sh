export PYTHONPATH=/datasets/codes_zsqiao/LTX-2/packages/ltx-core/src:/datasets/codes_zsqiao/LTX-2/packages/ltx-pipelines/src:/datasets/codes_zsqiao/LTX-2/packages/ltx-trainer/src

export CUDA_VISIBLE_DEVICES=2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_one_stage \
    --checkpoint-path /models/Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --lora /datasets/temp_zsqiao_annotate/lora_weights_step_00400.safetensors \
    --gemma-root /models/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "A young woman with curly hair, dressed in a black blazer and a blue polka-dot shirt, is speaking while holding a presentation remote in her hand." \
    --image /datasets/wqf_codes/video_generation/e2v/bench_v6_train_passed/jm_MjyIlAQ0.7_segment_0_228-Scene-001_noFacePose_0_196/first_frame.jpg 0 1.0 \
    --image /datasets/wqf_codes/video_generation/e2v/bench_v6_train_passed/jm_MjyIlAQ0.7_segment_0_228-Scene-001_noFacePose_0_196/ref_cross_top3/frame_00000_yp13.9_pn07.2_rp03.4.jpg 144 1.0 \
    --image /datasets/wqf_codes/video_generation/e2v/bench_v6_train_passed/jm_MjyIlAQ0.7_segment_0_228-Scene-001_noFacePose_0_196/ref_cross_top3/frame_00024_yn30.9_pp09.0_rp09.5.jpg 168 1.0 \
    --image /datasets/wqf_codes/video_generation/e2v/bench_v6_train_passed/jm_MjyIlAQ0.7_segment_0_228-Scene-001_noFacePose_0_196/ref_cross_top3/frame_00170_yp29.1_pn07.9_rp05.7.jpg 192 1.0 \
    --output-path ./ref400.mp4 \
    --num-inference-steps 50 \
    --height 704 --width 1280 \
    --frame-rate 24 \
    --num-frames 121 \
    --seed 42


# 步数蒸馏

# PROMPT_TEXT="The camera remains stationary. The figure in the frame is smiling warmly towards the camera and naturally waves to greet, as if seeing a long-lost friend."

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.distilled \
#     --distilled-checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
#     --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
#     --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
#     --prompt "$PROMPT_TEXT" \
#     --image /data/ckpt_zsqiao/codes/dataset_cases/000494/input_img/first_frame.png 0 1.0 \
#     --output-path ./000494.mp4 \
#     --height 1536 --width 1024 \
#     --frame-rate 24 --num-frames 121 \
#     --seed 42