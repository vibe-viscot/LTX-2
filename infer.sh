PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /yke/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --lora /data/ckpt_zsqiao/sft_lora_ltx2.3_keyframe2v/checkpoints/lora_weights_step_02000.safetensors \
    --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "The camera remained stationary. The figure in the frame was smiling warmly towards the camera and waved his hand in greeting, as if he were seeing a long-lost friend." \
    --image /data/ckpt_zsqiao/codes/omni_ref_v0.1/multi_view_output/000000/input_img/first_frame.png 0 1.0 \
    --image /data/ckpt_zsqiao/codes/omni_ref_v0.1/multi_view_output/000000/human_id/frontal_image.png 144 1.0 \
    --image /data/ckpt_zsqiao/codes/omni_ref_v0.1/multi_view_output/000000/human_id/refer_images_1.png 168 1.0 \
    --image /data/ckpt_zsqiao/codes/omni_ref_v0.1/multi_view_output/000000/human_id/refer_images_2.png 192 1.0 \
    --output-path ./ref0.mp4 \
    --num-inference-steps 50 \
    --height 1536 --width 1024 \
    --frame-rate 24 \
    --num-frames 121 \
    --seed 42


# 步数蒸馏

# PROMPT_TEXT="The camera remains completely static and locked-off throughout the entire video. A young boy with short dark hair, wearing a grey geometric-patterned puffer jacket and a blue backpack, stands under a large red and white umbrella on a rainy sidewalk. His overall disposition is cheerful and friendly, maintaining a stationary torso throughout the shot. The camera captures him in a medium shot, centered in the frame against a background of parked scooters and fallen leaves. He keeps his right hand steady and motionless on the black umbrella handle, looking directly into the lens with a warm smile, while he raises his left hand to give a cheerful and natural wave to the camera. The sound of light rain tapping against the taut fabric of the umbrella provides a constant auditory backdrop."

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.distilled \
#     --distilled-checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
#     --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
#     --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
#     --prompt "$PROMPT_TEXT" \
#     --image /data/ckpt_zsqiao/codes/test_i2v/test_shuangren/6c2422d706be0ae92739f9f950131a5d.jpg 0 1.0 \
#     --output-path ./2-distill.mp4 \
#     --height 1280 --width 768 \
#     --seed 42