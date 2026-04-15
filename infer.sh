PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path /yke/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /yke/models/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors 0.8 \
    --lora /data/ckpt_zsqiao/sft_lora_ltx2.3_keyframe_0415/checkpoints/lora_weights_step_01000.safetensors \
    --spatial-upsampler-path /yke/models/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root /yke/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "A fair-skinned young woman with blonde hair styled in low pigtails secured with small clear butterfly clips, wearing a pale lavender strapless mini dress printed with tiny delicate purple floral patterns, stands next to a dark wooden park bench in a sun-dappled green park with lush trees and bushes under soft warm daylight. Her gaze stays fixed directly on the camera throughout the shot, eyes soft and warm, with one natural slow blink occurring mid-wave. Her initial soft closed-mouth smile grows slightly wider as she greets, lips parting just enough to reveal a glimpse of her upper teeth, faint crow's feet crinkling at the outer corners of her eyes, subtle dimples forming on both cheeks, her smooth skin has a faint dewy sunlit glow on her cheekbones. Her posture stays balanced, weight supported on her right leg, while her left leg remains lifted at the calf, her right hand stays steady holding her white slip-on shoe against the sole of her lifted left foot. She smoothly lifts her left hand from the hem of her dress up to shoulder height, palm facing the camera, fingers slightly loose as she gives a gentle, playful casual wave to the camera, her upper body sways very slightly with the playful motion, the fabric of her dress shifts softly with the movement. The shot remains a steady medium shot focused on her, with no scene transitions." \
    --image /data/ckpt_zsqiao/codes/omni_ref_v0.1/multi_view_output/000099/input_img/first_frame.png 0 1.0 \
    --image /data/ckpt_zsqiao/codes/omni_ref_v0.1/multi_view_output/000099/human_id/guai.png 144 1.0 \
    --output-path ./ref0.mp4 \
    --num-inference-steps 50 \
    --height 1536 --width 1024 \
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