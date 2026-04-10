export WANDB_API_KEY=""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes 8 --config_file /data/ckpt_zsqiao/codes/LTX-2/packages/ltx-trainer/configs/accelerate/ddp.yaml \
  /data/ckpt_zsqiao/codes/LTX-2/packages/ltx-trainer/scripts/train.py \
  /data/ckpt_zsqiao/codes/lllltxt/LTX-2/packages/ltx-trainer/configs/ltx2_keyframe_lora.yaml