# export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
# export WANDB_MODE="offline"
# export NCCL_P2P_DISABLE=1
# export TORCH_NCCL_ENABLE_MONITORING=0

# Get GPU IDs from command line argument, default to "2,3" if not provided
GPU_IDS=${1:-"2,3"}

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4" "1e-3")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw" "adam")
MAX_TRAIN_STEPS=("3000")

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_2.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="/workspace/video-dataset"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="/workspace/output_models/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_image_to_video_lora.py \
          --pretrained_model_name_or_path THUDM/CogVideoX-5b-I2V \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --id_token BW_STYLE \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers 8 \
          --pin_memory \
          --validation_prompt \"BW_STYLE The video depicts a scene set in a classroom, where a group of students is seated at desks, attentively facing the front of the room. The classroom is characterized by its traditional setup, featuring a blackboard at the front and a window that allows natural light to illuminate the space. The students are dressed in uniforms, indicating a formal educational setting. Among them, one student stands out by raising their hand, suggesting an attempt to answer a question or participate in the lesson. This action is captured in a moment of anticipation, with the raised hand held high and the student's expression reflecting eagerness or excitement.:::BW_STYLE The video features a young boy with short hair, wearing a collared shirt, who appears to be in a state of distress. He is shown in a classroom setting, with a desk and a window in the background. The boy's facial expression changes throughout the video, with his mouth opening wider and his eyes squinting, suggesting a range of emotions or reactions. The video maintains a consistent monochromatic color scheme, with the boy's clothing and the classroom environment rendered in shades of gray. The camera angle remains static, focusing on the boy's upper body and face.\" \
          --validation_images \"/workspace/validation/image1.png:::/workspace/validation/image2.png\"
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs 10 \
          --seed 42 \
          --rank 128 \
          --lora_alpha 128 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps 1000 \
          --gradient_accumulation_steps 1 \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 400 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --noised_image_dropout 0.05 \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --nccl_timeout 1800"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done