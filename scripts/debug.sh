export CUDA_VISIBLE_DEVICES=0

export HF_TOKEN='YOUR_HF_TOKEN'

wandb login "YOUR_WANDB_TOKEN"

accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines=1 \
  --num_processes=1 \
  -m scripts.train_sd3 \
  --config config/sd3m/thurstone-12layer.yaml

  
  