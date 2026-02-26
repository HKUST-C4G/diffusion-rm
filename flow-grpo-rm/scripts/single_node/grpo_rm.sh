accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_sd3_fast_rm.py --config config/grpo.py:pickscore_sd3_fast_rm

