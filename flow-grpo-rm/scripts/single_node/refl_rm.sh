accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29500 scripts/train_sd3_refl_rm.py --config config/refl.py:pickscore_sd3_refl_rm_8gpu
