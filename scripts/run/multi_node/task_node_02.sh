
#######################################
### Your multi node configuration

export OMP_NUM_THREADS=8
export NCCL_IB_TIMEOUT=50
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,INIT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0.3306
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_HCA="^mlx5_bond_0,mlx5_bond_1"
export TORCH_CUDA_ARCH_LIST="9.0"
export GLOO_SOCKET_IFNAME=bond0.3306
export GLOO_SOCKET_TIMEOUT=3600
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=2097152

#######################################


export HF_TOKEN='YOUR_HF_TOKEN'

wandb login "YOUR_WANDB_TOKEN"

NNODES=2
NODE_RANK=1
MASTER_ADDR="192.168.1.1"
MASTER_PORT=29500
TOTAL_PROCESSES=$((NNODES * 8))
CONFIG_PATH="config/sd3m/thurstone-12layer.yaml"

accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_machines=$NNODES \
  --machine_rank=$NODE_RANK \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  --num_processes=$TOTAL_PROCESSES \
  -m scripts.train_sd3 \
  --config $CONFIG_PATH

# sh tools/gpu_occupy/start.sh
  