import io, os, argparse
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import queue, threading
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline
from diffusion_rm.utils.vae_utils import VAEProcessor
from diffusion_rm.data.bucket_manager import BucketManager


BUCKET_MANAGER = BucketManager(ar_thresh=0.03, divisible=32, debug=False)

# --- 分布式与数据加载 ---
def setup_ddp():
    """初始化 DDP，默认强制使用 NCCL 后端及 GPU。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"| Distributed init: rank {rank}, world_size {world_size}", flush=True)

    return local_rank, rank, world_size

def load_dataset_safe(path, rank, local_rank):
    """安全的分布式数据集加载，避免多进程同时下载导致锁冲突。"""
    if rank == 0:
        print("Master Rank 0 is preparing the dataset...")
        load_dataset(path)
    if dist.is_initialized():
        dist.barrier()
    
    if local_rank == 0 and rank != 0:
        load_dataset(path)
    if dist.is_initialized():
        dist.barrier()
        
    return load_dataset(path)

# --- 核心图像处理逻辑 ---
def _find_shared_bucket_for_pair(h1, w1, h2, w2, aspect_ratio_threshold=0.05):
    """计算两张图像对齐后的共同 Bucket 尺寸 (H, W)。"""
    ar1, ar2 = w1 / float(h1), w2 / float(h2)
    ar_diff = abs(ar1 - ar2) / max(ar1, ar2)

    if ar_diff >= aspect_ratio_threshold:
        return None

    max_edge = max(w1, h1, w2, h2)

    def _get_scaled_edge(w, h, max_e):
        if max(w, h) >= max_e: return w, h
        return (max_e, int(h * max_e / w)) if w > h else (int(w * max_e / h), max_e)

    w1r, h1r = _get_scaled_edge(w1, h1, max_edge)
    w2r, h2r = _get_scaled_edge(w2, h2, max_edge)

    return BUCKET_MANAGER.assign_bucket((min(h1r, h2r), min(w1r, w2r)), return_res=True)

def ndarray_to_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

# --- 数据集与 DataLoader ---
class ImagePairDataset(Dataset):
    def __init__(self, samples, process_fn):
        self.samples = samples
        self.process_fn = process_fn
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            return self.process_fn(self.samples[idx])
        except Exception as e:
            print(f"Error loading sample idx={idx}: {e}")
            return None

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    out = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack([v.contiguous() for v in vals], dim=0)
        elif isinstance(vals[0], (bytes, bytearray, memoryview, str)):
            out[key] = vals
        else:
            out[key] = default_collate(vals)
    return out

# --- 异步 I/O ---
class AsyncParquetWriter:
    def __init__(self, out_path: str, required_fields=None):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.out_path = out_path
        self.buffer_size = 100  # 固定为合理默认值
        self.buffer = []
        self.required_fields = required_fields
        
        # 在初始化时一次性构建 Schema，消除运行时开销
        self._schema = self._build_schema()
        self._writer = pq.ParquetWriter(self.out_path, self._schema, compression="zstd")
        
        self._write_queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._write_thread = threading.Thread(target=self._writer_loop)
        self._write_thread.start()
    
    def _build_schema(self):
        fields = [
            pa.field("prompt", pa.string()),
            pa.field("model1", pa.string()),
            pa.field("model2", pa.string()),
            pa.field("detailed_results", pa.string()),
            pa.field("image1_path", pa.string()),
            pa.field("image2_path", pa.string()),
            pa.field("latent1", pa.binary()),
            pa.field("latent2", pa.binary()),
            pa.field("latent1_shape", pa.list_(pa.int64())),
            pa.field("latent2_shape", pa.list_(pa.int64())),
        ]
        if self.required_fields:
            fields = [f for f in fields if f.name in self.required_fields]
        return pa.schema(fields)
    
    def _writer_loop(self):
        while not self._stop_event.is_set() or not self._write_queue.empty():
            try:
                records = self._write_queue.get(timeout=0.1)
                if records is None: break
                
                data_dict = {name: [r.get(name) for r in records] for name in self._schema.names}
                self._writer.write_table(pa.Table.from_pydict(data_dict, schema=self._schema))
            except queue.Empty:
                continue
    
    def add_records(self, records):
        self.buffer.extend(records)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        if self.buffer:
            self._write_queue.put(self.buffer.copy())
            self.buffer.clear()
    
    def close(self):
        self.flush()
        self._write_queue.put(None)
        self._write_thread.join()
        if self._writer is not None:
            self._writer.close()

# --- 核心推理管线 ---
@torch.no_grad()
def process_split_with_dataloader(samples, vae_processor, device, batch_size, process_fn, num_workers):
    dataset = ImagePairDataset(samples, process_fn)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=3,  # 硬编码最优值
        pin_memory=True,    # 强制固定内存，加速 GPU 传输
        persistent_workers=(num_workers > 0),
        collate_fn=collate_skip_none,
    )

    for batch in dataloader:
        if not batch: continue
        
        tensors1 = batch['tensor1'].to(device, non_blocking=True)
        tensors2 = batch['tensor2'].to(device, non_blocking=True)
        
        lat1 = vae_processor.encode(tensors1)
        lat2 = vae_processor.encode(tensors2)
        
        # 强制在 GPU 上完成 dtype 转换，再异步复制到主存
        lat1_np = lat1.to(torch.float32).cpu().numpy()
        lat2_np = lat2.to(torch.float32).cpu().numpy()
        
        batch_records = []
        for j in range(len(batch['prompt'])):
            record = {k: batch[k][j] for k in batch if k not in ['tensor1', 'tensor2']}
            record.update({
                "latent1": ndarray_to_npy_bytes(lat1_np[j]),
                "latent2": ndarray_to_npy_bytes(lat2_np[j]),
                "latent1_shape": list(lat1_np[j].shape),
                "latent2_shape": list(lat2_np[j].shape),
            })
            batch_records.append(record)
        
        yield batch_records

# --- 业务处理逻辑 ---
def process_fn_hpdv3(sample, meta_dir=None, align_images=True):
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    base_path1 = os.path.join(meta_dir, sample["path1"]) if meta_dir else sample["path1"]
    base_path2 = os.path.join(meta_dir, sample["path2"]) if meta_dir else sample["path2"]
    
    img1 = Image.open(base_path1).convert("RGB")
    img2 = Image.open(base_path2).convert("RGB")

    w1, h1 = img1.size
    w2, h2 = img2.size
    if w1 * h1 < 65536 or w2 * h2 < 65536:  # 256 * 256
        return None

    bucket_1 = BUCKET_MANAGER.assign_bucket((h1, w1), return_res=True) 
    bucket_2 = BUCKET_MANAGER.assign_bucket((h2, w2), return_res=True)
    if not bucket_1 or not bucket_2: return None

    if align_images and bucket_1 != bucket_2:
        shared_size = _find_shared_bucket_for_pair(bucket_1[0], bucket_1[1], bucket_2[0], bucket_2[1])
        if not shared_size: return None
        img1_size = img2_size = shared_size
    else:
        img1_size, img2_size = bucket_1, bucket_2

    img1 = img1.resize((img1_size[1], img1_size[0]), Image.LANCZOS)
    img2 = img2.resize((img2_size[1], img2_size[0]), Image.LANCZOS)

    detailed_results = {
        "votes_chosen": int(sample['choice_dist'][0] or 0),
        "votes_rejected": int(sample['choice_dist'][1] or 0),
        "confidence": float(sample['confidence'] or 0.0),
        "orig_chosen_hw": [h1, w1], "orig_rejected_hw": [h2, w2],
        "final_chosen_hw": [img1_size[0], img1_size[1]],
        "final_rejected_hw": [img2_size[0], img2_size[1]],
    }
    
    return {
        'tensor1': transform_fn(img1),
        'tensor2': transform_fn(img2),
        'prompt': sample['prompt'],
        'model1': sample['model1'],
        'model2': sample['model2'],
        'detailed_results': str(detailed_results),
        'image1_path': sample["path1"],
        'image2_path': sample['path2'],
    }

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Distributed VAE Latent Extraction Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the huggingface dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for Parquet files.")
    parser.add_argument("--meta_dir", type=str, default=None, help="Base directory for image paths.")
    parser.add_argument("--dataset_name", type=str, default="hpsv3")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader I/O workers.")
    args = parser.parse_args()

    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    print(f"[R{rank}] Initialized on Device: {device}")

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    # 模型加载
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium").to(device)
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    # pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16, )
    # pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,)
    pipe.vae.eval()
    vae_proc = VAEProcessor(pipe.vae)
    # vae_proc = VAEProcessor_QwenImage(pipe.vae)
    
    if rank == 0:
        print(f"[R{rank}] VAE Shift: {vae_proc.vae_config_shift_factor} | Scale: {vae_proc.vae_config_scaling_factor}")

    # 路由与参数绑定
    if args.dataset_name != 'hpsv3':
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}. Only 'hpsv3' is implemented.")
    
    required_fields = ['prompt', 'model1', 'model2', 'detailed_results', 'image1_path', 'image2_path', 'latent1', 'latent2', 'latent1_shape', 'latent2_shape']
    bound_process_fn = partial(process_fn_hpdv3, meta_dir=args.meta_dir, align_images=True)

    # 数据集加载与分配
    ds = load_dataset_safe(args.dataset_path, rank, local_rank)
    split_names = [args.split] if args.split in ds else ds.keys()
    
    out_path = os.path.join(args.out_dir, f"part_rank{rank:02d}.parquet")
    if os.path.exists(out_path):
        raise FileExistsError(f"Output file already exists: {out_path}")
        
    writer = AsyncParquetWriter(out_path, required_fields=required_fields)
    
    total_processed = 0
    for split_name in split_names:
        split_data = ds[split_name]
        idxs = [i for i in range(len(split_data)) if (i % world_size) == rank]
        
        try:
            rank_samples = split_data.select(idxs)
        except AttributeError:
            rank_samples = [split_data[i] for i in idxs]
            
        if rank == 0:
            print(f"[R{rank}] Processing {split_name}: {len(rank_samples)} samples assigned.")
            
        pbar = tqdm(
            process_split_with_dataloader(
                samples=rank_samples,
                vae_processor=vae_proc,
                device=device,
                batch_size=args.batch_size,
                process_fn=bound_process_fn,
                num_workers=args.num_workers,
            ),
            desc=f"[R{rank}] {split_name}",
            total=(len(rank_samples) + args.batch_size - 1) // args.batch_size,
            disable=(rank != 0),
        )
        
        for batch_records in pbar:
            writer.add_records(batch_records)
            total_processed += len(batch_records)
            pbar.set_postfix(total=total_processed)

    writer.close()
    if rank == 0:
        print(f"[R{rank}] Pipeline execution finished. Total samples processed globally by this rank: {total_processed}")

if __name__ == "__main__":
    main()