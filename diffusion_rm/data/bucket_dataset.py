import io, os, json, argparse
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict, deque
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
import io
import glob

from diffusion_rm.data.bucket_manager import BucketManager

def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


class BucketDataset(IterableDataset):
    def __init__(
        self,
        parquet_files: List[str],
        bucket_manager: BucketManager,
    ):
        super().__init__()
        self.parquet_files = parquet_files
        self.bucket_manager = bucket_manager

        print(f"初始化流式数据集，共 {len(parquet_files)} 个parquet文件")


        self.total_samples = len(self.bucket_manager.res_map)

        # load metadata from parquet files
        self.parquet_files = {
            os.path.basename(f): f for f in parquet_files
        }


    def _process_sample(self, row: Dict) -> Dict:
        """处理单个样本"""
        # 解码latent
        latent1 = npy_bytes_to_ndarray(row['latent1'])
        latent2 = npy_bytes_to_ndarray(row['latent2'])
        
        # 判断chosen/reject
        chosen = latent1
        reject = latent2
        
        return {
            'prompt': row['prompt'],
            'latent_chosen': torch.from_numpy(chosen),
            'latent_reject': torch.from_numpy(reject),
        }
    
    def __len__(self):
        # batch_total 是每个进程的 batch 数量
        return self.bucket_manager.batch_total * self.bucket_manager.world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        self.bucket_manager.set_worker_info(
            worker_id=worker_id,
            num_workers=num_workers,
        )

        for batch_ids, resolution, sample_infos in self.bucket_manager.generator():
            batch_data = []
            target_bsz = len(sample_infos)
            for sample_info in sample_infos:
                try:
                    parquet_file = os.path.basename(sample_info['parquet_path'])
                    row_group = sample_info['row_group']
                    row_id = sample_info['row_index']
                    pf = pq.ParquetFile(self.parquet_files[parquet_file])

                    row = pf.read_row_group(row_group, columns=['prompt', 'latent1', 'latent2'])
                    # batch_data.append(self._process_sample(row))
                    data = self._process_sample({
                        'prompt': row.column('prompt')[row_id].as_py(),
                        'latent1': row.column('latent1')[row_id].as_py(),
                        'latent2': row.column('latent2')[row_id].as_py(),
                    })
                    batch_data.append(data)
                except Exception as e:
                    print(f"Error processing sample {sample_info['global_id']}: {e}")
            yield batch_data


def collate_fn(batch_list: List[List[Dict]]) -> Dict:
    if len(batch_list) != 1:
        raise ValueError(f"Expected batch_list length 1, got {len(batch_list)}")

    batch = batch_list[0]

    if not batch:
        return {}
    collated = {
        'prompt': [item['prompt'] for item in batch],
        'latent_chosen': torch.stack([item['latent_chosen'] for item in batch]),
        'latent_reject': torch.stack([item['latent_reject'] for item in batch]),
    }

    return collated


class BucketDataLoader:
    def __init__(
            self,
            parquet_files: List[str],
            bucket_file: str,
            base_resolution=(1024, 1024),
            bsz=3,
            world_size=1,
            global_rank=0,
            shuffle: bool = True,
            seed: int = 42,
            num_workers: int = 0,
            use_dynamic_bsz: bool = True,
            ):
        
        self.bucket_manager = BucketManager(
            bucket_file=bucket_file,
            divisible=32,
            ar_thresh=0.03,
            base_resolution=base_resolution,
            bsz=bsz,
            world_size=world_size,
            global_rank=global_rank,
            seed=seed,
            use_dynamic_bsz=use_dynamic_bsz,
            debug=False,
        )

        print(f"Total samples loaded in BucketManager: {len(self.bucket_manager.res_map)}")
        print(f"Total buckets: {len(self.bucket_manager.buckets)}")

        self.dataset = BucketDataset(
            parquet_files=parquet_files,
            bucket_manager=self.bucket_manager,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # 因为已经在Dataset中按batch划分好了
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        self.epoch_count = 0

    def __iter__(self):
        """迭代器"""
        return iter(self.dataloader)
    
    def __len__(self):
        """返回 batch 数量"""
        return len(self.dataset)
    
    def start_epoch(self):
        """开始新的 epoch"""
        self.epoch_count += 1
        self.bucket_manager.start_epoch()


def create_bucket_dataloader(
    parquet_files: Union[str, List[str]],      # str or list of parquet file paths
    bucket_file: str,
    reference_size=1024,
    base_batch_size=3,
    world_size=1,
    global_rank=0,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    use_dynamic_bsz: bool = True,
) -> BucketDataLoader:
    """创建 BucketDataLoader"""
    if isinstance(parquet_files, str):
        parquet_files = sorted(glob.glob(parquet_files))

    bucket_loader = BucketDataLoader(
        parquet_files=parquet_files,
        bucket_file=bucket_file,
        base_resolution=(reference_size, reference_size),
        bsz=base_batch_size,
        world_size=world_size,
        global_rank=global_rank,
        shuffle=shuffle,
        seed=seed,
        num_workers=num_workers,
        use_dynamic_bsz=use_dynamic_bsz,
    )

    return bucket_loader
        


def main_test_dataloader():
    """完整使用示例"""
    import glob
    import tqdm
    
    # 1. 准备配置
    parquet_files = glob.glob("/path/to/preprocess_data/sd35m/part_rank*.parquet")
    
    bucket_file = "/path/to/preprocess_data/sd35m/bucket_index.csv"

    # 2. 创建 DataLoader
    bucket_loader = create_bucket_dataloader(
        parquet_files=parquet_files,
        bucket_file=bucket_file,
        reference_size=1024,
        base_batch_size=3,
        world_size=8,
        global_rank=0,
        shuffle=True,
        seed=42,
        num_workers=8,
    )
    print(f"Total batches in DataLoader: {len(bucket_loader)}")
    # 3. 迭代数据
    prompts = []
    for epoch in range(5):
        for step, batch in tqdm.tqdm(enumerate(bucket_loader)):
            if not batch:
                print(f"Step {step}: Empty batch, skipping.")
                continue

            prompt = batch['prompt']  # just for demo
            prompts.extend(prompt)

            prompt = [p[:50] for p in prompt]


            # latent_chosen = batch['latent_chosen']  # just for demo
            # latent_reject = batch['latent_reject']  # just for demo

            print(f"Step {step}:")
            # print(f"  Batch size: {len(prompt)}")
            print(f"  Prompt example: {prompt}...")
            # print(f"  latent_chosen shape: {latent_chosen.shape}")
            # print(f"  latent_reject shape: {latent_reject.shape}")

            
            # if step >= 20:  # 只测试前几个 batch
            #     break
            if step % 100 == 0:
                print(f"Processed {step} batches.")
                print(f"Prompt example: {batch['prompt'][0][:50]}...")
    
    print(f"Total prompts collected: {len(prompts)}")
    print(f"Unique prompts: {len(set(prompts))}")

if __name__ == "__main__":
    main_test_dataloader()