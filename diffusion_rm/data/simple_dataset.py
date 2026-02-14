import os
import glob
from typing import List, Dict, Any, Optional, Callable, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import io


def npy_bytes_to_ndarray(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


class SimpleParquetDataset(Dataset):
    def __init__(
        self,
        parquet_files: List[str],
        process_fn: Optional[Callable] = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        
        self.parquet_files = parquet_files
        self.process_fn = process_fn or self._default_process_fn
        self.shuffle = shuffle
        self.seed = seed

        self.data = []

        self.total_samples = 0
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            self.total_samples += len(df)

            self.data.append(df)

        self.data = pd.concat(self.data, ignore_index=True)

        if self.shuffle:
            self.data = self.data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def _default_process_fn(self, row: Dict) -> Dict:
        """
        默认的数据处理函数（HPD格式）
        
        可以通过传入自定义process_fn来覆盖此函数
        """
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
        """返回数据集总样本数"""
        return self.total_samples

    def __getitem__(self, index: int):
        """根据索引返回单个样本"""
        row = self.data.iloc[index]
        sample = self.process_fn(row)
        return sample


# ============================================================================
# 使用示例和工具函数
# ============================================================================

def create_simple_dataloader(
    parquet_files: Union[str, List[str]],
    process_fn: Optional[Callable] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    **kwargs
) -> DataLoader:

    if isinstance(parquet_files, str):
        parquet_files = sorted(glob.glob(parquet_files))

    dataset = SimpleParquetDataset(
        parquet_files=parquet_files,
        process_fn=process_fn,
        shuffle=shuffle,
        **kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # collate_fn=collate_fn,
        drop_last=False,
    )




# ============================================================================
# 完整使用示例
# ============================================================================

def example_basic_usage():
    """示例1: 基本使用"""
    import glob
    
    # 获取所有parquet文件
    # parquet_files = glob.glob("/mnt/shangcephfs/mm-base-vision-ascend-2/layke/data/genai-bench/sd3_data/test_strict_aligned_res.parquet")
    parquet_files = glob.glob("/path/to/preprocess_data/sd35m-test/sample_1000.parquet")
    
    loader = create_simple_dataloader(
        parquet_files=parquet_files,
        batch_size=1,
        num_workers=4,
        shuffle=True,
    )
    
    # 迭代
    for i, sample in enumerate(loader):
        # sample 是单个样本的dict
        if i % 100 == 0:
            print(f"Sample {i}:")
            print(f"Prompt: {sample['prompt']}")
            print(f"Shape: {sample['latent_chosen'].shape}")
        # import pdb; pdb.set_trace()


def example_custom_process():
    """示例2: 自定义处理函数"""
    import glob
    
    def my_process_fn(row):
        """只加载图像，不区分chosen/reject"""
        latent = npy_bytes_to_ndarray(row['latent1'])
        return {
            'image': torch.from_numpy(latent),
            'text': row['prompt'],
        }
    
    dataset = SimpleParquetDataset(
        parquet_files=glob.glob("/path/to/data/*.parquet"),
        process_fn=my_process_fn,  # 使用自定义函数
    )
    
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    for sample in loader:
        image = sample['image']
        text = sample['text']
        # 训练代码...


def example_manual_batching():
    """示例3: 手动组batch（适合需要动态batch_size的场景）"""
    import glob
    
    dataset = SimpleParquetDataset(
        parquet_files=glob.glob("/path/to/data/*.parquet"),
    )
    
    # batch_size=1，手动组batch
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    manual_batch = []
    target_batch_size = 8
    
    for sample in loader:
        manual_batch.append(sample)
        
        if len(manual_batch) >= target_batch_size:
            # 组成一个batch
            batch = {
                'latent_chosen': torch.stack([s['latent_chosen'] for s in manual_batch]),
                'latent_reject': torch.stack([s['latent_reject'] for s in manual_batch]),
                'prompt': [s['prompt'] for s in manual_batch],
            }
            
            # 训练
            loss = model(batch)
            loss.backward()
            
            # 清空buffer
            manual_batch = []


def example_with_quickstart_function():
    """示例4: 使用快捷函数"""
    import glob
    
    # 一行创建DataLoader
    loader = create_simple_dataloader(
        parquet_files=glob.glob("/path/to/data/*.parquet"),
        batch_size=1,
        shuffle_files=True,
    )
    
    # 直接使用
    for sample in loader:
        # 你的训练代码
        pass


# ============================================================================
# 性能测试
# ============================================================================

def benchmark_reading_speed():
    """测试读取速度"""
    import time
    import glob
    
    parquet_files = glob.glob("/path/to/data/*.parquet")[:3]  # 测试3个文件
    
    dataset = SimpleParquetDataset(
        parquet_files=parquet_files,
        verbose=False,
    )
    
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    start_time = time.time()
    sample_count = 0
    
    for sample in loader:
        sample_count += 1
        if sample_count >= 1000:  # 只测试1000个样本
            break
    
    elapsed = time.time() - start_time
    speed = sample_count / elapsed
    
    print(f"读取速度: {speed:.1f} samples/sec")
    print(f"总样本数: {sample_count}")
    print(f"总耗时: {elapsed:.2f}s")


if __name__ == "__main__":
    print("🚀 SimpleParquetDataset - 简单流式数据集")
    print("="*80)
    
    # 取消注释来运行示例
    example_basic_usage()
    # example_custom_process()
    # example_manual_batching()
    # example_with_quickstart_function()
    # benchmark_reading_speed()
    
    print("\n✅ 模块加载成功！")
    print("\n使用方法:")
    print("  from simple_parquet_dataset import SimpleParquetDataset")
    print("  from simple_parquet_dataset import create_simple_dataloader  # 快捷函数")