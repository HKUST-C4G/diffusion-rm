# Modified from https://github.com/NovelAI/novelai-aspect-ratio-bucketing/blob/main/bucketmanager.py

import numpy as np
import pickle
import time
import pandas as pd
import tqdm

# CUSTOM_BUCKETS = [
#     (256, 256), (512, 512), (1024, 1024),   # 1:1
#     (1248, 832), (1344, 896), (1536, 1024), (1888, 1248),    # 3:2
#     (832, 1248), (896, 1344), (1024, 1536), (1248, 1888),   # 2:3
#     (1024, 768),    # 4:3
#     (768, 1024),    # 3:4
#     # (768, 1344),    # 4:7
#     (1472, 832),   # 16:9
#     (832, 1472),    # 9:16
#     (1152, 832), (832, 1152)   # others
# ]

CUSTOM_BUCKETS = [
    (256, 256), (512, 512), (1024, 1024),   # 1:1
    (1248, 832), (1344, 896),    # 3:2
    (832, 1248), (896, 1344),   # 2:3
    (1024, 768),    # 4:3
    (768, 1024),    # 3:4
    # (768, 1344),    # 4:7
    (1472, 832),   # 16:9
    (832, 1472),    # 9:16
    (1152, 832), (832, 1152)   # others
]

def get_prng(seed):
    return np.random.RandomState(seed)


class BucketManager:
    def __init__(self, 
                 bucket_file=None, 
                 divisible=32, 
                 ar_thresh=0.03, 
                 base_resolution=(1024, 1024),
                 bsz=3,
                 world_size=1,
                 global_rank=0,
                 seed=42,
                 use_dynamic_bsz=True,
                 debug=False,
                 num_workers: int = 1,
                 worker_id: int = 0
                 ):
        self.div = divisible
        self.ar_thresh = ar_thresh
        self.debug = debug

        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank

        self.num_workers = num_workers
        self.worker_id = worker_id

        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed)

        self.resolutions = None
        self.aspects = None
        self.areas = None

        self.res_map = {}  # {global_id: (H, W)}
        self.sample_info = {}  # {global_id: {'parquet_path': ..., 'row_group': ..., 'row_index': ..., 'width': ..., 'height': ...}}
        self.buckets = {}

        self.epoch = None
        self.batch_total = 0
        self.batch_delivered = 0

        self.gen_buckets()
        if use_dynamic_bsz:
            base_area = base_resolution[0] * base_resolution[1]
            bucket_areas = self.areas
            self.bucket_bsz = np.maximum(1, (base_area // bucket_areas) * self.bsz)
        else:
            self.bucket_bsz = np.full((len(self.resolutions),), self.bsz, dtype=np.int32)

        if bucket_file is not None:
            self.load_from_csv(bucket_file)
            if len(self.res_map) > 0:
                self.build_buckets()
                self.start_epoch()

    def set_worker_info(self, num_workers: int, worker_id: int):
        self.num_workers = num_workers
        self.worker_id = worker_id

    def load_from_csv(self, bucket_file):
        """
        从 CSV 文件读取样本信息
        CSV 格式: global_id, parquet_path, row_group, row_index, width, height, bucket_id
        """
        # reprocess buckets
        res_to_id = { (int(h), int(w)) : i for i, (h, w) in enumerate(self.resolutions) }
        df = pd.read_csv(bucket_file)
        df['bucket_id'] = df.apply(lambda row: res_to_id.get((int(row['height']), int(row['width']))), axis=1)
        missed = df['bucket_id'].isnull().sum()
        if missed > 0:
            print(f"Warning: {missed} entries in bucket index could not be matched to any bucket.")
        df = df.dropna(subset=['bucket_id'])
        # select the first 1000 to debug
        # df = df.reset_index(drop=True)
        # df = df.iloc[:1000]

        for row in df.itertuples(index=False):
            gid = int(row.global_id)
            w = int(row.width)
            h = int(row.height)
            b_id = int(row.bucket_id)

            self.res_map[gid] = (h, w)

            self.sample_info[gid] = {
                'global_id': gid,
                'parquet_path': row.parquet_path,
                'row_group': int(row.row_group),
                'row_index': int(row.row_index),
                'bucket_id': b_id,
                'width': w,
                'height': h
            }

        if self.debug:
            print(f"[load_from_csv] loaded {len(self.res_map)} items")

    def gen_buckets(self):
        buckets = list(CUSTOM_BUCKETS)

        # normalize and check
        normed = []
        seen = set()
        for h, w in buckets:
            h, w = int(h), int(w)
            if h <= 0 or w <= 0:
                continue
            if (h, w) in seen:
                continue
            if self.div is not None and (h % self.div != 0 or w % self.div != 0):
                print(f"Warning: bucket {(h,w)} is not divisible by {self.div}, rounding down.")
                h = (h // self.div) * self.div
                w = (w // self.div) * self.div
            seen.add((h, w))
            normed.append((h, w))
        buckets = normed

        if not buckets:
            raise ValueError("No valid buckets remain after filtering. Check CUSTOM_BUCKETS / constraints.")

        # sorted by area
        ordered = sorted(buckets, key=lambda x: x[0]*x[1])

        # create resolutions and aspects arrays
        self.resolutions = np.array(ordered, dtype=np.int32)          # 形如 [[W,H],[W,H],...]
        self.aspects = np.array([h / float(w) for (h, w) in ordered], dtype=np.float32)
        self.areas = np.array([h * w for (h, w) in ordered], dtype=np.int32)

        if self.debug:
            print(f"resolutions:\n{self.resolutions}")

    def assign_bucket(self, res, ar_thresh=None, return_res=False):
        h, w = int(res[0]), int(res[1])

        # first find candidate buckets within threshold
        aspect = h / float(w)
        bucket_aspects = self.aspects
        bucket_res = self.resolutions

        ar_errors = np.abs(np.log(bucket_aspects) - np.log(aspect))

        thr = ar_thresh or self.ar_thresh

        cand_dixs = np.where(ar_errors <= thr)[0]
        if len(cand_dixs) == 0:
            return None
        
        # from candidates, pick the one with the most similar area
        area = h * w
        cand_areas = self.areas[cand_dixs]
        area_errors = np.abs(np.log(cand_areas) - np.log(area))
        best_cand = cand_dixs[np.argmin(area_errors)]

        if return_res:
            return tuple(bucket_res[best_cand])
        else:
            return best_cand

    def build_buckets(self):
        """把 res_map 按 assign_bucket 分到 self.buckets"""
        self.buckets = {}
        skipped = 0
        self.aspect_errors = []

        for post_id, (h, w) in self.res_map.items():
            bid = self.assign_bucket((h, w))
            if bid is None:
                skipped += 1
                continue
            self.buckets.setdefault(bid, []).append(post_id)
            if self.debug:
                # 记录线性 AR 误差用于参考
                aspect = float(h) / float(w)
                self.aspect_errors.append(abs(self.aspects[bid] - aspect))

        if self.debug:
            self.aspect_errors = np.array(self.aspect_errors) if self.aspect_errors else np.array([0.0])
            print(f"[build_buckets] skipped: {skipped}")
            for bid in sorted(self.buckets.keys()):
                print(f"  bucket {bid} {tuple(self.resolutions[bid])}: {len(self.buckets[bid])} items")
            print(f"[build_buckets] aspect error mean={self.aspect_errors.mean():.4f}, "
                  f"median={np.median(self.aspect_errors):.4f}, max={self.aspect_errors.max():.4f}")
            print(f"[build_buckets] total buckets used: {len(self.buckets)}")
            print(f"[build_buckets] total items assigned: "
                  f"{sum(len(v) for v in self.buckets.values())}")

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            t0 = time.perf_counter()
        
        if world_size is not None:
            self.world_size = int(world_size)
        if global_rank is not None:
            self.global_rank = int(global_rank)

        # shuffle & slice for this rank
        index = np.array(sorted(self.res_map.keys()), dtype=np.int64)
        index = self.epoch_prng.permutation(index)

        batch_entries = []

        for bucket_id in tqdm.tqdm(sorted(self.buckets.keys()), desc="Preparing buckets"):
            bucket_bs = self.bucket_bsz[bucket_id]

            global_bucket_bs = bucket_bs * self.world_size
            # arr = np.array([pid for pid in self.buckets[bucket_id] if pid in index], dtype=np.int64)
            arr = np.array(self.buckets[bucket_id], dtype=np.int64)
            if arr.size == 0:
                continue

            self.prng.shuffle(arr)

            usable_len = (arr.size // global_bucket_bs) * global_bucket_bs
            if usable_len == 0:
                continue
            arr = arr[:usable_len]

            # Rank 切片
            arr_rank = arr[self.global_rank::self.world_size]

            # 按 bucket_bs 切分完整 batch
            n_full = (arr_rank.size // bucket_bs) * bucket_bs
            arr_rank = arr_rank[:n_full]

            for i in range(0, n_full, bucket_bs):
                batch_ids = arr_rank[i : i + bucket_bs].tolist()
                batch_entries.append(
                    (bucket_id, bucket_bs, tuple(self.resolutions[bucket_id]), batch_ids)
                )

        if not batch_entries:
            raise RuntimeError(
                "No valid batches produced; try reducing batch sizes or merging buckets."
            )

        # 打乱 batch 队列
        perm = self.epoch_prng.permutation(len(batch_entries))
        self._batch_queue = [batch_entries[i] for i in perm]

        # 更新状态
        self.batch_total = len(self._batch_queue)
        self.batch_delivered = 0
        self.epoch = True

        if self.debug:
            print(f"[start_epoch] rank={self.global_rank}: "
                f"{self.batch_total} batches, time={time.perf_counter()-t0:.4f}s")

    # def get_batch(self):
    #     if self._batch_queue is None or self.batch_delivered is None or self.batch_delivered >= self.batch_total:
    #         self.start_epoch()

    #     bucket_id, bucket_bs, resolution, batch_ids = self._batch_queue[self.batch_delivered]
    #     # import pdb; pdb.set_trace()
    #     self.batch_delivered += 1

    #     batch_sample_info = [self.sample_info[pid] for pid in batch_ids]

    #     if self.debug:
    #         print(f"[get_batch] step={self.batch_delivered-1} "
    #             f"bucket={bucket_id} res={resolution} ids={batch_ids}")
    #     return (batch_ids, resolution, batch_sample_info)
    
    def generator(self):
        if self._batch_queue is None or self.batch_delivered is None or self.batch_delivered >= self.batch_total:
            self.start_epoch()
        # while self.batch_delivered < self.batch_total:
        #     yield self.get_batch()
        for batch_idx in range(self.batch_total):
            if batch_idx % self.num_workers != self.worker_id:
                continue

            bucket_id, bucket_bs, res, batch_ids = self._batch_queue[batch_idx]
            batch_sample_infos = [self.sample_info[pid] for pid in batch_ids]

            yield (batch_ids, res, batch_sample_infos)

def main_test_bucket_assign():
    bm = BucketManager(debug=True)
    test_res = [
        (512, 512),
        (1024, 1024),
        (1200, 800),
        (800, 1200),
        (1024, 768),
        (768, 1024),
        (800, 1400),
        (1600, 900),
        (900, 1600),
        (1234, 824),
        (1000, 900),
    ]
    for res in test_res:
        bucket_id = bm.assign_bucket(res)
        if bucket_id is not None:
            print(f"res {res} assigned to bucket {bm.resolutions[bucket_id]}")
        else:
            print(f"res {res} could not be assigned to any bucket")


def main_test_bucket_manager():
    # ====== 1) 构造一个假的 res_map（id -> (W,H)）======
    def synth_res_map(n=1000, seed=0):
        rng = np.random.RandomState(seed)
        res_map = {}
        all_buckets = [tuple(x) for x in CUSTOM_BUCKETS]
        B = len(all_buckets)
        # 随机把样本分到各目标桶附近（±5% 抖动）
        for i in range(n):
            bw, bh = all_buckets[rng.randint(0, B)]
            dw = int(bw * (1.0 + rng.uniform(-0.05, 0.05)))
            dh = int(bh * (1.0 + rng.uniform(-0.05, 0.05)))
            res_map[i] = (max(16, dw), max(16, dh))
        return res_map

    res_map = synth_res_map(n=10000, seed=123)

    # ====== 2) 单机单卡测试 ======
    print("\n=== Single GPU test ===")
    bm = BucketManager(bucket_file=None, ar_thresh=0.08, bsz=2, world_size=1, global_rank=0, debug=False)
    bm.res_map = res_map
    bm.build_buckets()
    bm.start_epoch()
    import pdb; pdb.set_trace()
    seen = 0
    for step, (ids, res) in enumerate(bm.generator()):
        seen += len(ids)
    print(f"[single] total seen={seen}")

    # ====== 3) 多卡一致性测试（world_size=4）=====
    print("\n=== Multi-GPU slicing test (4 ranks) ===")
    world = 8
    managers = []
    for r in range(world):
        m = BucketManager(bucket_file=None, ar_thresh=0.08, bsz=2, world_size=world, global_rank=r, debug=False)
        m.res_map = res_map
        m.build_buckets()
        m.start_epoch()
        managers.append(m)

    # 每个 rank 迭代自己那份，统计全局覆盖是否一致、不重不漏
    covered = set()
    for r, m in enumerate(managers):
        for ids, res in m.generator():
            covered.update(ids)

    print(f"[multi] unique covered={len(covered)}")
    # 注意：因为 start_epoch 会对全局样本整除裁剪，所以 unique covered <= len(res_map)
    # 且 total_rank_seen == len(covered) * world_size / world_size == len(covered)


def main_test_csv_loading():
    """测试 CSV 加载功能"""
    print("\n=== Testing CSV loading ===")
    
    # 假设你的 CSV 文件路径
    csv_path = "/path/to/preprocess_data/sd35m/bucket_index.csv"
    
    bm = BucketManager(
        bucket_file=csv_path,
        ar_thresh=0.03,
        bsz=2,
        world_size=8,
        global_rank=0,
        debug=True
    )
    
    print(f"\nTotal samples loaded: {len(bm.res_map)}")
    print(f"Total buckets: {len(bm.buckets)}")
    
    # 测试获取 batch
    print("\n=== Testing batch generation ===")
    for step, (ids, res, sample_infos) in enumerate(bm.generator()):
        print(f"\nStep {step}:")
        print(f"  Resolution: {res}")
        print(f"  Batch size: {len(ids)}")
        print(f"  Sample IDs: {ids}")
        print(f"  First sample info: {sample_infos[0]}")
        
        if step >= 5:  # 只测试前几个 batch
            break

if __name__ == "__main__":
    # main_test_bucket_assign()
    # main_test_bucket_manager()
    main_test_csv_loading()
