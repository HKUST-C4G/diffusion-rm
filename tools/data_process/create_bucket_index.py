import io, os, json, argparse, sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffusion_rm.data.bucket_manager import BucketManager

BUCKET_MANAGER = BucketManager(ar_thresh=0.03, divisible=32, debug=False)

def build_bucket_index_pandas(
    parquet_paths, 
    out_csv, 
    max_per_prompt=10, 
    filter_fn=None, 
    filter_columns=None
):
    """
    构建 Bucket 索引，支持自定义行级过滤。
    
    参数:
        filter_fn (callable): 过滤函数，接收 DataFrame 作为输入，返回一个 Boolean Series。
        filter_columns (list): 执行 filter_fn 所需的额外列名列表（例如 ["detailed_results"]）。
    """
    res_array = BUCKET_MANAGER.resolutions
    res_to_idx = { (int(h), int(w)) : i for i, (h, w) in enumerate(res_array) }
    
    all_data = []
    
    # 动态构建需要从 Parquet 中读取的列
    base_columns = ["latent1_shape", "prompt"]
    if filter_columns:
        # 去重并合并列名，防止重复加载
        read_columns = list(set(base_columns + filter_columns))
    else:
        read_columns = base_columns

    print(f"Step 1: Loading {len(parquet_paths)} parquet files...")
    for p in tqdm(parquet_paths):
        try:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                # 严格按照 read_columns 读取，避免内存爆炸
                table = pf.read_row_group(rg, columns=read_columns) 
                df = table.to_pandas()
                
                df['parquet_path'] = p
                df['row_group'] = rg
                df['row_index'] = range(len(df))
                
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue
    
    if not all_data:
        print("No data loaded!")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    total_scanned = len(full_df)

    # Step 1.5: 执行自定义过滤 (如果提供)
    if filter_fn is not None:
        print(f"Step 1.5: Applying custom filter...")
        # 记录过滤前的数量以供校验
        pre_filter_len = len(full_df)
        full_df = full_df[filter_fn(full_df)].reset_index(drop=True)
        print(f"Filtered out {pre_filter_len - len(full_df)} rows based on custom criteria.")

    # Step 2: 解析分辨率
    def get_hw(shape):
        try:
            h = int(shape[-2] * 8)
            w = int(shape[-1] * 8)
            return h, w
        except:
            return None, None

    print("Step 2: Processing resolutions...")
    hw_series = full_df['latent1_shape'].apply(get_hw)
    full_df['height'] = hw_series.apply(lambda x: x[0])
    full_df['width'] = hw_series.apply(lambda x: x[1])

    # Step 3: 筛选逻辑 - 随机选择 Top K
    print(f"Step 3: Filtering prompts (Max {max_per_prompt} per prompt)...")
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    full_df = full_df.groupby('prompt').head(max_per_prompt)

    # Step 4: 匹配 Bucket ID
    print("Step 4: Mapping to buckets...")
    full_df['bucket_id'] = full_df.apply(
        lambda row: res_to_idx.get((row['height'], row['width'])), axis=1
    )
    
    full_df = full_df.dropna(subset=['bucket_id'])
    full_df['bucket_id'] = full_df['bucket_id'].astype(int)

    # Step 5: 排序并生成 Global ID
    full_df = full_df.sort_values(['parquet_path', 'row_group', 'row_index']).reset_index(drop=True)
    full_df['global_id'] = full_df.index

    final_cols = ["global_id", "parquet_path", "row_group", "row_index", "width", "height", "bucket_id"]
    full_df[final_cols].to_csv(out_csv, index=False)

    print("-" * 30)
    print(f"Final Statistics:")
    print(f"Total samples scanned: {total_scanned}")
    print(f"Samples kept (GID):    {len(full_df)}")
    print(f"Retention rate:        {(len(full_df)/total_scanned)*100:.2f}%")
    print(f"Bucket index written to: {out_csv}")
    print("-" * 30)


def filter_fn(df):
    import ast
    parsed_results = df['detailed_results'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
    )
    
    votes_chosen = parsed_results.apply(lambda d: d.get('votes_chosen', 0))
    votes_rejected = parsed_results.apply(lambda d: d.get('votes_rejected', 0.0))

    confidence = votes_chosen / (votes_chosen + votes_rejected + 1e-6)  # 避免除零
    
    # 返回 Boolean Series
    return (confidence > 0.6) | (votes_chosen == 0) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build bucket index CSV from processed Parquet latents.")
    parser.add_argument("--parquet_dir", type=str, required=True, help="Glob pattern for input Parquet files.")
    parser.add_argument("--out_index", type=str, required=True, help="Path to save the output CSV index.")
    parser.add_argument("--max_per_prompt", type=int, default=8, help="Max samples to keep per unique prompt.")
    parser.add_argument("--use_filter", action="store_true", help="Enable the custom filtering logic.")
    
    args = parser.parse_args()

    # 处理通配符路径
    parquet_paths = sorted(glob.glob(args.parquet_dir, recursive=True))
    
    if not parquet_paths:
        print(f"[!] No parquet files found for pattern: {args.parquet_dir}")
        sys.exit(1)

    # 准备过滤参数
    f_fn = filter_fn if args.use_filter else None
    f_cols = ["detailed_results"] if args.use_filter else None

    build_bucket_index_pandas(
        parquet_paths=parquet_paths, 
        out_csv=args.out_index, 
        max_per_prompt=args.max_per_prompt,
        filter_fn=f_fn,
        filter_columns=f_cols
    )