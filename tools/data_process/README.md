# Data Processing Pipeline for HPDv3

This document outlines the data processing pipeline for preparing the HPDv3 dataset. The pipeline is designed to handle high-resolution image data efficiently by converting pixels to VAE latents, assigning aspect-ratio buckets, and building a scalable columnar index.

## Step 1: Data Download and Extraction

First, download the raw HPDv3 dataset from Hugging Face.

```bash
export DATA_ROOT="/path/to/your/hpdv3/ori_data"
export META_DIR="/path/to/your/meta_dir" # Target directory for images

huggingface-cli download \
    MizzenAI/HPDv3 \
    --repo-type dataset \
    --local-dir ${DATA_ROOT} \
    --local-dir-use-symlinks False


cd ${DATA_ROOT}
cat images.tar.gz.* | gunzip | tar -xvf - -C ${META_DIR}    # This avoids the I/O when load_datasets

```


## Step 2: Preprocessing (Pixel to Latent)

**Scripts:** * `tools/data_process/prepare_dataset.py`

* `tools/data_process/process.sh`

During this stage, we encode the raw RGB images into continuous latent representations using the designated VAE. Processing data offline significantly drastically reduces GPU memory consumption and compute overhead during the actual training loop.

**Core Operations:**

1. **Aspect Ratio (AR) Bucketing:** Images are dynamically grouped into predefined resolution buckets. This guarantees that the model learns from high-fidelity data that preserves its native aspect and resolution.
2. **VAE Encoding:** Images are batched and passed through the VAE.
3. **Parquet Storage:** The metadata, latent shapes, and binary latent byte streams are written to `.parquet` files. We use an asynchronous columnar writer to decouple high-throughput GPU encoding from disk I/O.

**Execution:**
Execute the distributed processing script. Ensure your multi-node/multi-GPU is correctly configured in the shell script.

```bash
bash tools/data_process/process.sh

```

## Step 3: Index Generation and Filtering

**Script:** `tools/data_process/create_bucket_index.py`

Before training, you must construct a global CSV index that maps every valid latent to its corresponding bucket ID. 

**Custom Data Filtering:**
The index generation script exposes a vectorized `filter_fn` interface via Pandas. You can define custom criteria to discard low-quality or irrelevant data prior to indexing.

For example, you can filter the dataset based on the parsed `detailed_results` column to strictly retain sample pairs where the `votes_chosen` strictly exceeds a predefined confidence threshold.

**Execution:**

```bash
python tools/data_process/create_bucket_index.py \
    --parquet_dir "/path/to/parquet/outputs/part_rank*.parquet" \
    --out_index "/path/to/output/bucket_index.csv"

```



