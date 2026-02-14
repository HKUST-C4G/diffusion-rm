from setuptools import setup, find_packages

setup(
    name="diffusion-rm",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio",
        "transformers==4.51.3",
        "accelerate==1.8.1",
        "diffusers==0.36.0", 
        "deepspeed==0.17.1", 
        "peft==0.17.1", 
        
        "omegaconf",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",

        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "wandb==0.18.7",
        "requests==2.32.3",
        "matplotlib==3.10.0",
        
        # "flash-attn==2.7.4.post1",
        "huggingface-hub==0.36.2",
        "datasets==4.0.0",
        "tokenizers==0.21.2",
        
        "einops==0.8.1",
        "xformers",
        "absl-py==2.1.0",
        "ml_collections",
        "sentencepiece>=0.2.0",
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
