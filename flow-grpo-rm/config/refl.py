import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_refl_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.train.learning_rate = 3e-5
    config.sample.num_steps = 40
    config.sample.train_num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.use_cfg_train = False
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 4
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-4
    config.mixed_precision = "bf16"

    config.train.pretrain_data = "/path/to/your/pretrained_data"
    config.train.pretrain_batch_size = config.train.batch_size
    config.train.pretrain_loss = 0.0

    config.train.beta = 0.02   # no use
    config.sample.global_std = False
    config.sample.noise_level = 5
    config.train.ema = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.run_name = '[ReFL + SD3.5-M + hpsv3]_8gpu_bs2_accumulation4_lr3e-5_kl0.001'
    config.save_dir = f'logs/{config.run_name}'
    config.save_eval_dir = f'log_save_images/{config.run_name}'
    config.reward_fn = {
        "hpsv3": 1.0,
        # 'pickscore': 1.0
    }

    config.proxy_reward_fn = {
        'pickscore': 1.0
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_refl_rm_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.train.learning_rate = 1e-4
    config.sample.num_steps = 40
    config.sample.train_num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 1
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.use_cfg_train = True
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 4
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 3e-5
    config.mixed_precision = "bf16"

    ## we provide the implementation of pretrain loss in the training script
    ## ReFL is easily to be hacked by the reward model
    ## using pretrain_loss>0 will enable the pretrain loss to help avoid reward hacking
    ## recommend dataset: https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M
    config.train.pretrain_data = "/path/to/your/pretrained_data"
    config.train.pretrain_batch_size = config.train.batch_size
    config.train.pretrain_loss = 0.0

    config.train.beta = 0.02
    config.sample.global_std = False
    config.sample.noise_level = 5
    config.train.ema = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.run_name = '[ReFL + SD3.5-M + ours]_8gpu_bs2_accumulation4_lr3e-5_kl0.001'
    config.save_dir = f'logs/{config.run_name}'
    config.save_eval_dir = f'log_save_images/{config.run_name}'
    config.reward_fn = {
        # "hpsv3": 1.0,
        'pickscore': 1.0    # just 
    }
    config.proxy_reward_fn = {
        'pickscore': 1.0
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = False
    return config


def pickscore_sd3_refl_debug():
    gpu_number=1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.train_num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 2
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.use_cfg_train = False
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-4
    config.mixed_precision = "bf16"

    config.train.beta = 0.001
    config.train.pretrain_data = "/path/to/your/pretrained_data"
    config.train.pretrain_batch_size = config.train.batch_size
    config.train.pretrain_loss = 0.0

    config.sample.global_std = False
    config.sample.noise_level = 5
    config.train.ema = False
    config.save_freq = 3000 # epoch
    config.eval_freq = 30
    config.run_name = 'debug'
    config.save_dir = f'logs/{config.run_name}'
    config.save_eval_dir = f'log_save_images/{config.run_name}'
    config.reward_fn = {
        # "hpsv3": 1.0,
        'pickscore': 1.0
    }
    config.proxy_reward_fn = {
        'pickscore': 1.0
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()