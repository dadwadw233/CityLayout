from model.DDPM import GaussianDiffusion
from model.Unet import Unet
from trainer import Trainer
from utils.utils import load_config, OSMVisulizer, Vectorizer
import argparse
import torch
import numpy as np
import random

argparser = argparse.ArgumentParser()
argparser.add_argument('--train_type', type=str, default='one-hot')
argparser.add_argument('--eval', type=str, default='True')


print('training data type: ',argparser.parse_args().train_type)

train_type = argparser.parse_args().train_type



if train_type == 'one-hot':
    data_config = load_config('/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/osm_cond_loader.yaml')
    trainer_config = load_config('/home/admin/workspace/yuyuanhong/code/CityLayout/config/train/osm_generator.yaml')
elif train_type == 'rgb':
    data_config = load_config('/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/osm_loader_rgb.yaml')
    trainer_config = load_config('/home/admin/workspace/yuyuanhong/code/CityLayout/config/train/osm_generator_rgb.yaml')

if argparser.parse_args().eval == 'True':
    trainer_config = load_config('/home/admin/workspace/yuyuanhong/code/CityLayout/config/train/osm_cond_generator_sample.yaml')

seed_value = 3407  

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
np.random.seed(seed_value)
random.seed(seed_value)

model = Unet(
    dim = trainer_config['model']['dim'],
    init_dim = None if trainer_config['model']['init_dim'] == 'None' else trainer_config['model']['init_dim'],
    out_dim = None if trainer_config['model']['out_dim'] == 'None' else trainer_config['model']['out_dim'],
    dim_mults = trainer_config['model']['dim_mults'],
    channels = trainer_config['model']['channels'],
    self_condition = trainer_config['model']['self_condition'],
    resnet_block_groups = trainer_config['model']['resnet_block_groups'],
    learned_variance = trainer_config['model']['learned_variance'],
    learned_sinusoidal_cond =  trainer_config['model']['learned_sinusoidal_cond'],
    random_fourier_features = trainer_config['model']['random_fourier_features'],
    learned_sinusoidal_dim = trainer_config['model']['learned_sinusoidal_dim'],
    sinusoidal_pos_emb_theta = trainer_config['model']['sinusoidal_pos_emb_theta'],
    attn_dim_head = trainer_config['model']['attn_dim_head'],
    attn_heads = trainer_config['model']['attn_heads'],
    full_attn = None if trainer_config['model']['full_attn'] == 'None' else trainer_config['model']['full_attn'],   
    flash_attn = trainer_config['model']['flash_attn'],
    resnet_block_num = trainer_config['model']['resnet_block_num'], 
    conditional=trainer_config['model']['conditional'],
    conditional_dim=trainer_config['model']['conditional_dim'],
)

diffusion = GaussianDiffusion(
    model,
    image_size = trainer_config['diffusion']['image_size'],
    timesteps = trainer_config['diffusion']['timesteps'], # number of steps
    sampling_timesteps = trainer_config['diffusion']['ddim_timestep'], # number of steps used for sampling
    objective=trainer_config['diffusion']['objective'], 
    beta_schedule=trainer_config['diffusion']['beta_schedule'],
    ddim_sampling_eta=trainer_config['diffusion']['ddim_sampling_eta'],
    auto_normalize=trainer_config['diffusion']['auto_normalize'],
    offset_noise_strength=trainer_config['diffusion']['offset_noise_strength'],
    min_snr_gamma=trainer_config['diffusion']['min_snr_gamma'],
    min_snr_loss_weight=trainer_config['diffusion']['min_snr_loss_weight'],
)

trainer = Trainer(
    diffusion,
    dataset_config=data_config,
    trainer_config=trainer_config,
)

if trainer_config['trainer']['mode'] == 'eval':
    
        
    vis = OSMVisulizer(trainer_config["vis"]["channel_to_rgb"])
    vec = Vectorizer()

    ret = trainer.sample(trainer_config['trainer']['num_samples'], trainer_config['trainer']['batch_size'], trainer_config['trainer']['milestone'],trainer_config['trainer']['condition'])

    if train_type == 'one-hot':
        vis.visulize_onehot_layout(ret, "./sample-onehot.png")
        vis.visualize_rgb_layout(ret, "./sample-rgb.png")

        # data_for_vec = ret
        # f = vec.vectorize(data_for_vec[:, 0:1, :, :], 'building')
        # vec.vectorize(data_for_vec[:, 2:3, :, :], data_type='road', init_features=f, color='blue')
    
    print(ret.shape)

else :
    trainer.train()