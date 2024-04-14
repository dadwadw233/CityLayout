from .log import INFO
from model.Unet import Unet
from model.DDPM import GaussianDiffusion

def init_backbone(cfg=None):
        backbone = Unet(**cfg)
        INFO(f"Backbone initialized!")
        return backbone
        
    
def init_diffuser(backbone, cfg=None):
    diffusion_config = cfg.copy()
    diffusion_config['model'] = backbone
    diffusion = GaussianDiffusion(**diffusion_config)
    return diffusion