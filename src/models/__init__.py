from .mlp import SimpleMLPAdaLN
from .unet import UNetModel
from .dit import DiT

def init_model(model_name, **kwargs):
    model_constructors = {
        'mlp': SimpleMLPAdaLN,
        'unet_xs': UNET_XS,
        'unet_s': UNET_S,
        'unet_m': UNET_M,
        'unet_l': UNET_L,
        'unet_xl': UNET_XL,
        'dit': DiT,
    }
    if model_name not in model_constructors:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(model_constructors.keys())}")
    return model_constructors[model_name](**kwargs)

def create_unet_model(latent_size=32, model_channels=256, num_res_blocks=2, num_heads=8, channel_mult=[1,2,4], context_dim=512, ncls=15):
    model = UNetModel(image_size=latent_size, 
                    in_channels=272,
                    model_channels=model_channels, 
                    out_channels=272, 
                    num_heads=num_heads, 
                    num_res_blocks=num_res_blocks, 
                    dropout=0.,
                    attention_resolutions=[2, 4, 8], 
                    channel_mult = channel_mult,
                    num_head_channels= model_channels//num_heads,
                    use_spatial_transformer= False,
                    num_classes=ncls,
                    transformer_depth= 2,
                    context_dim=None,
                )
    return model
    
def UNET_XS(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=64, num_heads=4, channel_mult=[1,2,4], context_dim=128, ncls=ncls)

def UNET_S(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=128, num_heads=4, channel_mult=[1,2,4], context_dim=256, ncls=ncls)

def UNET_M(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=192, num_heads=6, channel_mult=[1,2,4], context_dim=384, ncls=ncls)

def UNET_L(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=256, num_heads=8, channel_mult=[1,2,4], context_dim=512, ncls=ncls)

def UNET_XL(latent_size=32, ncls=15, **kwargs):
    return  create_unet_model(latent_size=latent_size, model_channels=320, num_heads=12, channel_mult=[1,2,4], context_dim=640, ncls=ncls)

UNET_models = {
'UNet_XS' : UNET_XS, 
'UNet_S' : UNET_S, 
'UNet_M' : UNET_M, 
'UNet_L' : UNET_L, 
'UNet_XL' : UNET_XL, 
}