import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, ResNet50_Weights
from .efficientnet import build_efficient
from .vae import create_vae

from einops import rearrange

class PDNWrapper(nn.Module):
    def __init__(self, model):
        super(PDNWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        out = self.model(x)
        return out, [out]  # Return the output and a list of features (single feature in this case)
    
class VAEWrapper(nn.Module):
    def __init__(self, model):
        super(VAEWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model.encode(x).sample().mul_(0.18215)  # Scale the output to match the latent space scaling
        return out, [out]  # Return the output and a list of features (single feature in this case)
    
    def encode(self, x):
        out = self.model.encode(x)  
        return out

class MAEWrapper(nn.Module):
    def __init__(self, model, out_block=-1, patch_size=16):
        super(MAEWrapper, self).__init__()
        self.model = model
        self.out_block = out_block
        self.patch_size = patch_size
    
    def forward(self, x):
        out = self.model(pixel_values=x, output_hidden_states=True)
        feat = out.hidden_states[self.out_block][:, 1:, :].permute(0, 2, 1)  # (B, num_patches, hidden_dim) -> (B, hidden_dim, num_patches), no CLS
        h = w = int(feat.shape[2] ** 0.5)  # Assuming square input, calculate height and width of the feature map
        feat_2d = feat.reshape(feat.shape[0], feat.shape[1], h, w)  # Reshape to (B, hidden_dim, H, W)
        return feat_2d, [feat_2d]  # Return the output and a list of features (single feature in this case)

class SiglipWrapper(nn.Module):
    def __init__(self, model, out_block=-1, patch_size=16):
        self.out_block = out_block
        self.patch_size = patch_size
        super(SiglipWrapper, self).__init__()
        self.model = model
      
    def forward(self, x):
        inputs = {"pixel_values": x}
        out = self.model(**inputs, output_hidden_states=True)
        feat = out.hidden_states[self.out_block].permute(0, 2, 1)  # (B, num_patches, hidden_dim) -> (B, hidden_dim, num_patches)
        h = w = int(feat.shape[2] ** 0.5)  # Assuming square input, calculate height and width of the feature map
        feat_2d = feat.reshape(feat.shape[0], feat.shape[1], h, w)  # Reshape to (B, hidden_dim, H, W)
        return feat_2d, [feat_2d]  # Return the output and a list of features (single feature in this case)

class DINOWrapper(nn.Module):
    def __init__(self, model, out_blocks=None, out_res=None, **kwargs):
        super(DINOWrapper, self).__init__()
        self.model = model
        self.out_blocks = out_blocks if out_blocks is not None else [-1]
        self.out_res = out_res if out_res is not None else (14, 14)  # Default shape for DINO Base
    
    def forward(self, x):
        """
        Forward pass through the DINO model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output features from the specified blocks.
        """
        outputs = self.model.forward_features(x)
        outputs = outputs[:, 1:]  # Skip the first token (CLS token)
        outputs = rearrange(outputs, 'b (h w) c -> b c h w', h=self.out_res[0], w=self.out_res[1])
        return outputs, [outputs]  # Return the output and a list of features (single feature in this case)

class DINO2Wrapper(nn.Module):
    def __init__(self, model, out_blocks=None, out_res=None, **kwargs):
        super(DINO2Wrapper, self).__init__()
        self.model = model
        self.out_blocks = out_blocks if out_blocks is not None else [-1]
        self.out_res = out_res if out_res is not None else (16, 16)  # Default shape for DINOv2 Base
    
    def forward(self, x):
        """
        Forward pass through the DINOv2 model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output features from the specified blocks.
        """
        outputs = self.model(x)
        hidden_states = outputs.hidden_states
        out_features = []
        for blk in self.out_blocks:
            state = hidden_states[blk][:, 1:]
            state = rearrange(state, 'b (h w) c -> b c h w', h=self.out_res[0], w=self.out_res[1])
            out_features.append(state)
        out = torch.cat(out_features, dim=1)
        return out, out_features
    
def get_normalization_func(normalization_method=None):
    if normalization_method is None:
        return nn.Identity()
    elif normalization_method == "instance":
        def norm_fn(x):
            x_mu = x.mean(dim=(1,2,3), keepdim=True)
            x_sigma = x.std(dim=(1,2,3), keepdim=True)
            return (x - x_mu) / (x_sigma + 1e-8)
        return norm_fn
    elif normalization_method == "batch":
        def norm_fn(x):
            x_mu = x.mean(dim=(0,2,3), keepdim=True)
            x_sigma = x.std(dim=(0,2,3), keepdim=True)
            return (x - x_mu) / (x_sigma + 1e-8)
        return norm_fn
    elif normalization_method == "layer":
        def norm_fn(x):
            x_mu = x.mean(dim=(1,2,3), keepdim=True)
            x_sigma = x.std(dim=(1,2,3), keepdim=True)
            return (x - x_mu) / (x_sigma + 1e-8)
        return norm_fn
    else:
        raise ValueError(f"Unsupported normalization method: {normalization_method}")
        
def get_backbone_feature_shape(model_name):
    if model_name == "efficientnet-b4":
        return (272, 16, 16)
    elif model_name == "vae-kl16":
        return (16, 16, 16)
    elif model_name == "wide_resnet50_2":
        return (2048, 8, 8)
    elif model_name == "dinov2-small":
        return (384, 16, 16)
    elif model_name == "dinov2-base":
        return (768, 16, 16)
    elif model_name == "dinov2-large":
        return (1024, 16, 16)
    elif model_name == "dinov1-small":
        return (384, 14, 14)
    elif model_name == "dinov1-base":
        return (768, 14, 14)
    elif model_name == "google/siglip2-base-patch16-224":
        return (768, 14, 14)
    elif model_name == "facebook/vit-mae-base":
        return (768, 14, 14)
    elif model_name == "pdn_small":
        return (384, 16, 16)
    elif model_name == "pdn_medium":
        return (384, 16, 16)
    elif model_name == "vgg19":
        return (960, 8, 8)
    elif model_name == "pixel":
        return (3, 256, 256)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

def get_efficientnet(**kwargs):
    return build_efficient(**kwargs)

def get_dino(model_name, **kwargs):
    if model_name == "dinov2-small":
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained("facebook/dinov2-small", output_hidden_states=True).eval()
        model = DINO2Wrapper(model, **kwargs)
        return model
    elif model_name == "dinov2-base":
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained("facebook/dinov2-base", output_hidden_states=True).eval()
        model = DINO2Wrapper(model, **kwargs)
        return model
    elif model_name == "dinov2-large":
        from transformers import Dinov2Model
        model = Dinov2Model.from_pretrained("facebook/dinov2-large", output_hidden_states=True).eval()
        model = DINO2Wrapper(model, **kwargs)
        return model
    elif model_name == "dinov1-small":
        import timm
        model = timm.create_model(
            'vit_small_patch16_224.dino',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).eval()
        return DINOWrapper(model, **kwargs)
    elif model_name == "dinov1-base":
        import timm
        model = timm.create_model(
            'vit_base_patch16_224.dino',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).eval()
        return DINOWrapper(model, **kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def get_pdn_small(out_channels=384, padding=False, **kwargs):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

def get_pdn_medium(out_channels=384, padding=False, **kwargs):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

def get_siglip(model_name, **kwargs):
    from transformers import SiglipVisionModel
    model = SiglipVisionModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash_attention_2 also works where supported
    ).eval()
    return model

def get_mae(model_name, **kwargs):
    from transformers import ViTMAEModel
    model = ViTMAEModel.from_pretrained(model_name).eval()
    model.config.mask_ratio = 0.0
    return model

def get_backbone_model(model_name):
    if model_name == "vgg19":
        return models.vgg19(weights=VGG19_Weights.DEFAULT)
    elif model_name == "efficientnet-s":
        return models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    elif model_name == "efficientnet-m":
        return models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    elif model_name == "efficientnet-l":
        return models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    elif model_name == "resnet50":
        return models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == "identical":
        return nn.Identity()
    elif model_name == "pdn_small":
        return get_pdn_small()
    elif model_name == "pdn_medium":
        return get_pdn_medium()

def get_backbone(**kwargs):
    model_name = kwargs['model_name']
    if 'pdn_small' in model_name:
        net = get_pdn_small(**kwargs)
        return PDNWrapper(net)
    elif 'pdn_medium' in model_name:
        net = get_pdn_medium(**kwargs)
        return PDNWrapper(net)
    elif 'vae' in model_name:
        net = create_vae(**kwargs)
        return VAEWrapper(net)
    elif 'efficientnet' in model_name:
        net =  get_efficientnet(**kwargs)
        return BackboneWrapper(net, scale_factors=[0.125, 0.25, 0.5, 1.0])
    elif 'dinov1' in model_name:
        net = get_dino(model_name, **kwargs)
        return net
    elif 'dinov2' in model_name:
        net = get_dino(model_name, **kwargs)
        return net
    elif 'siglip' in model_name:
        net = get_siglip(**kwargs)
        return SiglipWrapper(net, out_block=kwargs.get('out_block', -1))
    elif 'mae' in model_name:
        net = get_mae(**kwargs)
        return MAEWrapper(net, out_block=kwargs.get('out_block', -1))
    elif 'vgg' in model_name:
        return BackboneModel(model_name, [3, 8, 17, 26])
    elif 'pixel' in model_name:
        return Identity()
    else:
        raise ValueError(f"Invalid backbone model: {model_name}")

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, x, *args, **kwargs):
        return x, [x]

def get_intermediate_output_hook(layer, input, output):
    BackboneModel.intermediate_cache.append(output)

class BackboneModel(nn.Module):
    intermediate_cache = []
    
    def __init__(self, model_name: str, extract_indices: list, feature_res: int = 64):
        super(BackboneModel, self).__init__()
        self.model_name = model_name
        self.model = get_backbone_model(model_name)
        self.model.eval()
        self.model_name = model_name
        self.extract_indices = extract_indices
        self.feature_res = feature_res
        
        if model_name in ["pdn_small", "pdn_medium"]:
            self.feature_dim = 384
        elif model_name == "identical":
            self.feature_dim = 3
        else:
            self._register_hook()
        
    
    def _register_hook(self):
        self.layer_hooks = []
        feature_dim = 0
        if self.model_name == "vgg19":
            for i, layer_idx in enumerate(self.extract_indices):
                module = self.model.features[layer_idx-1]
                if isinstance(module, nn.Conv2d):
                    feature_dim += module.out_channels
                elif isinstance(module, nn.Sequential):
                    if isinstance(module[-1], nn.SiLU):
                        feature_dim += module[-3].out_channels
                    else:
                        feature_dim += module[-1].out_channels
                layer_to_hook = self.model.features[layer_idx]
                hook = layer_to_hook.register_forward_hook(get_intermediate_output_hook)
                self.layer_hooks.append(hook)
        elif "resnet" in self.model_name:
            for i, layer_idx in enumerate(self.extract_indices):
                module = getattr(self.model, f"layer{layer_idx}")
                feature_dim += module[-1].conv3.out_channels
                layer_to_hook = getattr(self.model, f"layer{layer_idx}")
                hook = layer_to_hook.register_forward_hook(get_intermediate_output_hook)
                self.layer_hooks.append(hook)
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor):
        """Extract features from the backbone model. 
        Args:
            x (torch.Tensor): Input image tensor, shape (B, C, H, W)
            extract_indices (list): List of indices to extract features from the backbone model.
        Returns:
            torch.Tensor: Extracted features, shape (B, C, H', W')
        Examples:
            >>> backbone = get_backbone_model("vgg19", [3, 8, 17, 26])
            >>> features = backbone.extract_features(x)  # x shape (B, 960, 64, 64)
        """
        
        if self.model_name in ["pdn_small", "pdn_medium"]:
            with torch.no_grad():
                features = self.model(x)
                features = nn.functional.interpolate(features, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
            return features
        
        if self.model_name == "identical":
            return x
            
        with torch.no_grad():
            _ = self.model(x)
        self.intermediate_outputs = BackboneModel.intermediate_cache
        self._reset_cache()
        
        for i, intermediate_output in enumerate(self.intermediate_outputs):
            self.intermediate_outputs[i] = nn.functional.interpolate(intermediate_output, size=(self.feature_res, self.feature_res), mode="bilinear", align_corners=False)
        features = torch.cat(self.intermediate_outputs, dim=1)

        return features

    def _reset_cache(self):
        BackboneModel.intermediate_cache = []

class BackboneWrapper(nn.Module):
    def __init__(self, backbone, scale_factors=None, target_size=None):
        super(BackboneWrapper, self).__init__()
        self.backbone = backbone
        self.scale_factors = scale_factors
        self.target_size = target_size
        assert scale_factors is not None or target_size is not None, "Either scale_factors or target_size must be provided"
        
        self.downsamples = nn.ModuleList()
        if scale_factors is not None:
            for scale_factor in scale_factors:
                self.downsamples.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear'))
        elif target_size is not None:
            for _ in range(len(self.backbone.outblocks)):
                self.downsamples.append(nn.Upsample(size=target_size, mode='bilinear'))
        
    def forward(self, x):
    
        out = self.backbone(x)
        if isinstance(out, dict):
            y = out["features"]
        else:
            y = out
        concat_y = torch.cat([downsample(y[i]) for i, downsample in enumerate(self.downsamples)], dim=1)
        return concat_y, y
