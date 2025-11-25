import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import collections

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook to compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2


def clip(image_tensor, use_fp16=False):
    '''
    Adjust the input based on mean and variance of ImageNet dataset.
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    '''
    Convert normalized tensor back to [0,1] range
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)
    
    return image_tensor


def deep_inversion(net, bs=10, iterations=2000, lr=0.2, r_feature=0.05, 
                  tv_l1=0.0, tv_l2=0.0001, l2_scale=0.00001, 
                  main_loss_multiplier=1.0, jitter=30, class_idx=None,
                  use_fp16=False, random_labels=False):
    """
    Deep Inversion implementation for data-free knowledge transfer
    
    Args:
        net: Pre-trained model with BatchNorm layers
        bs: Batch size
        iterations: Number of optimization iterations  
        lr: Learning rate
        r_feature: Coefficient for feature distribution regularization
        tv_l1: Coefficient for total variation L1 loss
        tv_l2: Coefficient for total variation L2 loss
        l2_scale: L2 penalty weight on pixels
        main_loss_multiplier: Coefficient for main classification loss
        jitter: Amount of random shift applied to image at every iteration
        class_idx: Target class index (None for random classes)
        use_fp16: Use half precision
        random_labels: Use random labels instead of specific classes
    """
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model to device
    net = net.to(device)
    net.eval()
    
    # Initialize inputs - normalized for ImageNet
    if use_fp16:
        inputs = torch.randn((bs, 3, 224, 224), requires_grad=True, device=device, dtype=torch.half)
    else:
        inputs = torch.randn((bs, 3, 224, 224), requires_grad=True, device=device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam([inputs], lr=lr, betas=[0.5, 0.9], eps=1e-8)
    criterion = nn.CrossEntropyLoss()
    
    # Create hooks for all BatchNorm layers
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    
    print(f"Found {len(loss_r_feature_layers)} BatchNorm layers")
    
    # Set target labels
    if random_labels:
        targets = torch.randint(0, 1000, (bs,)).to(device)
    elif class_idx is not None:
        targets = torch.tensor([class_idx] * bs).to(device)
    else:
        # Use a variety of classes
        targets = torch.tensor([i % 1000 for i in range(bs)]).to(device)
    
    print(f"Target classes: {targets.cpu().numpy()}")
    
    # Optimization loop
    for iteration in range(iterations):
        # Apply random jitter for better results
        off1 = random.randint(-jitter, jitter)
        off2 = random.randint(-jitter, jitter)
        inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(inputs_jit)
        
        # Main classification loss - cross entropy
        loss_ce = criterion(outputs, targets)
        loss = main_loss_multiplier * loss_ce
        
        # Feature distribution regularization loss (from BN statistics)
        if len(loss_r_feature_layers) > 0:
            loss_r_feature = sum([mod.r_feature for mod in loss_r_feature_layers])
            loss += r_feature * loss_r_feature
        
        # Image prior losses (total variation)
        loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
        loss += tv_l1 * loss_var_l1
        loss += tv_l2 * loss_var_l2
        
        # L2 penalty on input pixels
        loss += l2_scale * torch.norm(inputs_jit, 2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Clip to stay in reasonable range
        inputs.data = clip(inputs.data, use_fp16)
        
        # Print progress
        if iteration % 200 == 0:
            print(f"Iteration {iteration:4d}: Total Loss {loss.item():.4f}, "
                  f"CE Loss {loss_ce.item():.4f}, "
                  f"R_feature {loss_r_feature.item() if len(loss_r_feature_layers) > 0 else 0:.4f}")
    
    # Clean up hooks
    for hook in loss_r_feature_layers:
        hook.close()
    
    # Denormalize for visualization
    final_images = denormalize(inputs.clone(), use_fp16)
    
    return final_images.detach()
