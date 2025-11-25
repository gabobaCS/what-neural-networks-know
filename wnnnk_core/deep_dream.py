import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import warnings


class Penalizer:
    """
    A class that applies regularization penalties to images to improve deep dream quality.
    
    Penalties help prevent generated images from becoming too noisy or unrealistic
    by encouraging smoothness, edge preservation, and reasonable pixel values.
    """
    
    def __init__(self, total_variation_weight: float = 1.0, 
                 l1_edge_weight: float = 1.0, 
                 l2_pixel_weight: float = 1.0):
        """
        Initialize the penalizer with configurable weights for different penalty types.
        
        Args:
            total_variation_weight: Weight for total variation penalty (smoothness)
            l1_edge_weight: Weight for L1 edge penalty (edge preservation)
            l2_pixel_weight: Weight for L2 pixel penalty (pixel value control)
        """
        self.tv_weight = total_variation_weight
        self.l1_weight = l1_edge_weight
        self.l2_weight = l2_pixel_weight

    def total_variation_penalty(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation penalty to encourage image smoothness.
        
        Args:
            img: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B] containing per-image TV penalties
        """
        if img.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {img.ndim}D")
            
        diff_h = (img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
        diff_w = (img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
        return diff_h + diff_w

    def l1_edge_penalty(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 edge penalty to preserve important image edges.
        
        Args:
            img: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B] containing per-image L1 edge penalties
        """
        if img.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {img.ndim}D")
            
        diff_x = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
        diff_y = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
        return diff_x + diff_y

    def l2_pixel_penalty(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 pixel penalty to prevent pixel values from becoming too large.
        
        Args:
            img: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B] containing per-image L2 pixel penalties
        """
        if img.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {img.ndim}D")
            
        return img.pow(2).mean(dim=(1, 2, 3))

    def compute_penalties(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined regularization penalties for all images in the batch.
        
        Args:
            img: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B] containing total per-image penalties
        """
        if img.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {img.ndim}D")
            
        penalty = torch.zeros(img.shape[0], device=img.device, dtype=img.dtype)
        
        if self.tv_weight > 0:
            penalty = penalty + self.tv_weight * self.total_variation_penalty(img)
        if self.l1_weight > 0:
            penalty = penalty + self.l1_weight * self.l1_edge_penalty(img)
        if self.l2_weight > 0:
            penalty = penalty + self.l2_weight * self.l2_pixel_penalty(img)
            
        return penalty


class Dreamer:
    """
    A class for performing deep dream optimization on neural network activations.
    
    Deep dream maximizes the activation of specific neurons in a model layer
    by iteratively optimizing input images using gradient ascent.
    """
    
    def __init__(self, model: torch.nn.Module, 
                 model_layer: torch.nn.Module,
                 penalizer: Penalizer,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None,
                 memory_check_freq: int = 10,
                 enable_memory_monitoring: bool = True):
        """
        Initialize the Dreamer with a model and configuration.
        
        Args:
            model: The neural network model to dream on
            model_layer: The specific layer whose activations to maximize
            penalizer: Penalizer instance for regularization
            mean: Dataset mean for normalization (optional)
            std: Dataset standard deviation for normalization (optional)
            device: Device to run computations on (auto-detected if None)
            memory_check_freq: How often to check GPU memory (every N iterations)
            enable_memory_monitoring: Whether to enable GPU memory monitoring
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_layer = model_layer
        self.penalizer = penalizer
        self.memory_check_freq = memory_check_freq
        self.enable_memory_monitoring = enable_memory_monitoring
        
        if mean is not None:
            self.mean = mean.to(self.device)
        else:
            self.mean = None
            
        if std is not None:
            self.std = std.to(self.device)
        else:
            self.std = None

        self._last_acts = None
        # Hook will be registered per dream_batch call, not in __init__

    def _hook_fn(self, module: torch.nn.Module, inputs: Tuple, output: torch.Tensor) -> None:
        """Hook function to capture activations from the target layer."""
        self._last_acts = output

    def _normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Normalize images using dataset statistics if provided.
        
        Args:
            imgs: Input images of shape [B, C, H, W]
            
        Returns:
            Normalized images
        """
        if self.mean is not None and self.std is not None:
            return (imgs - self.mean) / self.std
        return imgs

    def _check_shapes(self, base_images: torch.Tensor, target_neurons: torch.Tensor) -> None:
        """
        Validate input tensor shapes and dimensions.
        
        Args:
            base_images: Input images tensor
            target_neurons: Target neuron indices tensor
            
        Raises:
            ValueError: If shapes are invalid
        """
        if base_images.ndim != 4:
            raise ValueError(f"base_images must be 4D [B, C, H, W], got {base_images.ndim}D")
            
        if target_neurons.ndim != 1:
            raise ValueError(f"target_neurons must be 1D [B], got {target_neurons.ndim}D")
            
        if target_neurons.shape[0] != base_images.shape[0]:
            raise ValueError(f"Batch size mismatch: {base_images.shape[0]} images vs {target_neurons.shape[0]} neurons")

    def _check_gpu_memory(self) -> None:
        """Print current GPU memory usage for monitoring."""
        if not self.enable_memory_monitoring or not torch.cuda.is_available():
            return
            
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)

            print(f"GPU Memory - Total: {total_mem/1024**2:.1f} MB, "
                  f"Allocated: {allocated/1024**2:.1f} MB, "
                  f"Reserved: {reserved/1024**2:.1f} MB")
        except Exception as e:
            warnings.warn(f"Could not check GPU memory: {e}")

    def dream_batch(self, 
                   base_images: torch.Tensor, 
                   target_neurons: torch.Tensor, 
                   iterations: int = 100, 
                   lr: float = 0.05, 
                   clamp_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Perform deep dream optimization on a batch of images.
        
        Args:
            base_images: Input images of shape [B, C, H, W]
            target_neurons: Target neuron indices of shape [B]
            iterations: Number of optimization iterations
            lr: Learning rate for optimization
            clamp_range: Range to clamp pixel values (min, max)
            
        Returns:
            Tuple of (final_dream_images, loss_histories)
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If activations cannot be captured
        """
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if clamp_range[0] >= clamp_range[1]:
            raise ValueError("clamp_range must be (min, max) with min < max")
        
        self.model.eval()

        B, C, H, W = base_images.shape
        base_images = base_images.to(self.device)
        target_neurons = target_neurons.to(self.device).long()
        
        self._check_shapes(base_images, target_neurons)

        dream = base_images.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([dream], lr=lr)

        loss_histories = [[] for _ in range(B)]
        
        handle = self.model_layer.register_forward_hook(self._hook_fn)
        
        try:
            for it in range(1, iterations + 1):
                
                if (self.enable_memory_monitoring and torch.cuda.is_available() and 
                    it % self.memory_check_freq == 0):
                    self._check_gpu_memory()
                    

                optimizer.zero_grad(set_to_none=True)

                normed = self._normalize(dream)
                _ = self.model(normed)

                acts = self._last_acts

                if acts is None:
                    raise RuntimeError("Hook did not capture activations. Ensure model_layer is correct.")

                gathered = acts[torch.arange(B, device=acts.device), target_neurons]
                obj_terms = -gathered

                penalties = self.penalizer.compute_penalties(dream)

                per_image_loss = obj_terms + penalties
                total_loss = per_image_loss.sum()
                total_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    dream.data.clamp_(clamp_range[0], clamp_range[1])

                loss_values = per_image_loss.detach().tolist()
                for i in range(B):
                    loss_histories[i].append(float(loss_values[i]))

        finally:
            handle.remove()
            self._last_acts = None

        final_images = dream.detach()
        return final_images, loss_histories

    def dream_neuron(self, 
                    base_image: torch.Tensor, 
                    target_neuron: Union[int, torch.Tensor] = 0, 
                    iterations: int = 100, 
                    lr: float = 0.05, 
                    clamp_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[torch.Tensor, List[float]]:
        """
        Convenience method for dreaming on a single image.
        
        Wraps the image in a batch dimension and calls dream_batch internally.
        
        Args:
            base_image: Input image of shape [C, H, W] or [1, C, H, W]
            target_neuron: Target neuron index (integer or tensor)
            iterations: Number of optimization iterations
            lr: Learning rate for optimization
            clamp_range: Range to clamp pixel values (min, max)
            
        Returns:
            Tuple of (final_dream_image, loss_history)
        """
        if base_image.ndim == 3:
            base_image = base_image.unsqueeze(0)
        elif base_image.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {base_image.ndim}D")
        
        if isinstance(target_neuron, int):
            target_neurons = torch.tensor([target_neuron])
        elif isinstance(target_neuron, torch.Tensor):
            target_neurons = target_neuron
        else:
            raise ValueError(f"target_neuron must be int or torch.Tensor, got {type(target_neuron)}")
            
        final_images, loss_histories = self.dream_batch(
            base_image, target_neurons, iterations, lr, clamp_range
        )
        
        return final_images.squeeze(0), loss_histories[0]

    def __del__(self):
        """Cleanup hook when object is destroyed."""
        # No hook to clean up since hooks are registered per dream_batch call
        pass





