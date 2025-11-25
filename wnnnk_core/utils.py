import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from ipython.display import display, HTML
from IPython.display import display, HTML

class Plotter:
    @staticmethod
    def show_gif_from_snapshots(snapshots, iteration_numbers, interval=150):
        """
        Display an animation (GIF-style) from a list of image tensors.

        Args:
            snapshots (List[Tensor]): List of tensors with shape [1, C, H, W] or [C, H, W].
            iteration_numbers (List[int]): Iteration number for each frame.
            interval (int): Delay between frames in milliseconds.

        Returns:
            IPython.display.HTML: The HTML representation of the animation.
        """
        fig, ax = plt.subplots()
        ims = []

        for tensor, iter_num in zip(snapshots, iteration_numbers):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            img = tensor.detach().cpu().permute(1, 2, 0).numpy()
            im = ax.imshow(img, animated=True)
            text = ax.text(1, 2, f"Iteration {iter_num}", color='white',
                           fontsize=10, backgroundcolor='black')
            ims.append([im, text])

        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        plt.axis('off')
        plt.close(fig)
        display(HTML(ani.to_jshtml()))

    @staticmethod
    def plot_rgb_image(image_tensor, title="Image"):
        """
        Plots a single unnormalized RGB image.

        Args:
            image_tensor (Tensor): shape [1, 3, H, W] or [3, H, W], values in [0, 1]
        """
        img = image_tensor.squeeze().detach().cpu()
        img = img.permute(1, 2, 0).clamp(0, 1)

        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

