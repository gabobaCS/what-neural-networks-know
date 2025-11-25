import torch
from torchvision import datasets, transforms
import nltk
from nltk.corpus import wordnet as wn

IMAGENET1K_VALIDATION_PATH = r"E:\datasets\ImageNet\ILSVRC2012_img_val"
IMAGENET21K_VALIDATION_PATH = r"E:\datasets\imagenet21k\imagenet21k_resized\imagenet21k_val"

def load_dataset_from_folder(
    data_dir: str,
    transform=None,
    target_transform=None
):
    """
    Loads an image dataset from a folder structured like:
        data_dir/class_x/xxx.png
        data_dir/class_y/xxy.png

    Args:
        data_dir (str): Path to dataset root directory.
        transform (callable, optional): Transform to apply to images.
        target_transform (callable, optional): Transform to apply to labels.

    Returns:
        torchvision.datasets.ImageFolder: Dataset instance
    """
    if transform is None:
        # Default transform: convert to tensor only
        transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform,
        target_transform=target_transform
    )
    return dataset


def imagenet1k_val_dataset(transform=None, target_transform=None):
    """
    Returns the ImageNet-1k validation dataset.
    """
    data_dir = IMAGENET1K_VALIDATION_PATH
    return datasets.ImageFolder(root=data_dir, transform=transform, target_transform=target_transform)

def imagenet21k_val_dataset(transform=None, target_transform=None):
    """
    Returns the ImageNet-21k validation dataset.
    """
    data_dir = IMAGENET21K_VALIDATION_PATH
    return datasets.ImageFolder(root=data_dir, transform=transform, target_transform=target_transform)


def synset_id_to_text(synset_id):
    """
    Converts a WordNet synset ID (e.g., 'n01440764') to a human-readable label using nltk's WordNet.
    Returns the lemma names and definition if found, else raises a ValueError.
    """
    # WordNet synset offset is the last 8 digits, pos is the first character
    pos = synset_id[0]
    offset = int(synset_id[1:])
    try:
        synset = wn.synset_from_pos_and_offset(pos, offset)
        return f"{', '.join(synset.lemma_names())}"
    except Exception:
        raise ValueError(f"Synset ID '{synset_id}' not found in NLTK WordNet.")