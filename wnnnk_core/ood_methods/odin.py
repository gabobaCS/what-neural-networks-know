import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_odin_score(inputs, model, forward_func, method_args):
    """
    Compute ODIN scores following the paper implementation.
    
    Args:
        inputs: torch.Tensor (batch_size, C, H, W)
        model: torch.nn.Module
        forward_func: function that takes (inputs, model) and returns logits
        method_args: dict with 'temperature' and 'magnitude' keys
    
    Returns:
        scores: numpy array (batch_size,)
    """
    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.autograd.Variable(inputs, requires_grad=True)
    outputs = forward_func(inputs, model)
    
    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    
    # Using temperature scaling
    outputs = outputs / temper
    
    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    
    # Forward pass on perturbed inputs
    with torch.no_grad():
        outputs = forward_func(tempInputs, model)
    
    outputs = outputs / temper
    
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    
    scores = np.max(nnOutputs, axis=1)
    
    return scores


def forward_func(inputs, model):
    """Simple forward function."""
    return model(inputs)


def compute_auroc(id_scores, ood_scores):
    """Compute AUROC."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    return roc_auc_score(labels, scores) * 100


def compute_fpr95(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx] * 100


class FlatImageDataset(Dataset):
    """Dataset for loading images from flat directory structure."""
    def __init__(self, root_dir, transform=None, extensions=('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.root_dir.rglob(f'*{ext}')))
        
        self.image_paths.sort()
        print(f"  Found {len(self.image_paths)} images")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Dummy label


def evaluate_odin_ood(model, id_loader, ood_loaders_dict, temperature=1000, magnitude=0.0014):
    """
    ODIN OOD detection evaluation.

    Args:
        model: Trained neural network model
        id_loader: DataLoader for in-distribution data
        ood_loaders_dict: Dictionary mapping OOD dataset names to their DataLoaders
        temperature: Temperature scaling parameter
        magnitude: Input perturbation magnitude (epsilon)

    Returns:
        Dictionary containing FPR95 and AUROC metrics for each OOD dataset
    """
    print("="*70)
    print("ODIN OOD Detection")
    print(f"Temperature: {temperature}")
    print(f"Magnitude (epsilon): {magnitude}")
    print("="*70)
    
    model.eval()
    
    method_args = {
        'temperature': temperature,
        'magnitude': magnitude
    }
    
    # Compute ODIN scores for ID data
    print(f"\n[Step 1/2] Computing ODIN scores for ID data...")
    id_scores = []
    
    for batch_idx, (inputs, _) in enumerate(id_loader):
        inputs = inputs.to(device)
        batch_scores = get_odin_score(inputs, model, forward_func, method_args)
        id_scores.append(batch_scores)
        
        if (batch_idx + 1) % 50 == 0:
            processed = (batch_idx + 1) * id_loader.batch_size
            total = len(id_loader.dataset)
            print(f"  Progress: {processed}/{total} ({100*processed/total:.1f}%)")
    
    id_scores = np.concatenate(id_scores)
    print(f"\nID ODIN scores: min={id_scores.min():.4f}, max={id_scores.max():.4f}, mean={id_scores.mean():.4f}")
    
    # Evaluate on each OOD dataset
    print(f"\n[Step 2/2] Evaluating on OOD datasets...")
    print("="*70)
    
    results = {}
    
    for ood_idx, (ood_name, ood_loader) in enumerate(ood_loaders_dict.items(), 1):
        print(f"\nOOD Dataset {ood_idx}/{len(ood_loaders_dict)}: {ood_name}")
        
        ood_scores = []
        for batch_idx, (inputs, _) in enumerate(ood_loader):
            inputs = inputs.to(device)
            batch_scores = get_odin_score(inputs, model, forward_func, method_args)
            ood_scores.append(batch_scores)
            
            if (batch_idx + 1) % 50 == 0:
                processed = (batch_idx + 1) * ood_loader.batch_size
                total = len(ood_loader.dataset)
                print(f"  Progress: {processed}/{total} ({100*processed/total:.1f}%)")
        
        ood_scores = np.concatenate(ood_scores)
        print(f"  OOD ODIN scores: min={ood_scores.min():.4f}, max={ood_scores.max():.4f}, mean={ood_scores.mean():.4f}")
        
        # Compute metrics
        auroc = compute_auroc(id_scores, ood_scores)
        fpr95 = compute_fpr95(id_scores, ood_scores)
        
        results[ood_name] = {
            'fpr95': fpr95,
            'auroc': auroc,
            'num_samples': len(ood_scores)
        }
        
        print(f"  ✓ Results: FPR95={fpr95:.2f}%, AUROC={auroc:.2f}%")
    
    # Compute averages
    avg_fpr95 = np.mean([r['fpr95'] for r in results.values()])
    avg_auroc = np.mean([r['auroc'] for r in results.values()])
    results['average'] = {'fpr95': avg_fpr95, 'auroc': avg_auroc}
    
    # Print summary
    print(f"\n{'='*70}")
    print("ODIN OOD Detection Results")
    print(f"{'='*70}")
    print(f"{'OOD Dataset':<20} {'FPR95 ↓':>12} {'AUROC ↑':>12} {'Samples':>12}")
    print(f"{'-'*70}")
    
    for ood_name, metrics in results.items():
        if ood_name == 'average':
            continue
        print(f"{ood_name:<20} {metrics['fpr95']:>11.2f}% {metrics['auroc']:>11.2f}% {metrics['num_samples']:>12}")
    
    print(f"{'-'*70}")
    print(f"{'Average':<20} {avg_fpr95:>11.2f}% {avg_auroc:>11.2f}% {'-':>12}")
    print(f"{'='*70}\n")
    
    return results


model = torchvision.models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

print("\nLoading ID dataset...")
id_dataset_path = r"E:\datasets\ImageNet\ILSVRC2012_img_val"
id_dataset = ImageFolder(id_dataset_path, transform=transform)
id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
print(f"ID dataset: {len(id_dataset)} samples")

print("\nLoading OOD datasets...")
ood_datasets = {
    "iNaturalist": r"E:\datasets\KNN-OOD\iNaturalist\images",
    "SUN": r"E:\datasets\KNN-OOD\SUN\images",
    "Places": r"E:\datasets\KNN-OOD\Places\images",
    "Textures": r"E:\datasets\KNN-OOD\Textures\images",
}

ood_loaders_dict = {}
for ood_name, ood_path in ood_datasets.items():
    print(f"\n{ood_name}:")
    try:
        dataset = ImageFolder(ood_path, transform=transform)
    except:
        dataset = FlatImageDataset(ood_path, transform=transform)

    ood_loaders_dict[ood_name] = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

results = evaluate_odin_ood(
    model=model,
    id_loader=id_loader,
    ood_loaders_dict=ood_loaders_dict,
    temperature=1000,
    magnitude=0.0014
)

print("\nFinal Summary:")
for ood_name in ood_datasets.keys():
    print(f"  {ood_name}: FPR95={results[ood_name]['fpr95']:.2f}%, AUROC={results[ood_name]['auroc']:.2f}%")
print(f"\nAverage: FPR95={results['average']['fpr95']:.2f}%, AUROC={results['average']['auroc']:.2f}%")