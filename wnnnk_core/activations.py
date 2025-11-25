import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch.nn.functional as F
import time

class ActivationsHandler:
  @staticmethod
  def get_activations(network, network_layer, output_handler, data_loader):
    activations = []
    labels = []
    indices = []

    def hook_fn(module, input, output):
        handled_output = output_handler(output)
        activations.append(handled_output.detach())

    hook = network_layer.register_forward_hook(hook_fn)
    network.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(next(network.parameters()).device)
            _ = network(data)
            labels.append(target)

            # Calculate dataset indices without modifying the loader
            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + data.size(0)
            batch_indices = torch.arange(start_idx, end_idx)
            indices.append(batch_indices)

    hook.remove()

    activations_tensor = torch.cat(activations, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    indices_tensor = torch.cat(indices, dim=0) #Expects dataloader to be set with shuffle = False to be able to recover indices

    return activations_tensor, labels_tensor, indices_tensor

  @staticmethod
  def get_multiple_activations(network, layer_handler_pairs, data_loader):
    # Extracts and returns activations from multiple specified layers in one forward pass; expects layer_handler_pairs as a dict of {name: (layer, handler_fn)} and a DataLoader with shuffle=False.
    activations_dict = {name: [] for name in layer_handler_pairs}
    labels = []
    indices = []

    hooks = []

    def make_hook(name, handler_fn):
        return lambda module, input, output: activations_dict[name].append(handler_fn(output).detach())

    for name, (layer, handler_fn) in layer_handler_pairs.items():
        hook = layer.register_forward_hook(make_hook(name, handler_fn))
        hooks.append(hook)

    network.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            _ = network(data)
            labels.append(target)

            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + data.size(0)
            batch_indices = torch.arange(start_idx, end_idx)
            indices.append(batch_indices)

    for hook in hooks:
        hook.remove()

    activations_dict = {name: torch.cat(acts, dim=0) for name, acts in activations_dict.items()}
    labels_tensor = torch.cat(labels, dim=0)
    indices_tensor = torch.cat(indices, dim=0)

    return activations_dict, labels_tensor, indices_tensor

  @staticmethod
  def get_activations_dataframe(network, network_layer, data_loader):
    activations, labels, _ = ActivationsHandler.get_activations(network, network_layer, lambda x: x, data_loader)
    feature_columns = [f"Feature {i+1}" for i in range(activations.shape[1])]
    df = pd.DataFrame(activations, columns=feature_columns)
    df['Label'] = labels
    return df

  @staticmethod
  def get_dreamed_activations(network, network_layer, output_handler, dreamed_image, label=-1):
      device = next(network.parameters()).device
      dummy_label = torch.tensor([label], device=device)
      dreamed_image = dreamed_image.to(device)

      dataset = torch.utils.data.TensorDataset(dreamed_image, dummy_label)
      loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

      activations_tensor, _, _ = ActivationsHandler.get_activations(
          network=network,
          network_layer=network_layer,
          output_handler=output_handler,
          data_loader=loader
      )

      return activations_tensor

class ActivationsAnalyzer:
  def __init__(self, activations, labels, indices):
    self.activations = activations
    self.labels = labels
    self.indices = indices

  def compute_PCA(self, n_components=2):
    scaler = StandardScaler()
    scaled_activations = scaler.fit_transform(self.activations.cpu().numpy())
    pca = PCA(n_components=n_components)
    return pca.fit_transform(scaled_activations)

  def compute_activation_ratios(self):
    num_classes = self.labels.unique().numel()
    num_neurons = self.activations.shape[1]
    activation_ratios = torch.zeros(num_classes, num_neurons)

    for class_idx, class_label in enumerate(self.labels.unique(sorted=True)):
        class_mask = self.labels == class_label
        class_acts = self.activations[class_mask]
        if class_acts.shape[0] == 0:
            continue
        active_counts = (class_acts > 0).sum(dim=0)
        activation_ratios[class_idx] = active_counts.float() / class_acts.shape[0]

    return activation_ratios

  def compute_filtered_activation_ratios(self, min_thresh, max_thresh):
    activation_ratios = self.compute_activation_ratios()
    mask = (activation_ratios >= min_thresh) & (activation_ratios <= max_thresh)
    filtered_activation_ratios = activation_ratios * mask
    return filtered_activation_ratios

  def compute_cosine_similarity_df(self, query_activation):
    # Normalize both tensors
    query_norm = F.normalize(query_activation, p=2, dim=1)
    with_timer_start = time.perf_counter()
    activations_norm = F.normalize(self.activations, p=2, dim=1)
    elapsed = time.perf_counter() - with_timer_start
    print(f"Elapsed: {elapsed:.3f} sec")

    # Compute cosine similarity
    similarities = torch.mm(activations_norm, query_norm.T).squeeze().cpu().numpy()

    # Prepare dataframe
    df = pd.DataFrame({
        'id': [i.item() for i in self.indices],
        'label': [l for l in self.labels],
        'cosine_similarity': similarities
    })

    return df
  
  def compute_cosine_similarity_per_channel_df(self, query_activations):
    """Computes per-channel cosine similarity between query and stored activations; expects shape [1, C, H, W]."""
    # Returns a DataFrame with cosine similarities per channel (columns 'ch_0', 'ch_1', ..., 'ch_C') and associated 'id' and 'label' per sample (rows).

    # Normalize across spatial dimensions (per channel)
    query_norm = F.normalize(query_activations, p=2, dim=(2, 3))  # [1, C, H, W]
    stored_norm = F.normalize(self.activations, p=2, dim=(2, 3))  # [N, C, H, W]

    # Compute cosine similarity per channel (mean over H, W)
    similarities = (stored_norm * query_norm).mean(dim=(2, 3))  # [N, C]

    # Create DataFrame with one column per channel
    df = pd.DataFrame(similarities.cpu().numpy(), columns=[f'ch_{i}' for i in range(similarities.shape[1])])
    df['id'] = self.indices.cpu().numpy()
    df['label'] = self.labels.cpu().numpy()

    return df

class ActivationsVisualizer:
    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
      self.analyzer = ActivationsAnalyzer(activations, labels)

    def plot_PCA(self):
      labels_np = self.analyzer.labels.cpu().numpy()
      data_2d = self.analyzer.compute_PCA()

      plt.figure(figsize=(8, 6))
      scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1],
                            c=labels_np, cmap='tab10', s=10, alpha=0.7)

      plt.colorbar(scatter, label='Digit Label')
      plt.xlabel("PCA Componente 1")
      plt.ylabel("PCA Componente 2")
      plt.title("PCA Activaciones")
      plt.grid(True)
      plt.show()

    def plot_activations_ratio_heatmap(self):
      activation_ratios = self.analyzer.compute_activation_ratios()

      plt.figure(figsize=(12, 6))
      sns.heatmap(activation_ratios.numpy(), cmap='viridis', xticklabels=range(self.analyzer.activations.shape[1]), yticklabels=self.analyzer.labels.unique().tolist())
      plt.xlabel("Índice de Neurona (Capa FC1)")
      plt.ylabel("Label de Clase")
      plt.title("Frecuencia de Activación Nueronal")
      plt.tight_layout()
      plt.show()

    def plot_filtered_activations_ratio_heatmap(self, min_thresh, max_thresh):
      data = self.analyzer.compute_filtered_activation_ratios(min_thresh, max_thresh).numpy()
      plt.figure(figsize=(12, 6))
      sns.heatmap(data, cmap='viridis', xticklabels=range(data.shape[1]), yticklabels=range(data.shape[0]),
                  cbar=True, linewidths=0.1, linecolor='gray', square=False)
      plt.xlabel("Índice de Neurona (Capa FC1)")
      plt.ylabel("Label de Clase")
      plt.title(f"Frecuencia de Activación Nueronal Filtrada (Mín: {min_thresh}, Máx: {max_thresh})")
      plt.tight_layout()
      plt.show()

    def plot_sample_activations_per_class(self, n_per_class = 5):
      activations, labels = self.analyzer.activations, self.analyzer.labels
      all_indices = []
      class_labels = labels.unique(sorted=True)

      for class_label in class_labels:
          class_indices = (labels == class_label).nonzero(as_tuple=True)[0]
          selected = class_indices[:n_per_class]
          all_indices.append(selected)

      ordered_indices = torch.cat(all_indices)
      ordered_activations = activations[ordered_indices]
      ordered_labels = labels[ordered_indices]

      data = ordered_activations.numpy()

      plt.figure(figsize=(12, 6))
      sns.heatmap(data, cmap='viridis', cbar=True, xticklabels=True, yticklabels=False)
      plt.title(f"Samples de Activaciones por Clase ({n_per_class} Samples por Clase)")
      plt.xlabel("Índice de Neurona")
      plt.ylabel("Label de Clase")

      # Add horizontal lines between classes + custom ticks
      yticks = []
      ytick_labels = []
      for i, class_label in enumerate(class_labels):
          pos = i * n_per_class
          plt.axhline(pos, color='white', linestyle='--', linewidth=0.5)
          yticks.append(pos + n_per_class / 2)
          ytick_labels.append(str(class_label.item()))

      plt.yticks(yticks, ytick_labels, rotation=0)
      plt.tight_layout()
      plt.show()

    def plot_mean_activations_per_class(self):
      activations = self.analyzer.activations  # shape: [N, num_neurons]
      labels = self.analyzer.labels
      unique_labels = sorted(labels.unique().tolist())
      num_classes = len(unique_labels)
      num_neurons = activations.shape[1]

      # Prepare matrix: rows = classes, cols = neurons
      mean_activations = torch.zeros((num_classes, num_neurons))

      for i, label in enumerate(unique_labels):
          mask = labels == label
          class_acts = activations[mask]
          mean_activations[i] = class_acts.mean(dim=0)

      # Plot heatmap
      plt.figure(figsize=(12, 6))
      sns.heatmap(mean_activations.numpy(), cmap='viridis',
                  xticklabels=range(num_neurons), yticklabels=unique_labels)

      plt.xlabel("Neuron Index")
      plt.ylabel("Class Label")
      plt.title("Mean Activations per Neuron and Class")
      plt.tight_layout()
      plt.show()

class QuerySimilarity:
  def __init__(self, activations, labels, indices):
    self.normalized_activations =  F.normalize(activations, p=2, dim=1)
    self.labels = labels
    self.indices = indices
  
  def compute_cosine_similarity(self, query_activation, query_label):
    query_norm = F.normalize(query_activation, p=2, dim=1)
    similarities = torch.mm(self.normalized_activations, query_norm.T).squeeze().cpu().numpy()

    return similarities, self.labels, self.indices, query_label