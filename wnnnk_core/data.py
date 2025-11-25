import random
import torch
from collections import defaultdict
import numpy as np

class DataHandler:
  @staticmethod
  def get_loader_with_samples_per_class(dataset, samples_per_class, batch_size = 64, shuffle=False):
    # Given a dataset and desired samples per class, returns a balanced dataset with the desired number of samples per class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        selected = random.sample(indices, min(samples_per_class, len(indices)))
        selected_indices.extend(selected)

    random.shuffle(selected_indices)
    subset = torch.utils.data.Subset(dataset, selected_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loader
  

  @staticmethod
  def get_loader_with_samples_per_class_fast(
      dataset,
      samples_per_class,
      batch_size=64,
      shuffle=False,
      num_workers=8,
      pin_memory=True,
      prefetch_factor=4,
      persistent_workers=True,
      generator=None,
  ):
      # 1) Grab labels without touching images
      labels = np.asarray(dataset.targets)

      # 2) Build class->indices map in O(N) without decoding images
      class_to_indices = defaultdict(list)
      for idx, y in enumerate(labels):
          class_to_indices[int(y)].append(idx)

      # 3) Pick k per class (no image I/O)
      selected_indices = []
      for _, idxs in class_to_indices.items():
          if len(idxs) <= samples_per_class:
              selected_indices.extend(idxs)
          else:
              # numpy is faster than random.sample on large lists
              picked = np.random.choice(idxs, size=samples_per_class, replace=False)
              selected_indices.extend(picked.tolist())

      # 4) Shuffle the final index list if requested
      if shuffle:
          rng = np.random.default_rng() if generator is None else np.random.default_rng(generator.initial_seed())
          rng.shuffle(selected_indices)

      subset = torch.utils.data.Subset(dataset, selected_indices)

      # 5) DataLoader perf knobs
      loader = torch.utils.data.DataLoader(
          subset,
          batch_size=batch_size,
          shuffle=False,              # we already shuffled indices above if desired
          num_workers=num_workers,
          pin_memory=pin_memory,
          prefetch_factor=prefetch_factor if num_workers > 0 else None,
          persistent_workers=persistent_workers if num_workers > 0 else False,
      )
      return loader


  @staticmethod
  def filter_dataset_by_labels(dataset, label_list):
    # Given a dataset and a list of labels, returns a new dataset with only the samples corresponding to the provided labels
    filtered_indices = [
        idx for idx, (_, label) in enumerate(dataset)
        if label in label_list
    ]
    return torch.utils.data.Subset(dataset, filtered_indices)

  @staticmethod
  def get_random_sample_by_class(dataset, class_index):
    # Returns a random sample of the specified class
    matching_indices = [i for i, (_, label) in enumerate(dataset) if label == class_index]
    if not matching_indices:
        raise ValueError(f"No samples found for class index {class_index}.")
    selected_index = random.choice(matching_indices)
    return dataset[selected_index]

class DataFeeder:
  @staticmethod
  def test_accuracy(network, data_loader, return_count_of_correct = False):
    network.eval()
    correct = 0
    with torch.no_grad():
      for data, target in data_loader:
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    if return_count_of_correct:
      return correct.item()
    return correct.item() / len(data_loader.dataset)

  @staticmethod
  def predict_batch(network, data_loader):
    network.eval()
    predictions = []
    with torch.no_grad():
      for data, target in data_loader:
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1]
    return pred

  @staticmethod
  def predict(network, data):
    network.eval()
    with torch.no_grad():
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1]
    return pred


