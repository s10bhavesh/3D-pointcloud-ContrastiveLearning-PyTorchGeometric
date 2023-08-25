from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


# Limit to 5000 samples, due to RAM restrictions
dataset = ShapeNet(
    root=".", categories=["Table", "Lamp", "Guitar", "Motorbike"]
).shuffle()[:5000]
# print("Number of Samples: ", len(dataset))
# print("Sample: ", dataset[0])

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# We're lucky and pytorch geometric helps us with pre-implemented transforms
# which can also be applied on the whole batch directly
augmentation = T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])

# Augmented data point
transformered = augmentation(sample)
