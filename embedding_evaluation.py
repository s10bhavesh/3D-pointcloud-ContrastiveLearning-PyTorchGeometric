from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Get sample batch
sample = next(iter(data_loader))

# Get representations
h = model(sample.to(device), train=False)
h = h.cpu().detach()
labels = sample.category.cpu().detach().numpy()

# Get low-dimensional t-SNE Embeddings
h_embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(
    h.numpy()
)

# Plot
ax = sns.scatterplot(
    x=h_embedded[:, 0], y=h_embedded[:, 1], hue=labels, alpha=0.5, palette="tab10"
)

# Add labels to be able to identify the data points
annotations = list(range(len(h_embedded[:, 0])))


def label_points(x, y, val, ax):
    a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point["x"] + 0.02, point["y"], str(int(point["val"])))


label_points(
    pd.Series(h_embedded[:, 0]),
    pd.Series(h_embedded[:, 1]),
    pd.Series(annotations),
    plt.gca(),
)


import numpy as np


def sim_matrix(a, b, eps=1e-8):
    """
    Eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


similarity = sim_matrix(h, h)
max_indices = torch.topk(similarity, k=2)[1][:, 1]
max_vals = torch.topk(similarity, k=2)[0][:, 1]

# Select index
idx = 17
similar_idx = max_indices[idx]
print(f"Most similar data point in the embedding space for {idx} is {similar_idx}")
