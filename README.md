# ContrastiveLearning on 3D PointCloud Dataset using PyTorch & PyTorchGeometric

## Requirements:
```
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f $version

```

## Dataset
- Self-Supervised Representation Learning of Shapes
- Can be used for downstream tasks like clustering, fine-tuning, outlier-detection, ...
- Pointcloud = Set of unconnected nodes --> PyG
- ShapeNet Dataset - we just use a subset of classes and act like we didn't have labels
- I select 5k data points as otherwise I run out of memory on Colab

## Model
- Different choices for Point Cloud Feature-Learning layers (PointNet, PointNet++, EdgeConv, PointTransformer, etc.
- In PyTorch geometric we find an implementation of DynamicEdgeConv
- It uses the parameter k to detect the nearest neighbors which form a subgraph
- If you have many points, you can also sample a subset
- In the paper they use 4 layers, here we just have 2
- Implementation is inspired by [this PyG example](https://github.com/pyg-team/pytorch_geometric/blob/a6e349621d4caf8b381fe58f8e57109b2d0947ed/examples/dgcnn_segmentation.py)
- We only apply augmentations during training

## Training
- We use InfoNCE / NT-Xent Loss implemented in pytorch metric learning library
- Temperature allows to balance the similarity measure (make it more peaked)
- Typical values are around 0.1 / 0.2


