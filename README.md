# A simple and universal rotation equivariant point-cloud network

This repository is the official implementation of "A simple and universal rotation equivariant point-cloud network". 

## Requirements
This project is based on PyTorch 1.9.0, PyTorch Geometric library and pytorch3d 0.6.0.

To install dependencies:
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=10.2
(or conda install -c pytorch pytorch=1.10.0 torchvision cudatoolkit=11.3 )
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install pyg -c pyg -c conda-forge
```


To download the ModelNet40 dataset:
```
wget https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
unzip modelnet40_normal_resampled.zip
mv modelnet40_normal_resampled ./datasets/modelnet40_normal_resampled
```

## Input Arguments
* `--dataset`: Name of the dataset, all caps

* `--n_neighbors`: number of neighbors for our expanded ascending layer

* `--dynamic_knn`: Whether the neighbors for the KNN summation should be calculated base on
  the feature space (when True), or the point cloud data (when False).
  
* `--add_relus`: Whether to add activation layers

* `--eps`, `--share`, `--negative_slope`: the activation layer's hyperparameters

* `--k`: The highest order of representation

* `--add_linears`: Whether to add linear layers

* `--in_channel`: The number of channels we start with in our model

* `--u_shape`: Adds U-net style skip connection in between
  the different orders of representation of our ascending and descending layers

* `--z_align`: Assume all point clouds are aligned according to the z axis

* `--pool_type`: The pooling layer used

* `--drop_out`: The dropout used in our MLP

* `--additive_noise`: The maximal ratio of additive noise that is added to each example at train-time

* `--scale_noise`: The maximal scaling noise that is applied to each example in train-time

* `--SO3_train`: Changed the random rotation around the vertical axis to a random rotation
  
* `--seed`: a seed for reproducibility