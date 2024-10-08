32 control points             |  64 control points | 256 control points             |  512 control points
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](./assets/digits_32.png)  |  ![](./assets/digits_64.png)  |  ![](./assets/digits_256.png)  |  ![](./assets/digits_512.png)

# lsp-python

lsp-python is a lightweight python implementation of the Least Square Projection (LSP) dimensionality reduction technique using sklearn style API.

The implementation is based on the paper "Least Square Projection: A Fast High-Precision Multidimensional Projection Technique and Its Application to Document Mapping", which can be cited using:

```
@ARTICLE{4378370,
  author={Paulovich, Fernando V. and Nonato, Luis G. and Minghim, Rosane and Levkowitz, Haim},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Least Square Projection: A Fast High-Precision Multidimensional Projection Technique and Its Application to Document Mapping}, 
  year={2008},
  volume={14},
  number={3},
  pages={564-575},
  keywords={Least squares methods;Multidimensional systems;Data visualization;Least squares approximation;Data analysis;Computational geometry;Testing;Text processing;Data mining;Demography;Multivariate visualization;Data and knowledge visualization;Information visualization;Multivariate visualization;Data and knowledge visualization;Information visualization},
  doi={10.1109/TVCG.2007.70443}}
```

A small working example can be found in [tests/iris_example.py](tests/iris_example.py) and [tests/digits_example.py](tests/digits_example.py).

## Installation
The library currently only supports Python 3.11.

### Dependencies
The library depends on the following packages:
- numpy
- scikit-learn
- matplotlib

### Pip
The library can be installed using pip:

```bash
pip install lsp-python
```