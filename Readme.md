# IHS-BIN

This is the repository of codes in the paper __Sketching the Krylov Subspace: Faster Computation of the Entire Ridge Regularization Path__. 

## Requirements
We require the installation of [RidgeSketch](https://github.com/facebookresearch/RidgeSketch) for numerical comparison. Please follow the instructions in [RidgeSketch](https://github.com/facebookresearch/RidgeSketch) to install it.


## Dataset

All dataset files can be downloaded from [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). To generate CIFAR10 with kernel matrix and MNIST with quadratic feature embedding, run the following lines:

```
python dataset-CIFAR-kernel.py
```

```
python dataset-MNIST.py
```



## Reproducing results

### over-determined examples

- randomly generated data

```
python cmp_ridge_regression.py --data_name random --n 2e4 --d 4e3 --m 1600 --noise_level 4e-2 --lbd_min 1e0 --lbd_max 1e2 --lbd_list_len 200 --IHS_BIN --plot --svd --cg --native --eigATA
```

- real-sim

```
python cmp_ridge_regression.py --data_name realsim --n 36000 --d 20958 --m 3e3 --lbd_min 1e2 --lbd_max 1e4 --lbd_list_len 100 --cg --IHS_BIN --data_dir DIRECTORY_TO_DATASETS --shuffle --eigAAT --native --svd
```

- avazu

```
python cmp_ridge_regression.py --data_name avazu-app --n 2e5 --d 5e4 --m 1e4 --lbd_min 1e0 --lbd_max 1e2 --lbd_list_len 100 --cg --IHS_BIN --data_dir DIRECTORY_TO_DATASETS --shuffle
```

- CIFAR10 kernel

```
python cmp_ridge_regression.py --data_name CIFAR10-kernel --n 25000 --d 25000 --m 10000 --lbd_min 1e-1 --lbd_max 1e1 --lbd_list_len 20 --IHS_BIN --cg --data_dir DIRECTORY_TO_DATASETS
```



### over-parameterized examples

- randomly generated data

```
python cmp_ridge_regression.py --data_name random --n 4e3 --d 2e4 --m 2400 --noise_level 2e-2 --lbd_min 1e0 --lbd_max 1e2 --lbd_list_len 200 --IHS_BIN_over --ihs_iter_max_over 7 --plot --svd_over --native_over --eigAAT --cg_over
```

- rcv1

```
python cmp_ridge_regression.py --data_name rcv1 --n 1e4 --d 47236 --m 3e3 --lbd_min 1e2 --lbd_max 1e4 --lbd_list_len 100 --IHS_BIN_over --cg_over --data_dir DIRECTORY_TO_DATASETS --svd_over --native_over --shuffle --eigATA
```

- gisette

```
python cmp_ridge_regression.py --data_name gisette --n 3e3 --d 5e3 --m 800 --lbd_min 1e5 --lbd_max 1e7 --lbd_list_len 100 --plot --cg_over --IHS_BIN_over --eigAAT --svd_over --native_over --shuffle --data_dir DIRECTORY_TO_DATASETS
```

- E2006-tfidf

```
python cmp_ridge_regression.py --data_name tfidf --n 8e3 --d 150360 --m 2000 --lbd_min 1e-1 --lbd_max 1e1 --lbd_list_len 200 --sparsity 1 --ihs_iter_max_over 7 --cg_over --IHS_BIN_over --native_over --data_dir DIRECTORY_TO_DATASETS --shuffle
```

- MNIST with quadratic feature embedding

```
python cmp_ridge_regression.py --data_name MNIST-kron --n 3e4 --d 608400 --m 1e4 --lbd_min 1e-1 --lbd_max 1e1 --lbd_list_len 20 --cg_over --IHS_BIN_over --sparsity 1 --data_dir DIRECTORY_TO_DATASETS
```

- synthetic data with decay spectral

```
python cmp_ridge_regression.py --data_name random-cluster --n 2e3 --d 4e2 --m 160 --noise_level 1e-1 --lbd_min 1e0 --lbd_max 1e2 --lbd_list_len 100 --IHS_BIN --ridge_sketch --repeat 5 --ihs_iter_max 7 --svd --cg --native --eigATA
```

- synthetic data with clustered spectral

```
python cmp_ridge_regression.py --data_name random --n 2e3 --d 4e2 --m 160 --noise_level 4e-1 --lbd_min 1e1 --lbd_max 1e3 --lbd_list_len 100 --IHS_BIN --ridge_sketch --repeat 5 --ihs_iter_max 6 --svd --cg --native --eigATA
```



## Plot figure

To plot the figure based on the output, run the following line

```
python plot_figure.py --path PATH_TO_OUTPUT_FOLDER
```

