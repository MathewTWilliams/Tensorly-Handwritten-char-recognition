# Author: Matt Williams
# Version: 2/27/2022
# Reference: https://github.com/JeanKossaifi/tensorly-notebooks

from tensorly.decomposition import parafac, tucker
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from tensorly.base import tensor_to_vec, partial_tensor_to_vec
from tensorly.datasets.synthetic import gen_image
from tensorly.regression import CPRegressor
import tensorly.backend as T


tl.set_backend("numpy")


def low_rank_tensor_regression(): 
    image_height = 28
    image_width = 28

    rng = tl.check_random_state(1)
    X = T.tensor(rng.normal(size=(1000, image_height, image_width), loc = 0, scale = 1))
    print(X.shape)
    Y = partial_tensor_to_vec(X, skip_begin=1)
    print(Y.shape)
    Z = tensor_to_vec(X)
    print(Z.shape)

    weight_img = gen_image(region="swiss", image_height=image_height, image_width=image_width)
    weight_img = T.tensor(weight_img)
    print(tensor_to_vec(weight_img).shape)

    # the true labels are obtained by taking the dot product between the true regression weights and the input tensors
    y = T.dot(partial_tensor_to_vec(X, skip_begin=1), tensor_to_vec(weight_img))


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(T.to_numpy(weight_img), cmap=plt.cm.OrRd, interpolation = "nearest")
    ax.set_axis_off()
    ax.set_title('True regression weights')
    plt.show()


    estimator = CPRegressor(weight_rank=2, tol=10e-7, n_iter_max=100, reg_W=1, verbose=0)
    estimator.fit(X, y)
    estimator.predict(X)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(T.to_numpy(estimator.weight_tensor_), cmap=plt.cm.OrRd, interpolation = "nearest")
    ax.set_axis_off()
    ax.set_title("Learned regression weights")
    plt.show()

def tucker_decomposition(): 
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    core, factors = tucker(X, rank=(3,4,2))
    print(core)
    for factor in factors: 
        print(factor)


def cp_decomposition(): 
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    weights, factors = parafac(X, rank = 2)
    print("weights:", weights)
    print("factors:")
    for factor in factors: 
        print(factor)
    print("Length of factors:", len(factors))


def tensor_manipulation_example(): 
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    print("-----------------------------------------")
    print("regular print of Rank 3 Tensor")
    print(X)
    print("-----------------------------------------")
    print("Viewing Horizontal slices of rank 3 tensor")
    print("-----------------------------------------")
    for i in range(3): 
        print(X[i, :, :,])
    print("-----------------------------------------")
    print("Viewing Vertical slices or rank 3 tensor")
    print("-----------------------------------------")
    for i in range(4): 
        print(X[:, i, :,])
    print("-----------------------------------------")
    print("Viewing Frontal slices or rank 3 tensor")
    print("-----------------------------------------")
    for i in range(2): 
        print(X[:, :, i,])



def tensor_unfolding(): 
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    print(X)
    for i in range(3): 
        print("-----------------------------------------")
        X = tl.unfold(X, mode = i)
        print(X)
        print("-----------------------------------------")
        X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)

    print("Folding back to tensor after unfolding on mode = 2")
    X = tl.fold(X, mode = 2, shape = (3,4,2))
    print(X)

def parafac_example(): 
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    weights, factors = parafac(X, rank = 2)
    print(weights)
    for factor in factors: 
        print(factor)
    print(len(factors))


def n_mode_product(): 
    '''Also known as tensor contraction. '''
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype=tl.float32)
    M = tl.tensor(np.arange(20).reshape(5,4))
    

    # need to use mode = because that is the only matching size
    # in both the tensor and the matrix.
    # When multiplying by the matrix, we must use the tranpose
    res = tl.tenalg.mode_dot(X, M, mode = 1)
    print("Multiplying Tensor with matrix")
    print(res.shape)
    print("------------------------------")
    print(res)
    print("------------------------------")

    V = tl.tensor(np.arange(4))
    res = tl.tenalg.mode_dot(X, V, mode = 1)
    print("Multiplying Tensor with vector")
    print(res.shape)
    print("------------------------------")
    print(res)
    print("------------------------------")

    # we can get the columns of the result using the dot product with 
    # each of the frontal slices.
    print(X[:, :, 0] @ V)

if __name__ == "__main__": 
    #tensor_manipulation_example()
    #tensor_unfolding()
    #n_mode_product()
    #cp_decomposition()
    #tucker_decomposition()

    low_rank_tensor_regression()