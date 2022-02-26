from unicodedata import decomposition
from tensorly.decomposition import parafac, tucker
import numpy as np
import tensorly as tl

tl.set_backend("numpy")

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
    tucker_decomposition()
