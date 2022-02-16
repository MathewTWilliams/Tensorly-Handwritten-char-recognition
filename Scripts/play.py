import numpy as np
import tensorly as tl


def main(): 
    X = tl.tensor(np.arange(24).reshape((3,4,2)), dtype = tl.float32)
    print(X[:,:,0])
    print(X[:, :, 1])
    print("------------------------------------------------------")
    M = tl.tensor(np.arange(5*4).reshape((5,4)))
    print(M)
    print("------------------------------------------------------")

    Z = tl.tenalg.mode_dot(X,M, mode = 1)
    print(Z[:, :, 0])
    print(Z[:, :, 1])


if __name__ == "__main__": 
    main()