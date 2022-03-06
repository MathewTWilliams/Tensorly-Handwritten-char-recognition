# Author: Matt Williams
# Version: 2/27/2022
# Reference: https://github.com/ContinuumIO/gtc2017-numba

# stopped at CUDa device functionsw


from numba import vectorize, cuda
from timeit import Timer
import numpy as np
import math
import scipy.stats
from matplotlib import pyplot as plt


SQRT_2PI = np.float32((2*math.pi)**0.5)

@vectorize(['float32(float32, float32, float32)'], target = 'cuda')
def gaussian_pdf(x, mean, sigma):
    return math.exp(-0.5*((x-mean)/sigma)**2) / (sigma * SQRT_2PI)

def time_gaussian_pdf(): 
    x = np.random.uniform(-3, 3, 1000000).astype(np.float32)
    mean = np.float32(0.0)
    sigma = np.float32(1.0)

    norm_pdf = scipy.stats.norm
    t = Timer(lambda: norm_pdf.pdf(x, loc=mean, scale=sigma))
    print(t.timeit(number = 100))

    t = Timer(lambda: gaussian_pdf(x, mean, sigma))
    print(t.timeit(number = 100))


@cuda.jit(device=True)
def polar_to_cartesian(rho, theta): 
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y


@vectorize(['float32(float32, float32, float32, float32)'], target = "cuda")
def polar_distance(rho1, theta1, rho2, theta2): 
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)

    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def run_polar_distance(): 
    n = 1000000
    rho1 = np.random.uniform(0.5, 1.5, size = n).astype(np.float32)
    theta1 = np.random.uniform(-np.pi, np.pi, size = n).astype(np.float32)
    rho2 = np.random.uniform(0.5, 1.5, size = n).astype(np.float32)
    theta2 = np.random.uniform(-np.pi, np.pi, size = n).astype(np.float32)

    t = Timer(lambda: polar_distance(rho1, theta1, rho2, theta2))
    print(t.timeit(number = 100))





# using int64 will cause an error because the data types are larger than necessary
# 64 bit data types apparently don't work well on the GPU
@vectorize(['int32(int32, int32)'], target = 'cuda')
def add_ufunc(x,y): 
    return x + y

def gpu_u_funcs(): 
    a = np.array([1,2,3,4])
    b = np.array([10,20,30,40])
    b_col = b[:, np.newaxis]
    c = np.arange(4*4).reshape((4,4))

    print('a+b:\n', add_ufunc(a,b))
    print()
    print('b_col + c:\n', add_ufunc(b_col, c))

def time_it_gpu_u_funcs(): 
    b = np.array([10,20,30,40])[:, np.newaxis]
    c = np.arange(4*4).reshape((4,4))

    t = Timer(lambda: np.add(b,c))
    print(t.timeit(number = 1000000))

    # code below will generate a warning for GPU under-utilization
    t = Timer(lambda: add_ufunc(b, c))
    print(t.timeit(number = 1000000))



def u_funcs_np(): 
    #Universal function example
    a = np.array([1,2,3,4])
    b = np.array([10,20,30,40])

    # vector + vector, adds as expected
    print(np.add(a,b))

    #vector + a scalar, adds scalar to each vector element
    print(np.add(a, 100))
    
    #vector + matrix, must share a dimension
    #example if len(vector) == num_col(matrix), then the vector is added
    #to each row in the matrix
    c = np.arange(4*4).reshape((4,4))
    print(np.add(b,c))
    #example if len(vector) == num_row(matrix), then the vector is added
    #to each col in the matrix
    b_col = b[:, np.newaxis]
    print(np.add(b_col, c))


def run_ex2():
    n = 1000000
    noise = np.random.normal(size=n) * 3 
    pulses = np.maximum(np.sin(np.arange(n) / (n/23)) - 0.3, 0.0)
    waveform = ((pulses * 300) + noise).astype(np.int16)
    plt.plot(waveform)
    plt.show()

    @vectorize(['int16(int16, int16)'], target = "cuda")
    def zero_suppress(waveform_value, threshold): 
        return waveform_value if abs(waveform_value) > threshold else 0

    plt.plot(zero_suppress(waveform, 15.0))
    plt.show()
    
    




if __name__ == "__main__": 
    #u_funcs_np()
    #gpu_u_funcs()
    #time_it_gpu_u_funcs()
    #time_gaussian_pdf()
    #run_polar_distance()
    run_ex2()