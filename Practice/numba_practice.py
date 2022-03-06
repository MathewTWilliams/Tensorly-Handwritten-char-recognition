# Author: Matt Williams
# Version: 2/27/2022
# Reference: https://github.com/ContinuumIO/gtc2017-numba



from timeit import  Timer
from numba import jit
import math
import numpy as np

@jit
def hypot(x, y): 
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    t = t/x
    return x * math.sqrt(1 + t*t)


@jit(nopython = True)
def ex1(x,y,out): 
    for i in range(x.shape[0]):
        out[i] = hypot(x[i], y[i])

def run_ex1(): 
    in1 = np.arange(10, dtype=np.float64)
    in2 = 2 * in1 + 1
    out = np.empty_like(in1)
    print('in1:', in1)
    print('in2:', in2)
    ex1(in1, in2, out)
    print('out:', out)

    # This test will fail until you fix the ex1 function
    np.testing.assert_almost_equal(out, np.hypot(in1, in2))


def run_tests(): 

    t = Timer(lambda: hypot.py_func(3.0, 4.0))
    print(t.timeit(number = 1000000))

    t = Timer(lambda: hypot(3.0, 4.0))
    print(t.timeit(number = 1000000))

    t = Timer(lambda: math.hypot(3.0, 4.0))
    print(t.timeit(number = 1000000))


    print(hypot.inspect_types())



if __name__ == "__main__": 
    #run_tests()
    run_ex1()
    