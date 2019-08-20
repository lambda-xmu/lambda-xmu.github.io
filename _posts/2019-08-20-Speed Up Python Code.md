---
layout:     post
title:      2019-08-20-Speed-Up-Python-Code
subtitle:   "Nuba Tutorial"
date:       2018-08-20 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Python
- Numba
---

The Python library [Numba](http://numba.pydata.org/numba-doc/latest/index.html) gives us an easy way around that challenge — free speed ups without having to write any code other than Python!

## Introducing Numba
Numba is a compiler library that transforms Python code into optimised machine code. With this transformation, Numba can make numerical algorithms that are written in Python approach the speeds of C code.

## Speeding up Python loops
When we see a function that contains a loop written in pure Python, it’s usually a good sign that numba can help.
``` python
from numba import jit
import time
import random

num_loops = 50
len_of_list = 100000

@jit(nopython=True) # or @njit
def insertion_sort(arr):

    for i in range(len(arr)):
        cursor = arr[i]
        pos = i

        while pos > 0 and arr[pos - 1] > cursor:
            # Swap the number down the list
            arr[pos] = arr[pos - 1]
            pos = pos - 1
        # Break and do the final swap
        arr[pos] = cursor

    return arr

start = time.time()
list_of_numbers = list()
for i in range(len_of_list):
    num = random.randint(0, len_of_list)
    list_of_numbers.append(num)

for i in range(num_loops):
    result = insertion_sort(list_of_numbers)

end = time.time()

run_time = end - start

print("Average time = {}".format(run_time / num_loops))
```
The nopython argument specifies if we want Numba to use purely machine code or to fill in some Python code if necessary. This should usually be set to true to get the best performance unless you find that Numba throws an error.

And if your code is `parallelizable` you can also pass `parallel = True` as an argument, but it must be used in conjunction with `nopython = True`. For now it only works on CPU.

You can also specify function signature you want your function to have, but then it won’t compile for any other types of arguments you give to it.
```python
from numba import jit, int32
@jit(int32(int32, int32))
def function(a, b):
    # your loop or numerically intensive computations
    return result

# or if you haven't imported type names
# you can pass them as string
@jit('int32(int32, int32)')
def function(a, b):
    # your loop or numerically intensive computations
    return result
```
Now your function will only take two int32’s and return an int32.

While the `jit()` decorator is useful for many situations, sometimes you want to write a function that has different implementations depending on its input types. The `generated_jit()`` decorator allows the user to control the selection of a specialization at compile-time.
```python
import numpy as np
from numba import generated_jit, types

@generated_jit(nopython=True)
def is_missing(x):
    """
    Return True if the value is missing, False otherwise.
    """
    if isinstance(x, types.Float):
        return lambda x: np.isnan(x)
    elif isinstance(x, (types.NPDatetime, types.NPTimedelta)):
        # The corresponding Not-a-Time value
        missing = x('NaT')
        return lambda x: x == missing
    else:
        return lambda x: False
```

## Speeding up Numpy operations
```python
from numba import vectorize, int64
import time
import numpy as np

num_loops = 50
img_1 = np.ones((1000, 1000), np.int64) * 5
img_2 = np.ones((1000, 1000), np.int64) * 10
img_3 = np.ones((1000, 1000), np.int64) * 15

@vectorize([int64(int64, int64, int64)], target="parallel")
def add_arrays(img_1, img_2, img_3):
    return np.square(img_1 + img_2 + img_3)

start = time.time()

for i in range(num_loops):
    result = add_arrays(img_1, img_2, img_3)

end = time.time()
run_time = end - start

print("Average time = {}".format(run_time / num_loops))
```

The second input is called the “target”. It specifies how you would like to run your function:
- cpu: for running on a single CPU thread
- parallel: for running on a multi-core, multi-threaded CPU
- cuda: for running on the GPU

The parallel option tends to be much faster than the cpu option in almost all cases. The cuda option is mainly useful for very large arrays with many parallelizable operations, since in that case we can fully utilise the advantage of having so many cores on the GPU.

## Is it always super fast?
Numba is going to be most effective when applied in either of these areas:
- Places where Python code is slower than C code (typically loops)
- Places where the same operation is applied to an area (i.e the same operation on many elements)
