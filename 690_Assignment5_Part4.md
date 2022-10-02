**Numpy: Numeric computing library**

NumPy (Numerical Python) is one of the core packages for numerical computing in Python. Pandas, Matplotlib, Statmodels and many other Scientific libraries rely on NumPy.



```python
import sys
import numpy as np
```

**Basic numpy arrays**


```python
np.array([1, 2, 3, 4])
```




    array([1, 2, 3, 4])




```python
a= np.array([1, 2, 3, 4])
```


```python
a
```




    array([1, 2, 3, 4])




```python
b = np.array([0, .5, 1, 1.5, 2])
b
```




    array([0. , 0.5, 1. , 1.5, 2. ])




```python
a[0],a[1]
```




    (1, 2)




```python
a[0:1]
```




    array([1])




```python
a[:1]
```




    array([1])




```python
a[2:]
```




    array([3, 4])




```python
a[: :]
```




    array([1, 2, 3, 4])




```python
a[:-1]
```




    array([1, 2, 3])



**Array Types**


```python
a.dtype
```




    dtype('int32')




```python
b.dtype
```




    dtype('float64')




```python
np.array([1, 2, 3, 4], dtype=np.float)
```




    array([1., 2., 3., 4.])




```python
np.array([1, 2, 3, 4], dtype=np.int8)
```




    array([1, 2, 3, 4], dtype=int8)




```python
c = np.array(['a', 'b', 'c'])
```


```python
c.dtype
```




    dtype('<U1')




```python
d = np.array([{'a': 1}, sys])
```


```python
d.dtype
```




    dtype('O')



**Dimensions and shapes**


```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]])
```


```python
A.dtype
```




    dtype('int32')




```python
A.shape
```




    (2, 3)




```python
A.size
```




    6




```python
A.ndim
```




    2




```python
B = np.array([
    [
        [12, 11, 10],
        [9, 8, 7],
    ],
    [
        [6, 5, 4],
        [3, 2, 1]
    ]
])
```


```python
B
```




    array([[[12, 11, 10],
            [ 9,  8,  7]],
    
           [[ 6,  5,  4],
            [ 3,  2,  1]]])




```python
B.size
```




    12




```python
B.shape
```




    (2, 2, 3)




```python
B.ndim
```




    3



If the shape isn't consistent, it'll just fall back to regular Python objects:


```python
C = np.array([
    [
        [12, 11, 10],
        [9, 8, 7],
    ],
    [
        [6, 5, 4]
    ]
])
```

    <ipython-input-34-39b57e6bbad9>:1: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      C = np.array([
    


```python
C.dtype
```




    dtype('O')




```python
C.shape
```




    (2,)




```python
C.size
```




    2




```python
type(C[0])
```




    list



**Indexing and Slicing of Matrices**


```python
# Square matrix
A = np.array([
    [1, 2, 3], # 0
    [4, 5, 6], # 1
    [7, 8, 9]  # 2
])
```


```python
A[1]
```




    array([4, 5, 6])




```python
A[1][0]
```




    4




```python
A[1,2]
```




    6




```python
A[1:0]
```




    array([], shape=(0, 3), dtype=int32)




```python
A[:]
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
A[::]
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
A[:-1]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
A[1:]
```




    array([[4, 5, 6],
           [7, 8, 9]])




```python
A[:2]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
A[:0]
```




    array([], shape=(0, 3), dtype=int32)




```python
A[1] = np.array([10, 10, 10])
```


```python
A
```




    array([[ 1,  2,  3],
           [10, 10, 10],
           [ 7,  8,  9]])




```python
A[2]=88
```


```python
A
```




    array([[ 1,  2,  3],
           [10, 10, 10],
           [88, 88, 88]])



**Summary Statistics**


```python
a.sum()
```




    10




```python
a.std()
```




    1.118033988749895




```python
a.var()
```




    1.25




```python
a.mean()
```




    2.5




```python
A.sum()
```




    300




```python
A.mean()
```




    33.333333333333336




```python
A.std()
```




    38.7957615096174




```python
A.var()
```




    1505.111111111111




```python
A.sum(axis=1)
```




    array([  6,  30, 264])




```python
A.mean(axis=1)
```




    array([ 2., 10., 88.])




```python
A.std(axis=1)
```




    array([0.81649658, 0.        , 0.        ])




```python
A.var(axis=1)
```




    array([0.66666667, 0.        , 0.        ])



**Broadcasting and Vectorized operations**


```python
a = np.arange(4)
```


```python
a
```




    array([0, 1, 2, 3])




```python
a+10
```




    array([10, 11, 12, 13])




```python
a-10
```




    array([-10,  -9,  -8,  -7])




```python
a*10
```




    array([ 0, 10, 20, 30])




```python
a += 100
a
```




    array([300, 301, 302, 303])




```python
l = [0, 1, 2, 3]
[i * 10 for i in l]
```




    [0, 10, 20, 30]




```python
a = np.arange(4)
a
```




    array([0, 1, 2, 3])




```python
b = np.array([10, 10, 10, 10])
b
```




    array([10, 10, 10, 10])




```python
a+b
```




    array([10, 11, 12, 13])




```python
a*b
```




    array([ 0, 10, 20, 30])



**Boolean arrays**


```python
a = np.arange(4)
a
```




    array([0, 1, 2, 3])




```python
a[0],a[-1]
```




    (0, 3)




```python
a[[0,-1]]
```




    array([0, 3])




```python
a[[True, False, False, True]]
```




    array([0, 3])




```python
a >= 2
```




    array([False, False,  True,  True])




```python
a[a>=2]
```




    array([2, 3])




```python
a.mean()
```




    1.5




```python
a[a > a.mean()]
```




    array([2, 3])




```python
a[~(a > a.mean())]
```




    array([0, 1])




```python
a[(a == 0) | (a == 1)]
```




    array([0, 1])




```python
a[(a <= 2) & (a % 2 == 0)]
```




    array([0, 2])




```python
A = np.random.randint(100, size=(3, 3))
A
```




    array([[83, 42, 29],
           [87,  3,  3],
           [95, 65, 30]])




```python
A[np.array([
    [True, False, True],
    [False, True, False],
    [True, False, True]
])]
```




    array([83, 29,  3, 95, 30])




```python
A>30
```




    array([[ True,  True, False],
           [ True, False, False],
           [ True,  True, False]])




```python
A[A>30]
```




    array([83, 42, 87, 95, 65])



**Linear Algebra**


```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```


```python
B = np.array([
    [6, 5],
    [4, 3],
    [2, 1]
])
```


```python
A.dot(B)
```




    array([[20, 14],
           [56, 41],
           [92, 68]])




```python
A@B
```




    array([[20, 14],
           [56, 41],
           [92, 68]])




```python
B.T
```




    array([[6, 4, 2],
           [5, 3, 1]])




```python
B.T@A
```




    array([[36, 48, 60],
           [24, 33, 42]])



**Size of objects in Memory**
 Int, floats


```python
# An integer in Python is > 24bytes
sys.getsizeof(1)
```




    28




```python
# Longs are even larger
sys.getsizeof(10**100)
```




    72




```python
# Numpy size is much smaller
np.dtype(int).itemsize
```




    4




```python
# Numpy size is much smaller
np.dtype(np.int8).itemsize
```




    1




```python
np.dtype(float).itemsize

```




    8



**Lists are even larger**


```python
# A one-element list
sys.getsizeof([1])
```




    64




```python
# An array of one element in numpy
np.array([1]).nbytes
```




    4



**And performance is also important**


```python
l = list(range(100000))
a = np.arange(100000)
%time np.sum(a ** 2)
```

    Wall time: 924 µs
    




    216474736




```python
%time sum([x ** 2 for x in l])
```

    Wall time: 62.3 ms
    




    333328333350000



**Useful numpy functions**

random


```python
np.random.random(size=2)
```




    array([0.74825756, 0.93450607])




```python
np.random.normal(size=2)
```




    array([-0.41350984,  0.70281866])




```python
np.random.rand(2, 4)
```




    array([[0.12096507, 0.10456037, 0.78727658, 0.40665346],
           [0.59253123, 0.38312755, 0.86004097, 0.15006302]])



arange


```python
np.arange(10)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.arange(5, 10)
```




    array([5, 6, 7, 8, 9])




```python
np.arange(0, 1, .1)
```




    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])



reshape


```python
np.arange(10).reshape(2, 5)
```




    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])




```python
np.arange(10).reshape(5, 2)
```




    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])



linspace


```python
np.linspace(0, 1, 5)
```




    array([0.  , 0.25, 0.5 , 0.75, 1.  ])




```python
np.linspace(0, 1, 20)
```




    array([0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,
           0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
           0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
           0.78947368, 0.84210526, 0.89473684, 0.94736842, 1.        ])




```python
np.linspace(0, 1, 20, False)
```




    array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ,
           0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95])



zeros, ones, empty


```python
np.zeros(5)

```




    array([0., 0., 0., 0., 0.])




```python
np.zeros((3, 3))

```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])




```python
np.zeros((3, 3), dtype=np.int)

```




    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])




```python
np.ones(5)

```




    array([1., 1., 1., 1., 1.])




```python
np.ones((3, 3))

```




    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])




```python
np.empty(5)

```




    array([1., 1., 1., 1., 1.])




```python
np.empty((2, 2))
```




    array([[0.25, 0.5 ],
           [0.75, 1.  ]])



identity and eye


```python
np.identity(3)

```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
np.eye(3, 3)

```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
np.eye(8, 4)

```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
np.eye(8, 4, k=1)

```




    array([[0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
np.eye(8, 4, k=-3)

```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.],
           [0., 0., 0., 0.]])




```python
"Hello World"[6]
```




    'W'




```python

```
