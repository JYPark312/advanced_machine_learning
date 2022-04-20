# 과제 내용

##1st week

### f test, t test 실제로 구현
``` python
def ftest(X,y):
    # X: inpute variables
    # y: target
    
def ttest(X,y,varname=None):
    # X: inpute variables
    # y: target
```

##2nd week

### cosine_kernel, gaussian_kernel, gaussian_2d_kernal, epanechnikov_kernel, kde1d, kde2d 구현과 시각화
```python
def cosine_kernel(x, train, h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
def gaussian_kernel(x, train, h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
def gaussian_2d_kernel(x, train, h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
def epanechnikov_kernel(x,train,h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)

def kde1d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    # return d, d contains probaiblity density values of test samples (1d array)
```

##3rd week

### adj_matrix, cluster_label과 시각화
```python
def kde2d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    # return d, d contains probaiblity density values of test samples (1d array)

def cluster_label(A,bsv):
    # A: adjacent matrix size of n*n (if two points are connected A_ij=1)
    # bsv: index of bounded support vectors
    #######OUTPUT########
    # return cluster labels (if samples are bounded support vectors, label=-1)
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1)
    # Hint: use scipy.sparse.csgraph.connected_components
```

##4th week

### learning order od variables, learn parameter

```python
def get_order(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
```
