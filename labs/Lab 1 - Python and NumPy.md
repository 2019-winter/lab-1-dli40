---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Daniel Li


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


1. How do I input mathematical symbols?
2. How do I unsync a notebook from a MD file?
3. Why is there no Java kernel?


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
import numpy as np
```

```python
a = np.empty((6,4),dtype=np.int)
a.fill(2)
a
```

## Exercise 2

```python
b= np.empty((6,4),dtype=np.int)
b.fill(2)
np.fill_diagonal(b, 3)
b
```

## Exercise 3


So a*b works because we multiple each element by its corresponding partner in the other matrix. Like mulitplying by a constant. But np.dot(a,b) doesn't work because it actually tries to do matrix multiplication. For a mxn to multiply it needs to match the dimensions, so there must be n rows in the second matrix.


## Exercise 4

```python
print(np.dot(a.transpose(),b))
print(np.dot(a,b.transpose()))
```

Transposing flips rows and columns. So since A is 6x4 and B is also 6x4, flipping either one will result in a different multiplication. Depending on which one you flip, the resulting dot product is either 6x4 by 4x6 or 4x6 by 6x4, resulting in a 6x6 matrix or a 4x4 one.


## Exercise 5

```python
def my_print(word):
    print(word)
    
my_print('hello')
```

## Exercise 6

```python
def stats(arr_size):
    #first make random array of size arr_size
    my_arr = np.random.rand(arr_size)
    print(sum(my_arr))
    print(np.mean(my_arr))
    print(np.std(my_arr))
    
stats(8)
    #then get mean, sum, std_dev of arr
```

## Exercise 7

```python
def find_ones_for(arr):
    count=0
    for num in arr:
        if num ==1:
            count+=1
    return count

def find_ones_where(arr):
    return sum(arr[np.where(arr==1)])
    

test = np.array([0,1,1,1,1,4,6,2,1])
print('using for  : ',find_ones_for(test))
print('using where: ',find_ones_where(test))
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd

A = pd.DataFrame([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
A
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
B= np.empty((6,4),dtype=np.int)
B.fill(2)
np.fill_diagonal(B, 3)
B = pd.DataFrame(B)
B
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
print(A * B)
#print(np.dot(A,B))
#that won't work for the same reason.
```

So it looks like the behavior from Pandas is the same as numpy. So multiplying two matrices together is totally fine, but computing the dot product requires the dimensions to match up, where the columns of Matrix X must equal the rows of Matrix Y. 


## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
def find_ones_for(df):
    count=0
    for row in df.iterrows():
        if row[1][0]==1:
            count+=1
    return count

def find_ones_where(df):
    return df[df.values==1].sum().sum()

    
test= pd.DataFrame([0,1,1,1,1,4,6,2,1])
print('using for  : ',find_ones_for(test))
print('using where: ',find_ones_where(test))
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df.name
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
titanic_df.loc['female']
titanic_df.loc['female'].shape[0]
```

## Exercise 14
How do you reset the index?

```python
titanic_df = titanic_df.reset_index()
titanic_df
```
