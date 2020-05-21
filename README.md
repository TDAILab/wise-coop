# wise-coop

Cooperante is aimed at minimizing human coopertaion rate and maximizing scores of ML classification when human and ML work together.


## Introduction
A lot of efforts have been made to create models maximizing accuracy of ML classfication problem. These trials concern about complete replacement of human to algorithm. However, calssification with human assistance has been overlooked.

## Installation
Cooperante is available at [the Python Package Index](http://pypi.org/project/cooperante/) and on [Anaconda Cloud](http://anaconda.org/conda-forge/cooperante).




## Contribution
Any contribution are welcome!

## Licence
MIT Licence

## Usage

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.5)
model = LogisticRegression()
model.fit(X_train, y_train)

# make prediction (None, class_num)
y_pred_proba = model.predict_proba(X_test)

# make penalty array
penalty_array = np.array([0, 1, 3],
                         [1, 0, 1],
                         [5, 1, 0])

coop = Cooperante(penalty_array)

# coop_pred:(None,)
# expected_rink:(None,)
coop_pred, expected_risk = coop.fit(y_pred_proba)

# Visualize
coop.plot_eval(y_true, metrics='accuracy')

```
