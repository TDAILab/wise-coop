# wise-coop
Wise-coop make suggestions to people on which items to check in order to accomplish 99.9%-accuracy.

## Introduction
A lot of efforts have been made to create machine learning models which can replace human jobs. 
Scientists are eager to reach 100% score and SoTA method is renewed every day in the world. 
However, is it possible for AI to replace human completely? 
AI may have the limit of capability by itself.
If it has, how about cooperation? Human can help overcome the limit.
We may play role in pushing up 90%-accurate prediction submitted by AI to 99.9%-accurate prediction.

Wise-coop make suggestions to people on which items to check in order to accomplish 99.9%-accuracy.
Wise-coop receives AI predictions and penalty matrix as an input, then outputs new
predictions and priority of human cooperation for each sample.

## Installation
Wise-coop is available at [the Python Package Index](http://pypi.org/project/wisecoop/) 
```bash
# PyPI
$ pip install wisecoop
```

Wise-coop supports Python 3.7 or newer.


## Contribution
Any contribution is welcome! If you find an issue, please submit it in the GitHub issue tracker for this repository.

## Licence
MIT License (see [LICENSE](./LICENSE)).

## Usage
More detailed description is available at [our blog](https://www.wantedly.com/companies/company_2215096/post_articles/243981)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wisecoop import Cooperante

iris = datasets.load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.4)
model = LogisticRegression()
model.fit(X_train, y_train)

# make prediction (None, n_classes)
y_pred_proba = model.predict_proba(X_test)

# prepare penalty array of shape (n_classes, n_classes)
penalty_array = np.array([[0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 0]])

# Instantiation
coop = Cooperante(penalty_array)

# Fitting
# pred : prediction label
# penalty_min : expectation of penalty
pred, penalty_min = coop.fit(y_pred_proba)

# Visualize
# plot trasition of the score when the items are checked by human according to penalty_min
fig, ax = coop.plot_eval(y_test, metrics='recall_score', class_ref = [0,1])

#percentage of human cooperatoin to achieve your KPI (e.g. recall score >= 0.99)
coop.score_to_check_rate(class_ref = 1, metric = "recall_score", threshold = 0.99)
```