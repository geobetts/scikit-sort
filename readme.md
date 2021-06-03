# scikit-sort

![](/edvard-alexander-rolvaag-unsplash.jpg)

Algorithm to predict using sorting

Consistent with scikit-learn API.

The prediction is done by sorting the training data set and matching each unseen example 
to the 'closest' training set example in the sort.
    
The algorithm is designed for data with large amounts of categorical data as features which may be difficult to train via
traditional scikit-learn algorithm's due to one-hot encoding creating a high dimension space, which can 
adversely affect computational performance.

The algorithm assumes that the training set contains almost all of the examples for that problem,
such that very few unseen examples would be drastically different to all of the examples in the training set.

In addition, the algorithm may be worth considering when:
- something very simple and easily explainable is sufficient as compared to a typical scikit-learn algorithm (an example of this may be imputation of missing values)
- where the unseen examples are likely to be similar the training examples.
- where over fitting to the training data is less of a concern (i.e. the algorithm is not being deployed in a wide range of scenarios)
- where there are more training examples than unseen ones.

A simple example of how the mechanics of algorithm work:

training set

| a | b |
|---|---|
| 1 | 6 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |
| 4 | 6 |
| 4 | 6 |
| 4 | 6 |

targets

| c     |
|-------|
| 1     |
| 1     |
| 0     |
| 1     |
| 0     |
| 0     |
| 1     |


unseen examples

| a | b |
|---|---|
| 4 | 5 |
| 0 | 1 | 
| 5 | 7 | 
| 6 | 2 |

hierarchy

For the duplicates of a = 4 and b = 6, the median value of c is taken.

| a | b | c |
|---|---|---|
| 0 | 1 | ? |
| 1 | 6 | 1 |
| 2 | 3 | 1 |
| 3 | 4 | 0 |
| 4 | 5 | 1 |
| 4 | 5 | ? |
| 4 | 6 | 0 |
| 5 | 7 | ? |
| 6 | 2 | ? |


resultant predictions

| c |
|---|
| 1 |
| 1 |
| 0 |
| 0 |
