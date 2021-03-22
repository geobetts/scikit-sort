"""
Example of the prediction where hierarchical hot-deck may perform well.

Categorical variables with many categories - hh has an advantage due to no need for one-hot encoding.
One hot encoding with data that has many categories can lead to many features - some algorithms
may struggle to find useful information in this - hypothesis is that hh would work quicker and more
accurately with this type of data.

Below is a placeholder for the code - data is not large enough to to reach
any conclusions.
"""
from os import getcwd

from numpy import where
from pandas import read_csv, get_dummies, merge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

from hierarchical import HierarchicalHotDeck

test_data = read_csv('./test_data/test_data.csv')

test_data['targets'] = where(test_data['winner'] == ' England', 1, 0)

# re-order columns based on predictability
test_data = test_data[['year', 'location', 'championship', 'winner', 'targets']]

split2 = train_test_split(test_data, test_size=0.5, random_state=123)

train = split2[0]
test = split2[1]
train_targets = train['targets']
test_targets = test['targets']

train = train.drop(['winner', 'targets'], axis=1)
test = test.drop(['winner', 'targets'], axis=1)

x = HierarchicalHotDeck()
model_hh = x.fit(train, train_targets)
predictions_hh = model_hh.predict(test)
print(x)
print(accuracy_score(predictions_hh, test_targets))

# redo this
one_hot_data = merge(get_dummies(test_data[['location', 'championship']]), test_data['year'],
                     left_index=True, right_index=True)

train_one_hot = one_hot_data.filter(list(train.index), axis=0)
test_one_hot = one_hot_data.filter(list(test.index), axis=0)

for x in [DecisionTreeClassifier(), RandomForestClassifier(), DummyClassifier(strategy='most_frequent')]:
    model = x.fit(train_one_hot, train_targets)
    predictions = model.predict(test_one_hot)
    print(x)
    print(accuracy_score(predictions, test_targets))
