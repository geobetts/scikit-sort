"""
Example using synthetic data

Example is intended to show usage on features with many catagories and how this is
advantageous compared to traditional scikit-learn from a speed perspective.
"""
from time import time
from sklearn.tree import DecisionTreeClassifier
from hierarchical import HierarchicalHotDeck
from faker import Faker
from pandas import DataFrame, get_dummies
from numpy.random import choice, randint
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed(123)

sample_size = 100000
job_categories = 200

fake = Faker()

jobs = []

for _ in range(job_categories):
    jobs.append(fake.job())

df = DataFrame(
    {'targets': randint(0, 2, size=sample_size),
     'job': choice(jobs, size=sample_size)
     }
)

# hierarchical example
t = time()
hh_train, hh_test = train_test_split(df, test_size=0.2)

hh_fitted_model = HierarchicalHotDeck().fit(hh_train[['job']], hh_train['targets'])

hh_predictions = hh_fitted_model.predict(hh_test[['job']])

hh_accuracy = accuracy_score(hh_predictions, hh_test['targets'])
hh_time = time() - t

# decision tree example
t = time()
one_hot_encoded = get_dummies(df, columns=['job'])
dt_train, dt_test = train_test_split(one_hot_encoded, test_size=0.2)

dt_train_features = dt_train.drop('targets', axis=1)
dt_test_features = dt_test.drop('targets', axis=1)

dt_fitted_model = DecisionTreeClassifier().fit(dt_train_features, dt_train['targets'])

dt_predictions = dt_fitted_model.predict(dt_test_features)

dt_accuracy = accuracy_score(dt_predictions, dt_test['targets'])
dt_time = time() - t

print(f'HierarchicalHotDeck time: {hh_time}')
print(f'DecisionTreeClassifier time: {dt_time}')

print(f'HierarchicalHotDeck accuracy: {hh_accuracy}')
print(f'DecisionTreeClassifier accuracy: {dt_accuracy}')