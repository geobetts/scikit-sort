"""
Example to show computational advantages of hierarchical hotdeck with
features that have many categories.

E.g. of a result
training and test set of 965,511 examples and 786 unique categories
hierarchical hotdeck: 0.73 seconds to train and test
decision tree: 155 seconds to train and test

Accuracy results were roughly the same in the example above,
however it is not particularly meaningful as there is no pattern
to discover due to random data generation.
"""

from time import time
from sklearn.tree import DecisionTreeClassifier
from hierarchical import HierarchicalHotDeck
from faker import Faker
from pandas import DataFrame, get_dummies
from plotly.express import scatter_3d
from numpy.random import choice, randint
from random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed(123)

iterations = 100
samples = randint(10, 1_000_000, size=iterations)
categories = randint(1, 1_000, size=iterations)

# define empty lists required later
hh_times = []
dt_times = []
hh_accuracies = []
dt_accuracies = []

fake = Faker()

n = 1

for sample_size, job_categories in zip(samples, categories):

    print(f'Iteration {n}: {sample_size} samples and {job_categories} categories')
    n += 1

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
    hh_accuracies.append(hh_accuracy)
    hh_time = time() - t
    hh_times.append(hh_time)
    print('- hierarchical hotdeck complete')

    # decision tree example
    t = time()
    one_hot_encoded = get_dummies(df, columns=['job'])
    dt_train, dt_test = train_test_split(one_hot_encoded, test_size=0.2)

    dt_train_features = dt_train.drop('targets', axis=1)
    dt_test_features = dt_test.drop('targets', axis=1)

    dt_fitted_model = DecisionTreeClassifier().fit(dt_train_features, dt_train['targets'])

    dt_predictions = dt_fitted_model.predict(dt_test_features)

    dt_accuracy = accuracy_score(dt_predictions, dt_test['targets'])
    dt_accuracies.append(dt_accuracy)
    dt_time = time() - t
    dt_times.append(dt_time)
    print('- decision tree complete')

timing_info = DataFrame({'sample': list(samples) + list(samples),
                         'categories': list(categories) + list(categories),
                         'time': hh_times + dt_times,
                         'group': ['hierarchical'] * iterations + ['decision tree'] * iterations,
                         'accuracies': hh_accuracies + dt_accuracies})

fig = scatter_3d(timing_info, x='sample', y='categories', z='time',
                 color='group')

fig.write_html('./graph_decision_tree_time_v_hierarchical.html')
