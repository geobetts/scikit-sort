"""
Example of the prediction
"""

from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA

from hierarchical import HierarchicalHotDeck

housing = fetch_california_housing()

X = housing['data']
y = housing['target']

split = train_test_split(X, y, test_size=0.2, random_state=123)

train = split[0]
test = split[1]
train_targets = split[2]
test_targets = split[3]

selector = SelectFromModel(estimator=DecisionTreeRegressor(),
                           max_features=2).fit(train, train_targets)

feature_names = housing['feature_names']
selected = selector.get_support()

feature_info = dict(zip(feature_names, selected))
features_selected = {k: v for k, v in feature_info.items() if v}

print(f'features selected: {list(features_selected.keys())}')

train_sm = selector.transform(train)
test_sm = selector.transform(test)

for x in [HierarchicalHotDeck(), DecisionTreeRegressor(), DummyRegressor()]:
    model = x.fit(train_sm, train_targets)
    predictions = model.predict(test_sm)
    print(x)
    print(mean_squared_error(predictions, test_targets))

pca_model = PCA(n_components=1).fit(train)

explained_variance = pca_model.explained_variance_ratio_
print(f'Explained variance: {explained_variance[0]}')

train_pca = pca_model.transform(train)
test_pca = pca_model.transform(test)

model = x.fit(train_pca, train_targets)
predictions = model.predict(test_pca)
print('PCA version of Hierarchical hotdeck')
print(mean_squared_error(predictions, test_targets))

"""
CONCLUSION
----------

On this data, there would likely be no advantages to using hierarchical hotdeck.
Performance of decision tree is much better.
"""