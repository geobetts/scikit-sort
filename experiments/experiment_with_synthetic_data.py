"""
Example of the prediction
"""

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.decomposition import PCA

from hierarchical import HierarchicalHotDeck

X, y = make_classification(n_samples=4000,
                           n_features=2,
                           n_classes=2,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           random_state=123)

split = train_test_split(X, y, test_size=0.2, random_state=123)

train = split[0]
test = split[1]
train_targets = split[2]
test_targets = split[3]

for x in [HierarchicalHotDeck(), DecisionTreeClassifier(), DummyClassifier(strategy='most_frequent')]:
    model = x.fit(train, train_targets)
    predictions = model.predict(test)
    print(x)
    print(accuracy_score(predictions, test_targets))

pca_model = PCA(n_components=1).fit(train)

explained_variance = pca_model.explained_variance_ratio_
print(f'Explained variance: {explained_variance[0]}')

train_pca = pca_model.transform(train)
test_pca = pca_model.transform(test)

model = x.fit(train_pca, train_targets)
predictions = model.predict(test_pca)
print('PCA version of Hierarchical hotdeck')
print(accuracy_score(predictions, test_targets))

