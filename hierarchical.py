"""
Hierarchical hotdeck prediction
"""
import itertools
import string
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_series_equal
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class HierarchicalHotDeck:
    """
    Prediction by sorting the data set and matching each unseen example to the 'closest' training set example.
    """

    def fit(self, X, y):
        """
        Prepare data for sort.
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame of shape (n_samples, n_features)
            Training data.
            Features are sorted in the order they appear in the array / data frame. So, the most
            predictive feature should go first.
            Features should be chosen carefully. Fewer highly predictive features will likely perform better.
            The algorithm's performance may be better if features are given a numerically encoded logical order,
            in order to determine how similar different categories are.
            There are instances where features without any logical order will help the algorithm - however there
            should a larger number of training cases per category in these instances, with a variety of values for
            other features.
            This is to avoid a value being predicted from another nominal category that happens to be near it
            in the hierarchy (e.g. strings are sorted alphabetically, meaning the hierarchy may not make logical sense).

            Volume of data is important to the algorithm - very small training sets are unlikely to have enough data
            to yield a good variety of predictions. The algorithm essentially relies on having seen the exact (or
            very similar) same example as the unseen example in the training set. Preferably the same example has
            been seen multiple times.

        y : numpy.array or pandas.Series of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
        """

        if not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            raise TypeError(f"The training data X needs to be a numpy array or pandas data frame")

        if not isinstance(y, np.ndarray) and not isinstance(y, pd.Series):
            raise TypeError(f"The targets y need to be a numpy array or pandas series")

        if X.shape[1] > 52:
            raise ValueError(f'There are {X.shape[1]} columns. The maximum number of columns is 52')

        targets = pd.Series(y.transpose()) if isinstance(y, np.ndarray) else y
        features = pd.DataFrame(X) if isinstance(X, np.ndarray) else X

        targets.name = 'a' if targets.name is None else targets.name
        self.targets_name = targets.name

        self.data = pd.merge(targets, features, right_index=True, left_index=True)

        # remove rows that are exactly the same to reduce size of data set
        self.data = self.data.drop_duplicates()

        if isinstance(X, np.ndarray):
            alphabet_list = list(string.ascii_lowercase) + list(string.ascii_uppercase)
            cols = itertools.islice(alphabet_list, len(list(self.data.columns)))
            self.data.columns = cols

        feature_names = list(self.data.columns)
        feature_names.remove('a') if targets.name is None else feature_names.remove(targets.name)

        # if there are rows with the same set of features, take the median and keep one
        self.data.groupby(feature_names).median().reset_index()

        return self

    def predict(self, X):
        """
        Predict on test vectors X.
        Parameters
        ----------
        X : numpy.array or pandas.DataFrame of shape (n_samples, n_features)
            Test data.
        Returns
        -------
        output : numpy.array or pandas.Series of shape (n_samples,)
            Predicted target values for X. Pandas / numpy usage will depend on the input.
        """

        if not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            raise TypeError(f"The variable X needs to be a numpy array or pandas data frame")

        features_cols = [x for x in self.data.columns if x != self.targets_name]

        features = pd.DataFrame(X, columns=features_cols) if isinstance(X, np.ndarray) else X

        features[self.targets_name] = np.nan

        df = features.append(self.data, ignore_index=True, sort=True)
        df = df.reset_index(drop=True)

        index = df[self.targets_name].index[df[self.targets_name].apply(np.isnan)]

        df = df.sort_values(features_cols)
        df[self.targets_name] = df[self.targets_name].ffill()
        df[self.targets_name] = df[self.targets_name].bfill()

        predictions = df.loc[list(index), self.targets_name]

        output = predictions.to_numpy() if isinstance(X, np.ndarray) else predictions
        # if no numbers after the decimal place then return integer output
        output = output.astype(int) if np.isclose(output, np.round(output)).all() else output

        return output


class TestHierarchicalHotDeckNumpy(unittest.TestCase):
    """
    Regression and classification work the same.

    Proof of test:
    train, test and train targets combined and sorted:
    | 0 | 1 | blank |
    | 1 | 6 | 1     |
    | 2 | 3 | 1     |
    | 3 | 4 | 0     |
    | 4 | 5 | 1     |
    | 4 | 6 | 0     |
    | 5 | 7 | blank |
    | 6 | 2 | blank |

    back fill means first blank is 1, forward fill means last two blanks are 0.
    hence [1, 0, 0]
    """

    def test_hierarchical_hotdeck(self):
        train = np.array([[2, 3], [1, 6], [4, 5], [3, 4], [4, 6]])
        train_targets = np.array([1, 1, 1, 0, 0])
        test = np.array([[6, 2], [0, 1], [5, 7]])
        test_targets = np.array([0, 1, 0])

        model = HierarchicalHotDeck()
        fitted_model = model.fit(train, train_targets)
        output = fitted_model.predict(test)

        expected_output = test_targets

        assert_array_equal(x=output, y=expected_output)
        self.assertEqual(accuracy_score(output, test_targets), 1)


class TestHierarchicalHotDeckPandas(unittest.TestCase):
    """
    Regression and classification work the same.

    Proof of test:

    train, test and train targets combined and sorted:
    | 'a' | 2 | 1     |
    | 'a' | 4 | 1     |
    | 'a' | 5 | blank |
    | 'a' | 6 | blank |
    | 'b' | 0 | blank |
    | 'b' | 1 | 1     |
    | 'b' | 3 | 0     |
    | 'b' | 4 | 0     |

     forward fill means [1, 1, 1]
    """

    def test_hierarchical_hotdeck(self):
        train = pd.DataFrame({'a': ['a', 'b', 'a', 'b', 'b'],
                              'b': [2, 1, 4, 3, 4]})

        train_targets = pd.Series([1, 1, 1, 0, 0])
        train_targets.name = 'f'

        test = pd.DataFrame({'a': ['a', 'b', 'a'],
                             'b': [6, 0, 5]})

        test_targets = pd.Series([1, 1, 1])
        test_targets.name = 'f'

        model = HierarchicalHotDeck()
        fitted_model = model.fit(train, train_targets)
        output = fitted_model.predict(test)

        expected_output = test_targets

        assert_series_equal(left=output, right=expected_output, check_dtype=False)

        self.assertEqual(accuracy_score(output.to_numpy(), test_targets.to_numpy()), 1)


class TestColumnsError(unittest.TestCase):
    """
    Test the column error is shown
    """

    def test(self):
        train, train_targets = make_classification(n_samples=9000,
                                                   n_features=100,
                                                   n_classes=2,
                                                   n_informative=2,
                                                   n_redundant=0,
                                                   n_repeated=0,
                                                   random_state=123)

        try:
            HierarchicalHotDeck().fit(train, train_targets)
        except ValueError as v:
            self.assertEqual(str(v), 'There are 100 columns. The maximum number of columns is 52')


unittest.main() if __name__ == '__main__' else None
