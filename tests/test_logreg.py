from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from ..classifiers.logreg_numpy import LogisticRegression


def test_on_synthetic_data():
	X, y = make_classification(
		n_samples=500, 
		n_features=4, 
		n_redundant=0, 
		n_informative=1, 
		n_clusters_per_class=1, 
		random_state=14
	)
	clf = LogisticRegression(0.01, 100)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	assert accuracy_score(preds, y_test) > 0.8


def test_on_iris():
	data = load_iris()
	X, y = data.data[:100], data.target[:100]
	clf = LogisticRegression(0.01, 100)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	clf.fit(X_train, y_train)
	preds = clf.predict(X_test)
	assert accuracy_score(preds, y_test) > 0.8