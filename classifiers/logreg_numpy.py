import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LogisticRegression(BaseEstimator, ClassifierMixin):


	def __init__(self, lr, epochs):
		self.lr = lr
		self.epochs = epochs
		self.alpha = 1.

	
	@staticmethod
	def _sigmoid(X):
		'''Computes sigmoid activation

		Parameters
		----------

		X: {float, ndarray of shape (n_features,)}
		   Data sample.

		Returns
		----------
		{float, ndarray of shape (n_features,)}
		'''

		return 1. / (1. + np.exp(-X))


	def _mul(self, X):
		'''Multiplies features and weights

		Parameters
		----------

		X: ndarray of shape (n_features,)
		   Data sample.

		Returns
		----------
		float
		'''

		return np.dot(X, self.w)


	def _forward(self, X):
		'''Forward pass

		Parameters
		----------

		X: ndarray of shape (n_features,)
		   Data sample.

		Returns
		----------
		float
		'''

		return LogisticRegression._sigmoid(self._mul(X))


	def _logloss(self, X, y):
		'''Computes logloss

		Parameters
		----------

		X: ndarray of shape (n_samples, n_features)
		   Data sample.
		y: ndarray of shape (n_samples,)
		   Labels.

		Returns
		----------
		float
		'''

		m = X.shape[0]
		loss = -(1 / m) * np.sum(
			y * np.log(self._forward(X)) + (1 - y) * np.log(
				1 - self._forward(X)
			)
		) + self.alpha * np.dot(self.w.T, self.w)
		return loss


	def _gradient(self, x, y):
		'''Computes gradient

		Parameters
		----------

		X: ndarray of shape (n_features,)
		   Data sample.
		y: int
		   Label.

		Returns
		----------
		ndarray of shape (n_features,)
		'''

		return np.dot(
			x.T, 
			LogisticRegression._sigmoid(
				self._mul(x)
			) - y)


	def fit(self, X, y):
		'''Fits classifier

		Parameters
		----------

		X: ndarray of shape (n_samples, n_features)
		   Data sample.
		y: ndarray of shape (n_samples,)
		   Labels.

		Returns
		----------
		LogisticRegression instance
		'''

		X, y = check_X_y(X, y)
		n_features = X.shape[1]
		# self.w = np.random.normal((n_features, 1))
		self.w = np.zeros(n_features)
		self.logloss_history = np.zeros((self.epochs, 1))
		for i in range(self.epochs):
			sample_ind = np.random.randint(X.shape[0])
			sample, label = X[sample_ind], y[sample_ind]
			self.w -= self.lr * self._gradient(sample, label)
			self.logloss_history[i] = self._logloss(X, y)
		return self


	def predict(self, X):
		'''Makes predictions

		Parameters
		----------

		X: ndarray of shape (n_samples, n_features)
		   Data sample.

		Returns
		----------
		ndarray of shape (n_samples,)
		'''
		
		X = check_array(X)
		return np.round(LogisticRegression._sigmoid(self._mul(X)))


