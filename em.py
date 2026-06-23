import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def gaussian_(n :int, x :np.ndarray, sigma :np.ndarray, mu :np.ndarray) -> np.ndarray:
  x -= mu
  _pre = 1 / (2*math.pi)**n * sigma**0.5
  _in = -0.5 * x.T * np.linalg.inv(sigma) * x
  return _pre * np.exp(_in)

class e_m_:
  def __init__(self, X :np.ndarray, epochs :int, _ncluster :int):
    self.X = X
    self.epochs = epochs
    self.k = _ncluster

  def train_model(self):
    _nsample, _nfeatures = self.X.shape

    omega = np.zeros(shape=(_nsample, self.k))
    phi = np.ones(shape=(self.k))
    mu = np.zeros(shape=(self.k))
    sigma = np.ones(shape=(self.k))

    self._latent = np.zeros(shape=(self.k))

    for iter in range(self.epochs):
      for j in range(self.k):
        _den = [gaussian_(0.5, self.X[l], sigma, mu)*phi[l] for l in range(self.k)]
        _den = _den.sum()
        for sample in range(_nsample):
          p_x_zj = gaussian_(0.5, self.X[sample], sigma, mu)
          omega[sample, j] = p_x_zj * phi[j]
          omega[sample, j] /= _den

        #update the params
        phi[j] = omega[:, j].sum() / _nsample
        mu[j] = (omega[:, j].T @ self.X).sum()
        mu[j] /= omega[:, j].sum()

        _t = self.X - mu[j]
        sigma[j] = (omega[:, j] * _t * _t.T).sum()
        sigma[j] /= omega[:, j].sum()

    self._omega = omega


  def predict(self, test :np.ndarray):
    _res = np.zeros(shape=self._omega.shape)
    for j in range(self.k):
      _res[:, j] = self._omega[:, j] * test

    return _res.argmax(axis=1)

