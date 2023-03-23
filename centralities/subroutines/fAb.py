import numpy as np
import numpy.linalg as la
from scipy.linalg import expm


def expAb_sym(U, T, b, beta_subgraph):
	lamb, phi = la.eig(T)
	UTb = np.dot(U.T, b)
	fAb = U.dot(phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(UTb)
	return fAb

def resolventAb_sym(U, T, b, alpha_resolvent):
	k, k = T.shape
	lamb, phi = la.eig(T)
	UTb = np.dot(U.T, b)
	fAb = U.dot(phi).dot(la.solve((np.eye(k) - alpha_resolvent*np.diag(lamb)), np.eye(k))).dot(phi.T).dot(UTb)
	return fAb

def expAb_unsym(V, H, b, beta_subgraph):
	VTb = np.dot(V.T, b)
	fAb = V.dot(expm(beta_subgraph*H)).dot(VTb)
	return fAb

def resolventAb_unsym(V, H, b, alpha_resolvent):
	k, k = H.shape
	VTb = np.dot(V.T, b)
	fAb = V.dot(la.solve((np.eye(k) - alpha_resolvent*H), np.eye(k))).dot(VTb)
	return fAb
