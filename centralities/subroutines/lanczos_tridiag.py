import scipy.sparse as spsp
import numpy as np
import numpy.linalg as la

def lanczos_tridiag(A, u, maxit):
	nL, nL = A.shape
	u = u/la.norm(u)
	U = u
	alpha = []
	beta = []
	T = np.zeros((1,1))
	for j in range(maxit):
		if j==0:
			U = np.hstack([U, A.dot(U[:,[j]])])
		else:
			U = np.hstack([U, np.subtract(A.dot(U[:,[j]]), beta[j-1]*U[:,[j-1]])])
		alpha.append(np.asscalar(np.dot(U[:,[j+1]].T, U[:,[j]])))
		U[:,[j+1]] = np.subtract(U[:,[j+1]], alpha[j]*U[:,[j]])
		beta.append(la.norm(U[:,[j+1]]))

		if abs(beta[j]) < 1e-14:
			print('Warning: Symmetric Lanczos broke down with beta=0 and\nT=', T)
			break

		U[:,[j+1]] = U[:,[j+1]]/beta[j]

		T = np.hstack([T, np.zeros((j+1,1))])
		T = np.vstack([T, np.zeros((1,j+2))])
		T[j,j] = alpha[j]
		T[j, j+1] = beta[j]
		T[j+1, j] = beta[j]

	return U, T
