import numpy as np
import numpy.linalg as la
import scipy.sparse as spsp

# inspired by the Matalb implementation of the Arnoldi method from the funm_kryl toolbox
# http://guettel.com/funmkryl

def arnoldi(A, b, maxit, reorth=0):
	nL, nL = A.shape
	V = b/la.norm(b)
	H = np.zeros((1,1))

	for k in range(maxit):
		w = A.dot(V[:,k]).reshape(-1,1)

		for r in range(reorth+1):
			for j in range(k+1):
				ip = np.dot(w.T, V[:,j])
				H[j,k] = H[j,k] + ip
				w = w - V[:,j].reshape(-1,1)*ip

		H = np.hstack([H, np.zeros((k+1,1))])
		H = np.vstack([H, np.zeros((1,k+2))])
		H[k+1,k] = np.sqrt(np.dot(w.T, w))

		if H[k+1,k] < 1e-12:
			print('Warning: Arnoldi broke down!')

		V = np.hstack([V, (1/H[k+1,k])*w])

	return V, H
