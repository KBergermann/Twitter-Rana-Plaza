import numpy as np
import numpy.linalg as la

def gauss_subgraph(T, beta_subgraph):
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		lamb, phi = la.eig(T[0:k-1, 0:k-1])
		e = np.zeros([k-1,1])
		e[0] = 1
		int_val = np.asscalar(np.dot(e.T, phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(e))

	return int_val

def gauss_radau_subgraph(T, beta_subgraph, lambda_bound):
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		rhs = np.zeros([k-1,1])
		rhs[k-2] = T[k-2, k-1]**2
		e = np.zeros([k,1])
		e[0] = 1

		# prescribe the eigenvalue lambda_bound to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_bound*np.eye(k-1), rhs)
		T[k-1,k-1] = lambda_bound + delta[k-2]

		lamb, phi = la.eig(T)
		int_val = np.asscalar(np.dot(e.T, phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(e))

	return int_val

def gauss_lobatto_subgraph(T, beta_subgraph, lambda_min, lambda_max):
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		e_k = np.zeros([k-1,1])
		e_k[k-2] = 1
		e = np.zeros([k,1])
		e[0] = 1

		# prescribe both eigenvalues to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_min*np.eye(k-1), e_k)
		mu = la.solve(T[0:k-1, 0:k-1]-lambda_max*np.eye(k-1), e_k)
		T_entries = la.solve([[1, np.asscalar(-delta[k-2])], [1, np.asscalar(-mu[k-2])]], [[lambda_min], [lambda_max]])
		T[k-1, k-1] = T_entries[0]

		# catch error
		if T_entries[1] <= 0:
			int_val = 1
			print('Warning, prevented taking the root of %f in gauss_lobatto_subgraph. Set centrality value to 1.' % np.asscalar(T_entries[1]))
		else:
			T[k-2, k-1] = np.sqrt(T_entries[1])
			T[k-1, k-2] = np.sqrt(T_entries[1])

			lamb, phi = la.eig(T)
			int_val = np.asscalar(np.dot(e.T, phi).dot(np.diag(np.exp(beta_subgraph*lamb))).dot(phi.T).dot(e))

	return int_val

def gauss_resolvent(T, alpha_resolvent):
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		lamb, phi = la.eig(T[0:k-1, 0:k-1])
		e = np.zeros([k-1,1])
		e[0] = 1
		int_val = np.asscalar(np.dot(e.T, phi).dot(la.solve((np.eye(k-1) - alpha_resolvent*np.diag(lamb)), np.eye(k-1))).dot(phi.T).dot(e))

	return int_val

def gauss_radau_resolvent(T, alpha_resolvent, lambda_bound):
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		rhs = np.zeros([k-1,1])
		rhs[k-2] = T[k-2, k-1]**2
		e = np.zeros([k,1])
		e[0] = 1

		 # prescribe the eigenvalue lambda_bound to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_bound*np.eye(k-1), rhs)
		T[k-1,k-1] = lambda_bound + delta[k-2]

		lamb, phi = la.eig(T)
		int_val = np.asscalar(np.dot(e.T, phi).dot(la.solve((np.eye(k) - alpha_resolvent*np.diag(lamb)), np.eye(k))).dot(phi.T).dot(e))

	return int_val

def gauss_lobatto_resolvent(T, alpha_resolvent, lambda_min, lambda_max):
	k, k = T.shape
	if k==1:
		int_val=1
	else:
		e_k = np.zeros([k-1,1])
		e_k[k-2] = 1
		e = np.zeros([k,1])
		e[0] = 1

		# prescribe both eigenvalues to T
		delta = la.solve(T[0:k-1, 0:k-1]-lambda_min*np.eye(k-1), e_k)
		mu = la.solve(T[0:k-1, 0:k-1]-lambda_max*np.eye(k-1), e_k)
		T_entries = la.solve([[1, np.asscalar(-delta[k-2])], [1, np.asscalar(-mu[k-2])]], [[lambda_min], [lambda_max]])
		T[k-1, k-1] = T_entries[0]

		# catch error
		if T_entries[1] <= 0:
			int_val = 1
			print('Warning, prevented taking the root of %f in gauss_lobatto_subgraph. Set centrality value to 1.' % np.asscalar(T_entries[1]))
		else:
			T[k-2, k-1] = np.sqrt(T_entries[1])
			T[k-1, k-2] = np.sqrt(T_entries[1])

			lamb, phi = la.eig(T)
			int_val = np.asscalar(np.dot(e.T, phi).dot(la.solve((np.eye(k) - alpha_resolvent*np.diag(lamb)), np.eye(k))).dot(phi.T).dot(e))

	return int_val

