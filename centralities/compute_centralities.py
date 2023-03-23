import scipy.sparse as spsp
import numpy as np
import numpy.linalg as la
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from subroutines.lanczos_tridiag import *
from subroutines.fAb import *
from subroutines.gauss_quadrature_rules import *
from subroutines.arnoldi import *

def compute_centralities(saveStr, omega=1, alpha_const=0.5, beta_const=2, maxit_quadrature=10, maxit_fAb=20, top_k_centralities=10, sym_bipartite_graph=1):

	####################
	### Preparations ###
	####################
	
	# read and prepare data
	A_intra_retweet = spsp.load_npz('../networks_python/%s_Aintra_retweet.npz' % saveStr)
	A_intra_reply = spsp.load_npz('../networks_python/%s_Aintra_reply.npz' % saveStr)
	A_intra_mention = spsp.load_npz('../networks_python/%s_Aintra_mention.npz' % saveStr)
	
	
	n = A_intra_retweet.shape[0]
	L = 3
	nL = n*L
	
	try:
		userIDList = pd.read_csv('../networks_python/%s_user_IDs_screen_names.csv' % saveStr)
	except:
		userIDList = pd.DataFrame({'0': np.arange(n), 'user_id': np.arange(n), 'user_screen_name': ['user'+str(i) for i in range(n)], 'node_id': np.arange(n)})
		
	layerIDList = pd.DataFrame({'layer_id': np.arange(3), 'layer_name': ['Retweet', 'Reply', 'Mention']})
	
	
	A_intra = spsp.bmat([[A_intra_retweet, None, None], [None, A_intra_reply, None], [None, None, A_intra_mention]])
	A_inter_weight = np.ones([L,L])
	A_inter = spsp.kron(A_inter_weight, np.eye(n))

	A = A_intra + omega*A_inter
	
	#plt.spy(A, markersize=2)
	#plt.show()
	
	#####################################
	#####################################
	########## Shifted Arnoldi ##########
	#####################################
	#####################################
	if sym_bipartite_graph==0:
	
		########################
		### f(A)b quantities ###
		########################

		b = np.ones([nL,1])
		reorth=1

		####################################
		##### broadcaster centralities #####

		V, H = arnoldi(A, b, maxit_fAb, reorth)

		lamb, phi = la.eig(H[0:maxit_fAb, 0:maxit_fAb])
		lamb_max = max(lamb).real

		### Total communicability ###

		beta_subgraph = beta_const/lamb_max

		TC_broadcaster = expAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, beta_subgraph)

		### Katz centrality ###

		alpha_resolvent = alpha_const/lamb_max

		KC_broadcaster = resolventAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, alpha_resolvent)

		#################################
		##### receiver centralities #####

		V, H = arnoldi(A.T, b, maxit_fAb, reorth)

		lamb, phi = la.eig(H[0:maxit_fAb, 0:maxit_fAb])
		lamb_max = max(lamb).real

		### Total communicability ###

		beta_subgraph = beta_const/lamb_max

		TC_receiver = expAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, beta_subgraph)

		### Katz centrality ###

		alpha_resolvent = alpha_const/lamb_max

		KC_receiver = resolventAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, alpha_resolvent)


		#############################
		### u^T f(A) u quantities ###
		#############################

		print('Subgraph and resolvent-based subgraph centrality. Looping over node-layer pairs. Progress:')

		# initialization ensures that isolated nodes in SC and SCres are assigned the value 1
		SC_shifted_term = 2*np.ones([nL,1])
		SCres_shifted_term = 2*np.ones([nL,1])

		# loop over non-isolated node-layer pairs
		i=0
		for node_id in range(nL):

			sys.stdout.write("\r{}%".format(100*((i+1)/nL)))
			sys.stdout.flush()

			u = np.ones([nL,1])
			u[node_id]+=1

			# subgraph centrality, shifted term
			V, H = arnoldi(A, u, maxit_fAb, reorth)

			SC_shifted_term[node_id] = expAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], u, beta_subgraph)[node_id]

			# resolvent-based subgraph centrality, shifted term

			SCres_shifted_term[node_id] = resolventAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], u, alpha_resolvent)[node_id]

			i+=1

		SC = SC_shifted_term - TC_broadcaster
		SCres = SCres_shifted_term - KC_broadcaster



		############################################
		### print and write centralities to file ###
		############################################

		# printing to file
		directory = 'results'

		if not os.path.exists(directory):
			os.makedirs(directory)

		print_file = open("%s/centralities_%s_omega_%s_sym_bipartite_%d.txt" % (directory, saveStr, omega, sym_bipartite_graph), "w")
		print_file.write('Parameters:\nalpha_const=%f, beta_const=%f, maxit_quadrature=%d, maxit_fAb=%d, top_k_centralities=%d\n\n' % (alpha_const, beta_const, maxit_quadrature, maxit_fAb, top_k_centralities))

		### Print top k centralities ###
		k = top_k_centralities

		### Subgraph centrality ###
		SC_sorted = -np.sort(-SC, axis=None)
		SC_sorted_ind = np.argsort(-SC, axis=None)

		# top k JCs
		print('\n-----Subgraph centrality-----\n')
		print_file.write('-----Subgraph centrality-----\n')
		df_values={'SC_JC_ranking': range(1,k+1), 'value': SC_sorted[0:k]}
		df_nodes={'SC_JC_ranking': range(1,k+1), 'node_id': (SC_sorted_ind[0:k] % n)}
		df_layers={'SC_JC_ranking': range(1,k+1), 'layer_id': (SC_sorted_ind[0:k] // n)}
		SC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SC_JC_ranking'])
		SC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_JC_ranking'])
		top_k_SC_JC = pd.merge(pd.merge(SC_node_names, SC_layer_names, on='SC_JC_ranking'),  pd.DataFrame(data=df_values), on='SC_JC_ranking')
		print('Joint centralities:\n', top_k_SC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_SC_JC.to_string(index=False))
				
		# top k MNCs
		SC_MNC = np.sum(SC.reshape((L,n)).T, axis=1)
		SC_MNC_sorted = -np.sort(-SC_MNC, axis=None)
		SC_MNC_sorted_ind = np.argsort(-SC_MNC, axis=None)
		df_values={'SC_MNC_ranking': range(1,k+1), 'value': SC_MNC_sorted[0:k]}
		df_nodes={'SC_MNC_ranking': range(1,k+1), 'node_id': SC_MNC_sorted_ind[0:k]}
		SC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SC_MNC_ranking'])
		top_k_SC_MNC = pd.merge(SC_MNC_node_names, pd.DataFrame(data=df_values), on='SC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_SC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_SC_MNC.to_string(index=False))

		# top k MLCs
		SC_MLC = np.sum(SC.reshape((L,n)).T, axis=0)
		SC_MLC_sorted = -np.sort(-SC_MLC, axis=None)
		SC_MLC_sorted_ind = np.argsort(-SC_MLC, axis=None)
		df_values={'SC_MLC_ranking': range(1,4), 'value': SC_MLC_sorted}
		df_layers={'SC_MLC_ranking': range(1,4), 'layer_id': SC_MLC_sorted_ind}
		SC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_MLC_ranking'])
		top_k_SC_MLC = pd.merge(SC_MLC_node_names, pd.DataFrame(data=df_values), on='SC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_SC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_SC_MLC.to_string(index=False))
		

		### Resolvent-based subgraph centrality ###
		SCres_sorted = -np.sort(-SCres, axis=None)
		SCres_sorted_ind = np.argsort(-SCres, axis=None)

		# top k JCs
		print('\n-----Resolvent-based subgraph centrality-----\n')
		print_file.write('\n-----Resolvent-based subgraph centrality-----\n')
		df_values={'SCres_JC_ranking': range(1,k+1), 'value': SCres_sorted[0:k]}
		df_nodes={'SCres_JC_ranking': range(1,k+1), 'node_id': (SCres_sorted_ind[0:k] % n)}
		df_layers={'SCres_JC_ranking': range(1,k+1), 'layer_id': (SCres_sorted_ind[0:k] // n)}
		SCres_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SCres_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SCres_JC_ranking'])
		SCres_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_JC_ranking'])
		top_k_SCres_JC = pd.merge(pd.merge(SCres_node_names, SCres_layer_names, on='SCres_JC_ranking'),  pd.DataFrame(data=df_values), on='SCres_JC_ranking')
		print('Joint centralities:\n', top_k_SCres_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_SCres_JC.to_string(index=False))
				
		# top k MNCs
		SCres_MNC = np.sum(SCres.reshape((L,n)).T, axis=1)
		SCres_MNC_sorted = -np.sort(-SCres_MNC, axis=None)
		SCres_MNC_sorted_ind = np.argsort(-SCres_MNC, axis=None)
		df_values={'SCres_MNC_ranking': range(1,k+1), 'value': SCres_MNC_sorted[0:k]}
		df_nodes={'SCres_MNC_ranking': range(1,k+1), 'node_id': SCres_MNC_sorted_ind[0:k]}
		SCres_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SCres_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SCres_MNC_ranking'])
		top_k_SCres_MNC = pd.merge(SCres_MNC_node_names, pd.DataFrame(data=df_values), on='SCres_MNC_ranking')
		print('Marginal node centralities:\n', top_k_SCres_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_SCres_MNC.to_string(index=False))

		# top k MLCs
		SCres_MLC = np.sum(SCres.reshape((L,n)).T, axis=0)
		SCres_MLC_sorted = -np.sort(-SCres_MLC, axis=None)
		SCres_MLC_sorted_ind = np.argsort(-SCres_MLC, axis=None)
		df_values={'SCres_MLC_ranking': range(1,4), 'value': SCres_MLC_sorted}
		df_layers={'SCres_MLC_ranking': range(1,4), 'layer_id': SCres_MLC_sorted_ind}
		SCres_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_MLC_ranking'])
		top_k_SCres_MLC = pd.merge(SCres_MLC_node_names, pd.DataFrame(data=df_values), on='SCres_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_SCres_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_SCres_MLC.to_string(index=False))
				
		
		### Total communicability ###
		### broadcaster
		TC_sorted = -np.sort(-TC_broadcaster, axis=None)
		TC_sorted_ind = np.argsort(-TC_broadcaster, axis=None)

		# top k JCs
		print('\n-----Total communicability-----\n-----broadcaster communicabilities\n')
		print_file.write('\n-----Total communicability-----\n-----broadcaster communicabilities\n')
		df_values={'TC_JC_ranking': range(1,k+1), 'value': TC_sorted[0:k]}
		df_nodes={'TC_JC_ranking': range(1,k+1), 'node_id': (TC_sorted_ind[0:k] % n)}
		df_layers={'TC_JC_ranking': range(1,k+1), 'layer_id': (TC_sorted_ind[0:k] // n)}
		TC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_JC_ranking'])
		TC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_JC_ranking'])
		top_k_TC_JC = pd.merge(pd.merge(TC_node_names, TC_layer_names, on='TC_JC_ranking'),  pd.DataFrame(data=df_values), on='TC_JC_ranking')
		print('Joint centralities:\n', top_k_TC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_TC_JC.to_string(index=False))
				
		# top k MNCs
		TC_MNC = np.sum(TC_broadcaster.reshape((L,n)).T, axis=1)
		TC_MNC_sorted = -np.sort(-TC_MNC, axis=None)
		TC_MNC_sorted_ind = np.argsort(-TC_MNC, axis=None)
		df_values={'TC_MNC_ranking': range(1,k+1), 'value': TC_MNC_sorted[0:k]}
		df_nodes={'TC_MNC_ranking': range(1,k+1), 'node_id': TC_MNC_sorted_ind[0:k]}
		TC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_MNC_ranking'])
		top_k_TC_MNC = pd.merge(TC_MNC_node_names, pd.DataFrame(data=df_values), on='TC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_TC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_TC_MNC.to_string(index=False))

		# top k MLCs
		TC_MLC = np.sum(TC_broadcaster.reshape((L,n)).T, axis=0)
		TC_MLC_sorted = -np.sort(-TC_MLC, axis=None)
		TC_MLC_sorted_ind = np.argsort(-TC_MLC, axis=None)
		df_values={'TC_MLC_ranking': range(1,4), 'value': TC_MLC_sorted}
		df_layers={'TC_MLC_ranking': range(1,4), 'layer_id': TC_MLC_sorted_ind}
		TC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_MLC_ranking'])
		top_k_TC_MLC = pd.merge(TC_MLC_node_names, pd.DataFrame(data=df_values), on='TC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_TC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_TC_MLC.to_string(index=False))
		
		### receiver
		TC_sorted = -np.sort(-TC_receiver, axis=None)
		TC_sorted_ind = np.argsort(-TC_receiver, axis=None)

		# top k JCs
		print('\n-----Total communicability-----\n-----receiver communicabilities\n')
		print_file.write('\n-----Total communicability-----\n-----receiver communicabilities\n')
		df_values={'TC_JC_ranking': range(1,k+1), 'value': TC_sorted[0:k]}
		df_nodes={'TC_JC_ranking': range(1,k+1), 'node_id': (TC_sorted_ind[0:k] % n)}
		df_layers={'TC_JC_ranking': range(1,k+1), 'layer_id': (TC_sorted_ind[0:k] // n)}
		TC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_JC_ranking'])
		TC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_JC_ranking'])
		top_k_TC_JC = pd.merge(pd.merge(TC_node_names, TC_layer_names, on='TC_JC_ranking'),  pd.DataFrame(data=df_values), on='TC_JC_ranking')
		print('Joint centralities:\n', top_k_TC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_TC_JC.to_string(index=False))
				
		# top k MNCs
		TC_MNC = np.sum(TC_receiver.reshape((L,n)).T, axis=1)
		TC_MNC_sorted = -np.sort(-TC_MNC, axis=None)
		TC_MNC_sorted_ind = np.argsort(-TC_MNC, axis=None)
		df_values={'TC_MNC_ranking': range(1,k+1), 'value': TC_MNC_sorted[0:k]}
		df_nodes={'TC_MNC_ranking': range(1,k+1), 'node_id': TC_MNC_sorted_ind[0:k]}
		TC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_MNC_ranking'])
		top_k_TC_MNC = pd.merge(TC_MNC_node_names, pd.DataFrame(data=df_values), on='TC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_TC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_TC_MNC.to_string(index=False))

		# top k MLCs
		TC_MLC = np.sum(TC_receiver.reshape((L,n)).T, axis=0)
		TC_MLC_sorted = -np.sort(-TC_MLC, axis=None)
		TC_MLC_sorted_ind = np.argsort(-TC_MLC, axis=None)
		df_values={'TC_MLC_ranking': range(1,4), 'value': TC_MLC_sorted}
		df_layers={'TC_MLC_ranking': range(1,4), 'layer_id': TC_MLC_sorted_ind}
		TC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_MLC_ranking'])
		top_k_TC_MLC = pd.merge(TC_MLC_node_names, pd.DataFrame(data=df_values), on='TC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_TC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_TC_MLC.to_string(index=False))


		### Katz centrality ###
		### broadcaster
		KC_sorted = -np.sort(-KC_broadcaster, axis=None)
		KC_sorted_ind = np.argsort(-KC_broadcaster, axis=None)

		# top k JCs
		print('\n-----Katz centrality-----\n-----broadcaster centralities\n')
		print_file.write('\n-----Katz centrality-----\n-----broadcaster centralities\n')
		df_values={'KC_JC_ranking': range(1,k+1), 'value': KC_sorted[0:k]}
		df_nodes={'KC_JC_ranking': range(1,k+1), 'node_id': (KC_sorted_ind[0:k] % n)}
		df_layers={'KC_JC_ranking': range(1,k+1), 'layer_id': (KC_sorted_ind[0:k] // n)}
		KC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_JC_ranking'])
		KC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_JC_ranking'])
		top_k_KC_JC = pd.merge(pd.merge(KC_node_names, KC_layer_names, on='KC_JC_ranking'),  pd.DataFrame(data=df_values), on='KC_JC_ranking')
		print('Joint centralities:\n', top_k_KC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_KC_JC.to_string(index=False))
				
		# top k MNCs
		KC_MNC = np.sum(KC_broadcaster.reshape((L,n)).T, axis=1)
		KC_MNC_sorted = -np.sort(-KC_MNC, axis=None)
		KC_MNC_sorted_ind = np.argsort(-KC_MNC, axis=None)
		df_values={'KC_MNC_ranking': range(1,k+1), 'value': KC_MNC_sorted[0:k]}
		df_nodes={'KC_MNC_ranking': range(1,k+1), 'node_id': KC_MNC_sorted_ind[0:k]}
		KC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_MNC_ranking'])
		top_k_KC_MNC = pd.merge(KC_MNC_node_names, pd.DataFrame(data=df_values), on='KC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_KC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_KC_MNC.to_string(index=False))

		# top k MLCs
		KC_MLC = np.sum(KC_broadcaster.reshape((L,n)).T, axis=0)
		KC_MLC_sorted = -np.sort(-KC_MLC, axis=None)
		KC_MLC_sorted_ind = np.argsort(-KC_MLC, axis=None)
		df_values={'KC_MLC_ranking': range(1,4), 'value': KC_MLC_sorted}
		df_layers={'KC_MLC_ranking': range(1,4), 'layer_id': KC_MLC_sorted_ind}
		KC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_MLC_ranking'])
		top_k_KC_MLC = pd.merge(KC_MLC_node_names, pd.DataFrame(data=df_values), on='KC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_KC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_KC_MLC.to_string(index=False))
		
		### receiver
		KC_sorted = -np.sort(-KC_receiver, axis=None)
		KC_sorted_ind = np.argsort(-KC_receiver, axis=None)

		# top k JCs
		print('\n-----Katz centrality-----\n-----receiver centralities\n')
		print_file.write('\n-----Katz centrality-----\n-----receiver centralities\n')
		df_values={'KC_JC_ranking': range(1,k+1), 'value': KC_sorted[0:k]}
		df_nodes={'KC_JC_ranking': range(1,k+1), 'node_id': (KC_sorted_ind[0:k] % n)}
		df_layers={'KC_JC_ranking': range(1,k+1), 'layer_id': (KC_sorted_ind[0:k] // n)}
		KC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_JC_ranking'])
		KC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_JC_ranking'])
		top_k_KC_JC = pd.merge(pd.merge(KC_node_names, KC_layer_names, on='KC_JC_ranking'),  pd.DataFrame(data=df_values), on='KC_JC_ranking')
		print('Joint centralities:\n', top_k_KC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_KC_JC.to_string(index=False))
				
		# top k MNCs
		KC_MNC = np.sum(KC_receiver.reshape((L,n)).T, axis=1)
		KC_MNC_sorted = -np.sort(-KC_MNC, axis=None)
		KC_MNC_sorted_ind = np.argsort(-KC_MNC, axis=None)
		df_values={'KC_MNC_ranking': range(1,k+1), 'value': KC_MNC_sorted[0:k]}
		df_nodes={'KC_MNC_ranking': range(1,k+1), 'node_id': KC_MNC_sorted_ind[0:k]}
		KC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_MNC_ranking'])
		top_k_KC_MNC = pd.merge(KC_MNC_node_names, pd.DataFrame(data=df_values), on='KC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_KC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_KC_MNC.to_string(index=False))

		# top k MLCs
		KC_MLC = np.sum(KC_receiver.reshape((L,n)).T, axis=0)
		KC_MLC_sorted = -np.sort(-KC_MLC, axis=None)
		KC_MLC_sorted_ind = np.argsort(-KC_MLC, axis=None)
		df_values={'KC_MLC_ranking': range(1,4), 'value': KC_MLC_sorted}
		df_layers={'KC_MLC_ranking': range(1,4), 'layer_id': KC_MLC_sorted_ind}
		KC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_MLC_ranking'])
		top_k_KC_MLC = pd.merge(KC_MLC_node_names, pd.DataFrame(data=df_values), on='KC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_KC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_KC_MLC.to_string(index=False))


	
	
	#######################################################################
	#######################################################################
	########## Lanczos on the symmetric bipartite representation ##########
	#######################################################################
	#######################################################################
	elif sym_bipartite_graph==1:

		'''

		#############################
		### u^T f(A) u quantities ###
		#############################

		# set centrality of isolated nodes to 1
		SC_gauss_lower = np.ones([2*nL,1])
		SC_gauss_radau_lower = np.ones([2*nL,1])
		SC_gauss_radau_upper = np.ones([2*nL,1])
		SC_gauss_lobatto_upper = np.ones([2*nL,1])

		SCres_gauss_lower = np.ones([2*nL,1])
		SCres_gauss_radau_lower = np.ones([2*nL,1])
		SCres_gauss_radau_upper = np.ones([2*nL,1])
		SCres_gauss_lobatto_upper = np.ones([2*nL,1])

		A_bipartite = spsp.bmat([[None, A], [A.T, None]])


		# estimate extremal eigenvalues
		U, T = lanczos_tridiag(A_bipartite, np.ones([2*nL,1]), 20)

		lamb, phi = la.eig(T)
		lamb_min = min(lamb)
		lamb_max = max(lamb)

		alpha_resolvent = alpha_const/lamb_max
		beta_subgraph = beta_const/lamb_max


		print('Subgraph and resolvent-based subgraph centrality. Looping over nodes-layer pairs. Progress:')

		# loop over non-isolated node-layer pairs
		i=0
		for node_id in range(2*nL):

			sys.stdout.write("\r{}%".format(100*((i+1)/(2*nL))))
			sys.stdout.flush()

			u = np.zeros([2*nL,1])
			u[node_id]=1

			U, T = lanczos_tridiag(A_bipartite, u, maxit_quadrature)
			T_copy1 = np.copy(T)
			T_copy2 = np.copy(T)
			T_copy3 = np.copy(T)
			T_copy4 = np.copy(T)
			T_copy5 = np.copy(T)
			T_copy6 = np.copy(T)
			T_copy7 = np.copy(T)


			# subgraph centrality quadrature rules
			SC_gauss_lower[node_id] = gauss_subgraph(T, beta_subgraph)
			SC_gauss_radau_lower[node_id] = gauss_radau_subgraph(T_copy1, beta_subgraph, lamb_min)
			SC_gauss_radau_upper[node_id] = gauss_radau_subgraph(T_copy2, beta_subgraph, lamb_max)
			SC_gauss_lobatto_upper[node_id] = gauss_lobatto_subgraph(T_copy3, beta_subgraph, lamb_min, lamb_max)

			# resolvent-based subgraph centrality quadrature rules
			SCres_gauss_lower[node_id] = gauss_resolvent(T_copy4, alpha_resolvent)
			SCres_gauss_radau_lower[node_id] = gauss_radau_resolvent(T_copy5, alpha_resolvent, lamb_min)
			SCres_gauss_radau_upper[node_id] = gauss_radau_resolvent(T_copy6, alpha_resolvent, lamb_max)
			SCres_gauss_lobatto_upper[node_id] = gauss_lobatto_resolvent(T_copy7, alpha_resolvent, lamb_min, lamb_max)

			i+=1

		SC_broadcaster = SC_gauss_radau_lower[0:nL]
		SC_receiver = SC_gauss_radau_lower[nL:2*nL]
		SCres_broadcaster = SCres_gauss_radau_lower[0:nL]
		SCres_receiver = SCres_gauss_radau_lower[nL:2*nL]

		'''

		########################
		### f(A)b quantities ###
		########################

		b = np.ones([nL,1])
		reorth=1

		####################################
		##### broadcaster centralities #####

		V, H = arnoldi(A, b, maxit_fAb, reorth)

		lamb, phi = la.eig(H[0:maxit_fAb, 0:maxit_fAb])
		lamb_max = max(lamb).real

		### Total communicability ###

		beta_subgraph = beta_const/lamb_max

		TC_broadcaster = expAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, beta_subgraph)

		### Katz centrality ###

		alpha_resolvent = alpha_const/lamb_max

		KC_broadcaster = resolventAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, alpha_resolvent)

		#################################
		##### receiver centralities #####

		V, H = arnoldi(A.T, b, maxit_fAb, reorth)

		lamb, phi = la.eig(H[0:maxit_fAb, 0:maxit_fAb])
		lamb_max = max(lamb).real

		### Total communicability ###

		beta_subgraph = beta_const/lamb_max

		TC_receiver = expAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, beta_subgraph)

		### Katz centrality ###

		alpha_resolvent = alpha_const/lamb_max

		KC_receiver = resolventAb_unsym(V[:, 0:maxit_fAb], H[0:maxit_fAb, 0:maxit_fAb], b, alpha_resolvent)
		


		############################################
		### print and write centralities to file ###
		############################################
		
		# printing to file
		directory = 'results'

		if not os.path.exists(directory):
			os.makedirs(directory)

		print_file = open("%s/centralities_%s_omega_%s_sym_bipartite_%d.txt" % (directory, saveStr, omega, sym_bipartite_graph), "w")
		print_file.write('Parameters:\nalpha_const=%f, beta_const=%f, maxit_quadrature=%d, maxit_fAb=%d, top_k_centralities=%d\n\n' % (alpha_const, beta_const, maxit_quadrature, maxit_fAb, top_k_centralities))

		### Print top k centralities ###
		k = top_k_centralities
		
		'''

		### Subgraph centrality ###
		### broadcaster
		SC_sorted = -np.sort(-SC_broadcaster, axis=None)
		SC_sorted_ind = np.argsort(-SC_broadcaster, axis=None)

		# top k JCs
		print('\n-----Subgraph centrality-----\n-----broadcaster centralities\n')
		print_file.write('\n-----Subgraph centrality-----\n-----broadcaster centralities\n')
		df_values={'SC_JC_ranking': range(1,k+1), 'value': SC_sorted[0:k]}
		df_nodes={'SC_JC_ranking': range(1,k+1), 'node_id': (SC_sorted_ind[0:k] % n)}
		df_layers={'SC_JC_ranking': range(1,k+1), 'layer_id': (SC_sorted_ind[0:k] // n)}
		SC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SC_JC_ranking'])
		SC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_JC_ranking'])
		top_k_SC_JC = pd.merge(pd.merge(SC_node_names, SC_layer_names, on='SC_JC_ranking'),  pd.DataFrame(data=df_values), on='SC_JC_ranking')
		print('Joint centralities:\n', top_k_SC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_SC_JC.to_string(index=False))
				
		# top k MNCs
		SC_MNC = np.sum(SC_broadcaster.reshape((L,n)).T, axis=1)
		SC_MNC_sorted = -np.sort(-SC_MNC, axis=None)
		SC_MNC_sorted_ind = np.argsort(-SC_MNC, axis=None)
		df_values={'SC_MNC_ranking': range(1,k+1), 'value': SC_MNC_sorted[0:k]}
		df_nodes={'SC_MNC_ranking': range(1,k+1), 'node_id': SC_MNC_sorted_ind[0:k]}
		SC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SC_MNC_ranking'])
		top_k_SC_MNC = pd.merge(SC_MNC_node_names, pd.DataFrame(data=df_values), on='SC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_SC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_SC_MNC.to_string(index=False))

		# top k MLCs
		SC_MLC = np.sum(SC_broadcaster.reshape((L,n)).T, axis=0)
		SC_MLC_sorted = -np.sort(-SC_MLC, axis=None)
		SC_MLC_sorted_ind = np.argsort(-SC_MLC, axis=None)
		df_values={'SC_MLC_ranking': range(1,4), 'value': SC_MLC_sorted}
		df_layers={'SC_MLC_ranking': range(1,4), 'layer_id': SC_MLC_sorted_ind}
		SC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_MLC_ranking'])
		top_k_SC_MLC = pd.merge(SC_MLC_node_names, pd.DataFrame(data=df_values), on='SC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_SC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_SC_MLC.to_string(index=False))
		
		### receiver
		SC_sorted = -np.sort(-SC_receiver, axis=None)
		SC_sorted_ind = np.argsort(-SC_receiver, axis=None)

		# top k JCs
		print('\n-----Subgraph centrality-----\n-----receiver centralities\n')
		print_file.write('\n-----Subgraph centrality-----\n-----receiver centralities\n')
		df_values={'SC_JC_ranking': range(1,k+1), 'value': SC_sorted[0:k]}
		df_nodes={'SC_JC_ranking': range(1,k+1), 'node_id': (SC_sorted_ind[0:k] % n)}
		df_layers={'SC_JC_ranking': range(1,k+1), 'layer_id': (SC_sorted_ind[0:k] // n)}
		SC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SC_JC_ranking'])
		SC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_JC_ranking'])
		top_k_SC_JC = pd.merge(pd.merge(SC_node_names, SC_layer_names, on='SC_JC_ranking'),  pd.DataFrame(data=df_values), on='SC_JC_ranking')
		print('Joint centralities:\n', top_k_SC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_SC_JC.to_string(index=False))
				
		# top k MNCs
		SC_MNC = np.sum(SC_receiver.reshape((L,n)).T, axis=1)
		SC_MNC_sorted = -np.sort(-SC_MNC, axis=None)
		SC_MNC_sorted_ind = np.argsort(-SC_MNC, axis=None)
		df_values={'SC_MNC_ranking': range(1,k+1), 'value': SC_MNC_sorted[0:k]}
		df_nodes={'SC_MNC_ranking': range(1,k+1), 'node_id': SC_MNC_sorted_ind[0:k]}
		SC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SC_MNC_ranking'])
		top_k_SC_MNC = pd.merge(SC_MNC_node_names, pd.DataFrame(data=df_values), on='SC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_SC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_SC_MNC.to_string(index=False))

		# top k MLCs
		SC_MLC = np.sum(SC_receiver.reshape((L,n)).T, axis=0)
		SC_MLC_sorted = -np.sort(-SC_MLC, axis=None)
		SC_MLC_sorted_ind = np.argsort(-SC_MLC, axis=None)
		df_values={'SC_MLC_ranking': range(1,4), 'value': SC_MLC_sorted}
		df_layers={'SC_MLC_ranking': range(1,4), 'layer_id': SC_MLC_sorted_ind}
		SC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SC_MLC_ranking'])
		top_k_SC_MLC = pd.merge(SC_MLC_node_names, pd.DataFrame(data=df_values), on='SC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_SC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_SC_MLC.to_string(index=False))


		### Resolvent-based subgraph centrality ###
		### broadcaster
		SCres_sorted = -np.sort(-SCres_broadcaster, axis=None)
		SCres_sorted_ind = np.argsort(-SCres_broadcaster, axis=None)

		# top k JCs
		print('\n-----Resolvent-based subgraph centrality-----\n-----broadcaster centralities\n')
		print_file.write('\n-----Resolvent-based subgraph centrality-----\n-----broadcaster centralities\n')
		df_values={'SCres_JC_ranking': range(1,k+1), 'value': SCres_sorted[0:k]}
		df_nodes={'SCres_JC_ranking': range(1,k+1), 'node_id': (SCres_sorted_ind[0:k] % n)}
		df_layers={'SCres_JC_ranking': range(1,k+1), 'layer_id': (SCres_sorted_ind[0:k] // n)}
		SCres_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SCres_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SCres_JC_ranking'])
		SCres_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_JC_ranking'])
		top_k_SCres_JC = pd.merge(pd.merge(SCres_node_names, SCres_layer_names, on='SCres_JC_ranking'),  pd.DataFrame(data=df_values), on='SCres_JC_ranking')
		print('Joint centralities:\n', top_k_SCres_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_SCres_JC.to_string(index=False))
				
		# top k MNCs
		SCres_MNC = np.sum(SCres_broadcaster.reshape((L,n)).T, axis=1)
		SCres_MNC_sorted = -np.sort(-SCres_MNC, axis=None)
		SCres_MNC_sorted_ind = np.argsort(-SCres_MNC, axis=None)
		df_values={'SCres_MNC_ranking': range(1,k+1), 'value': SCres_MNC_sorted[0:k]}
		df_nodes={'SCres_MNC_ranking': range(1,k+1), 'node_id': SCres_MNC_sorted_ind[0:k]}
		SCres_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SCres_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SCres_MNC_ranking'])
		top_k_SCres_MNC = pd.merge(SCres_MNC_node_names, pd.DataFrame(data=df_values), on='SCres_MNC_ranking')
		print('Marginal node centralities:\n', top_k_SCres_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_SCres_MNC.to_string(index=False))

		# top k MLCs
		SCres_MLC = np.sum(SCres_broadcaster.reshape((L,n)).T, axis=0)
		SCres_MLC_sorted = -np.sort(-SCres_MLC, axis=None)
		SCres_MLC_sorted_ind = np.argsort(-SCres_MLC, axis=None)
		df_values={'SCres_MLC_ranking': range(1,4), 'value': SCres_MLC_sorted}
		df_layers={'SCres_MLC_ranking': range(1,4), 'layer_id': SCres_MLC_sorted_ind}
		SCres_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_MLC_ranking'])
		top_k_SCres_MLC = pd.merge(SCres_MLC_node_names, pd.DataFrame(data=df_values), on='SCres_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_SCres_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_SCres_MLC.to_string(index=False))
		
		### receiver
		SCres_sorted = -np.sort(-SCres_receiver, axis=None)
		SCres_sorted_ind = np.argsort(-SCres_receiver, axis=None)

		# top k JCs
		print('\n-----Resolvent-based subgraph centrality-----\n-----receiver centralities\n')
		print_file.write('\n-----Resolvent-based subgraph centrality-----\n-----receiver centralities\n')
		df_values={'SCres_JC_ranking': range(1,k+1), 'value': SCres_sorted[0:k]}
		df_nodes={'SCres_JC_ranking': range(1,k+1), 'node_id': (SCres_sorted_ind[0:k] % n)}
		df_layers={'SCres_JC_ranking': range(1,k+1), 'layer_id': (SCres_sorted_ind[0:k] // n)}
		SCres_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SCres_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SCres_JC_ranking'])
		SCres_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_JC_ranking'])
		top_k_SCres_JC = pd.merge(pd.merge(SCres_node_names, SCres_layer_names, on='SCres_JC_ranking'),  pd.DataFrame(data=df_values), on='SCres_JC_ranking')
		print('Joint centralities:\n', top_k_SCres_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_SCres_JC.to_string(index=False))
				
		# top k MNCs
		SCres_MNC = np.sum(SCres_receiver.reshape((L,n)).T, axis=1)
		SCres_MNC_sorted = -np.sort(-SCres_MNC, axis=None)
		SCres_MNC_sorted_ind = np.argsort(-SCres_MNC, axis=None)
		df_values={'SCres_MNC_ranking': range(1,k+1), 'value': SCres_MNC_sorted[0:k]}
		df_nodes={'SCres_MNC_ranking': range(1,k+1), 'node_id': SCres_MNC_sorted_ind[0:k]}
		SCres_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['SCres_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['SCres_MNC_ranking'])
		top_k_SCres_MNC = pd.merge(SCres_MNC_node_names, pd.DataFrame(data=df_values), on='SCres_MNC_ranking')
		print('Marginal node centralities:\n', top_k_SCres_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_SCres_MNC.to_string(index=False))

		# top k MLCs
		SCres_MLC = np.sum(SCres_receiver.reshape((L,n)).T, axis=0)
		SCres_MLC_sorted = -np.sort(-SCres_MLC, axis=None)
		SCres_MLC_sorted_ind = np.argsort(-SCres_MLC, axis=None)
		df_values={'SCres_MLC_ranking': range(1,4), 'value': SCres_MLC_sorted}
		df_layers={'SCres_MLC_ranking': range(1,4), 'layer_id': SCres_MLC_sorted_ind}
		SCres_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['SCres_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['SCres_MLC_ranking'])
		top_k_SCres_MLC = pd.merge(SCres_MLC_node_names, pd.DataFrame(data=df_values), on='SCres_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_SCres_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_SCres_MLC.to_string(index=False))
		
		'''
		
		### Total communicability ###
		### broadcaster
		TC_sorted = -np.sort(-TC_broadcaster, axis=None)
		TC_sorted_ind = np.argsort(-TC_broadcaster, axis=None)

		# top k JCs
		print('\n-----Total communicability-----\n-----broadcaster communicabilities\n')
		print_file.write('\n-----Total communicability-----\n-----broadcaster communicabilities\n')
		df_values={'TC_JC_ranking': range(1,k+1), 'value': TC_sorted[0:k]}
		df_nodes={'TC_JC_ranking': range(1,k+1), 'node_id': (TC_sorted_ind[0:k] % n)}
		df_layers={'TC_JC_ranking': range(1,k+1), 'layer_id': (TC_sorted_ind[0:k] // n)}
		TC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_JC_ranking'])
		TC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_JC_ranking'])
		top_k_TC_JC = pd.merge(pd.merge(TC_node_names, TC_layer_names, on='TC_JC_ranking'),  pd.DataFrame(data=df_values), on='TC_JC_ranking')
		print('Joint centralities:\n', top_k_TC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_TC_JC.to_string(index=False))
				
		# top k MNCs
		TC_MNC = np.sum(TC_broadcaster.reshape((L,n)).T, axis=1)
		TC_MNC_sorted = -np.sort(-TC_MNC, axis=None)
		TC_MNC_sorted_ind = np.argsort(-TC_MNC, axis=None)
		df_values={'TC_MNC_ranking': range(1,k+1), 'value': TC_MNC_sorted[0:k]}
		df_nodes={'TC_MNC_ranking': range(1,k+1), 'node_id': TC_MNC_sorted_ind[0:k]}
		TC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_MNC_ranking'])
		top_k_TC_MNC = pd.merge(TC_MNC_node_names, pd.DataFrame(data=df_values), on='TC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_TC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_TC_MNC.to_string(index=False))

		# top k MLCs
		TC_MLC = np.sum(TC_broadcaster.reshape((L,n)).T, axis=0)
		TC_MLC_sorted = -np.sort(-TC_MLC, axis=None)
		TC_MLC_sorted_ind = np.argsort(-TC_MLC, axis=None)
		df_values={'TC_MLC_ranking': range(1,4), 'value': TC_MLC_sorted}
		df_layers={'TC_MLC_ranking': range(1,4), 'layer_id': TC_MLC_sorted_ind}
		TC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_MLC_ranking'])
		top_k_TC_MLC = pd.merge(TC_MLC_node_names, pd.DataFrame(data=df_values), on='TC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_TC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_TC_MLC.to_string(index=False))
		
		### receiver
		TC_sorted = -np.sort(-TC_receiver, axis=None)
		TC_sorted_ind = np.argsort(-TC_receiver, axis=None)

		# top k JCs
		print('\n-----Total communicability-----\n-----receiver communicabilities\n')
		print_file.write('\n-----Total communicability-----\n-----receiver communicabilities\n')
		df_values={'TC_JC_ranking': range(1,k+1), 'value': TC_sorted[0:k]}
		df_nodes={'TC_JC_ranking': range(1,k+1), 'node_id': (TC_sorted_ind[0:k] % n)}
		df_layers={'TC_JC_ranking': range(1,k+1), 'layer_id': (TC_sorted_ind[0:k] // n)}
		TC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_JC_ranking'])
		TC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_JC_ranking'])
		top_k_TC_JC = pd.merge(pd.merge(TC_node_names, TC_layer_names, on='TC_JC_ranking'),  pd.DataFrame(data=df_values), on='TC_JC_ranking')
		print('Joint centralities:\n', top_k_TC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_TC_JC.to_string(index=False))
				
		# top k MNCs
		TC_MNC = np.sum(TC_receiver.reshape((L,n)).T, axis=1)
		TC_MNC_sorted = -np.sort(-TC_MNC, axis=None)
		TC_MNC_sorted_ind = np.argsort(-TC_MNC, axis=None)
		df_values={'TC_MNC_ranking': range(1,k+1), 'value': TC_MNC_sorted[0:k]}
		df_nodes={'TC_MNC_ranking': range(1,k+1), 'node_id': TC_MNC_sorted_ind[0:k]}
		TC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['TC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['TC_MNC_ranking'])
		top_k_TC_MNC = pd.merge(TC_MNC_node_names, pd.DataFrame(data=df_values), on='TC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_TC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_TC_MNC.to_string(index=False))

		# top k MLCs
		TC_MLC = np.sum(TC_receiver.reshape((L,n)).T, axis=0)
		TC_MLC_sorted = -np.sort(-TC_MLC, axis=None)
		TC_MLC_sorted_ind = np.argsort(-TC_MLC, axis=None)
		df_values={'TC_MLC_ranking': range(1,4), 'value': TC_MLC_sorted}
		df_layers={'TC_MLC_ranking': range(1,4), 'layer_id': TC_MLC_sorted_ind}
		TC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['TC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['TC_MLC_ranking'])
		top_k_TC_MLC = pd.merge(TC_MLC_node_names, pd.DataFrame(data=df_values), on='TC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_TC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_TC_MLC.to_string(index=False))


		### Katz centrality ###
		### broadcaster
		KC_sorted = -np.sort(-KC_broadcaster, axis=None)
		KC_sorted_ind = np.argsort(-KC_broadcaster, axis=None)

		# top k JCs
		print('\n-----Katz centrality-----\n-----broadcaster centralities\n')
		print_file.write('\n-----Katz centrality-----\n-----broadcaster centralities\n')
		df_values={'KC_JC_ranking': range(1,k+1), 'value': KC_sorted[0:k]}
		df_nodes={'KC_JC_ranking': range(1,k+1), 'node_id': (KC_sorted_ind[0:k] % n)}
		df_layers={'KC_JC_ranking': range(1,k+1), 'layer_id': (KC_sorted_ind[0:k] // n)}
		KC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_JC_ranking'])
		KC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_JC_ranking'])
		top_k_KC_JC = pd.merge(pd.merge(KC_node_names, KC_layer_names, on='KC_JC_ranking'),  pd.DataFrame(data=df_values), on='KC_JC_ranking')
		print('Joint centralities:\n', top_k_KC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_KC_JC.to_string(index=False))
				
		# top k MNCs
		KC_MNC = np.sum(KC_broadcaster.reshape((L,n)).T, axis=1)
		KC_MNC_sorted = -np.sort(-KC_MNC, axis=None)
		KC_MNC_sorted_ind = np.argsort(-KC_MNC, axis=None)
		df_values={'KC_MNC_ranking': range(1,k+1), 'value': KC_MNC_sorted[0:k]}
		df_nodes={'KC_MNC_ranking': range(1,k+1), 'node_id': KC_MNC_sorted_ind[0:k]}
		KC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_MNC_ranking'])
		top_k_KC_MNC = pd.merge(KC_MNC_node_names, pd.DataFrame(data=df_values), on='KC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_KC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_KC_MNC.to_string(index=False))

		# top k MLCs
		KC_MLC = np.sum(KC_broadcaster.reshape((L,n)).T, axis=0)
		KC_MLC_sorted = -np.sort(-KC_MLC, axis=None)
		KC_MLC_sorted_ind = np.argsort(-KC_MLC, axis=None)
		df_values={'KC_MLC_ranking': range(1,4), 'value': KC_MLC_sorted}
		df_layers={'KC_MLC_ranking': range(1,4), 'layer_id': KC_MLC_sorted_ind}
		KC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_MLC_ranking'])
		top_k_KC_MLC = pd.merge(KC_MLC_node_names, pd.DataFrame(data=df_values), on='KC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_KC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_KC_MLC.to_string(index=False))
		
		### receiver
		KC_sorted = -np.sort(-KC_receiver, axis=None)
		KC_sorted_ind = np.argsort(-KC_receiver, axis=None)

		# top k JCs
		print('\n-----Katz centrality-----\n-----receiver centralities\n')
		print_file.write('\n-----Katz centrality-----\n-----receiver centralities\n')
		df_values={'KC_JC_ranking': range(1,k+1), 'value': KC_sorted[0:k]}
		df_nodes={'KC_JC_ranking': range(1,k+1), 'node_id': (KC_sorted_ind[0:k] % n)}
		df_layers={'KC_JC_ranking': range(1,k+1), 'layer_id': (KC_sorted_ind[0:k] // n)}
		KC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_JC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_JC_ranking'])
		KC_layer_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_JC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_JC_ranking'])
		top_k_KC_JC = pd.merge(pd.merge(KC_node_names, KC_layer_names, on='KC_JC_ranking'),  pd.DataFrame(data=df_values), on='KC_JC_ranking')
		print('Joint centralities:\n', top_k_KC_JC.to_string(index=False))
		print_file.write('Joint centralities:\n')
		print_file.write(top_k_KC_JC.to_string(index=False))
				
		# top k MNCs
		KC_MNC = np.sum(KC_receiver.reshape((L,n)).T, axis=1)
		KC_MNC_sorted = -np.sort(-KC_MNC, axis=None)
		KC_MNC_sorted_ind = np.argsort(-KC_MNC, axis=None)
		df_values={'KC_MNC_ranking': range(1,k+1), 'value': KC_MNC_sorted[0:k]}
		df_nodes={'KC_MNC_ranking': range(1,k+1), 'node_id': KC_MNC_sorted_ind[0:k]}
		KC_MNC_node_names = pd.merge(pd.DataFrame(data=df_nodes), userIDList, on='node_id')[['KC_MNC_ranking', 'node_id', 'user_screen_name']].sort_values(by=['KC_MNC_ranking'])
		top_k_KC_MNC = pd.merge(KC_MNC_node_names, pd.DataFrame(data=df_values), on='KC_MNC_ranking')
		print('Marginal node centralities:\n', top_k_KC_MNC.to_string(index=False))
		print_file.write('\nMarginal node centralities:\n')
		print_file.write(top_k_KC_MNC.to_string(index=False))

		# top k MLCs
		KC_MLC = np.sum(KC_receiver.reshape((L,n)).T, axis=0)
		KC_MLC_sorted = -np.sort(-KC_MLC, axis=None)
		KC_MLC_sorted_ind = np.argsort(-KC_MLC, axis=None)
		df_values={'KC_MLC_ranking': range(1,4), 'value': KC_MLC_sorted}
		df_layers={'KC_MLC_ranking': range(1,4), 'layer_id': KC_MLC_sorted_ind}
		KC_MLC_node_names = pd.merge(pd.DataFrame(data=df_layers), layerIDList, on='layer_id')[['KC_MLC_ranking', 'layer_id', 'layer_name']].sort_values(by=['KC_MLC_ranking'])
		top_k_KC_MLC = pd.merge(KC_MLC_node_names, pd.DataFrame(data=df_values), on='KC_MLC_ranking')
		print('Marginal layer centralities:\n', top_k_KC_MLC.to_string(index=False))
		print_file.write('\nMarginal layer centralities:\n')
		print_file.write(top_k_KC_MLC.to_string(index=False))
		
		return TC_MNC


