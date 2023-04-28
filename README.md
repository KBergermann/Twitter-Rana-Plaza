# Twitter-Rana-Plaza

This repository contains python3 and Matlab codes reproducing the results from the 

**Paper:**

[1] K. Bergermann & M. Wolter (2023) A Twitter network and discourse analysis of the Rana Plaza collapse. [arXiv:...](https://arxiv.org/pdf/....pdf).

**Requirements:**
 - The Matlab file "communities/GenLouvain.m" requires the code from [https://github.com/GenLouvain/GenLouvain](https://github.com/GenLouvain/GenLouvain) to be saved in the directory "communities/"

This repository contains:

**License:**
 - LICENSE: GNU General Public License v2.0
 
**Directory:**
 - networks_Matlab: .mat files containing 3 sparse adjacency matrices for the 3 layers "retweet", "reply", and "mention" for the years 2013-2022
 - networks_python: .npz files containing 3 sparse adjacency matrices in CSR format for the 3 layers "retweet", "reply", and "mention" for the years 2013-2022
 - centralities: files computing matrix-function based centrality measures for multiplex networks [2]
 - communities: files computing the Generalized Louvain community structure for multiplex networks [3]. The implementation of the method from [3], which is freely available must be downloaded prior to execution, see Requirements
 
**Scripts:**
 - centralities/compute_centralities.py: implements matrix-function based centrality measures for multiplex networks [2]. Codes also available [here](https://github.com/KBergermann/Urban-multiplex-networks)
 - centralities/compute_centralities-master.py: runs "compute_centralities" for specified networks
 - communities/GenLouvain.m: computes the Generalized Louvain community structure for multiplex networks [3] if GenLouvain codes are available, see Requirements.
 
**References:**
 - [1] K. Bergermann & M. Wolter (2023) A Twitter network and discourse analysis of the Rana Plaza collapse. [arXiv:...](https://arxiv.org/pdf/....pdf).
 - [2] Bergermann, K. & Stoll, M. (2022) Fast computation of matrix function-based centrality measures for layer-coupled multiplex networks. Physical Review E, 105(3), 034305. https://doi.org/10.1103/PhysRevE.105.034305
 - [3] Mucha, P. J., Richardson, T., Macon, K., Porter, M. A. & Onnela, J.-P. (2010) Community structure in time-dependent, multiscale, and multiplex networks. Science 328, 876-878.
