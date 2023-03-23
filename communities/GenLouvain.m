
addpath('GenLouvain-master')
addpath('GenLouvain-master/HelperFunctions')

%% load layer adjacency matrices
filestr = 'rana_plaza_20140417-20140511';

fprintf(['\nYear ',filestr(12:15),'\n\n'])

load(['../networks_Matlab/',filestr,'_Aintra_retweet.mat'])
load(['../networks_Matlab/',filestr,'_Aintra_reply.mat'])
load(['../networks_Matlab/',filestr,'_Aintra_mention.mat'])

%% set multilayer network sizes
L = 3;
n = size(Aintra_retweet,1);

%% construct third order adjacency tensor
A = cell(3,1);
A{1} = Aintra_retweet;
A{2} = Aintra_reply;
A{3} = Aintra_mention;

%% build modularity matrix
[B,twom] = multiorddir_f(A,1,1); % A, gamma, omega

%% run the Generalized Louvain modularity optimization
[S,Q] = genlouvain(B,n);

%% extract community structure from genlouvain output vector S
[GC,GR] = groupcounts(S);
fprintf('\nFound %d communities of which %d communities consist of more than 30 node-layer pairs.\n', max(S), length(GC(GC>30)))

%% lists community numbers and sizes for the 10 largest communities
[GC_sorted, GC_ind] = sort(GC, 'descend');
fprintf('\nNumber of node-layer pairs in the 10 largest communities:\n')
fprintf('Community No.  |  Community count')
[GR(GC_ind(1:10)), GC_sorted(1:10)]
