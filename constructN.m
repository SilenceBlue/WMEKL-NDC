function N = constructN(X,Y,options)
%	Usage:
%	N 2= constructW(X, Y,options)
%   X and Y are different class
%	X: Rows of vectors of data points. Each row is x_i
%   Y: Rows of vectors of data points. Each row is y_i
%   options: Struct value in Matlab. The fields in options that can be set:
%           Metric -  Choices are:
%               'Euclidean' - Will use the Euclidean distance of two data 
%                             points to evaluate the "closeness" between 
%                             them. [Default One]
%                  
%           NeighborMode -  Indicates how to construct the graph. Choices
%                           are: 
%                'KNN'            -  Put an edge between two nodes if and
%                                    only if they are among the k nearst
%                                    neighbors of each other. You are
%                                    required to provide the parameter k in
%                                    the options. [Default One]
%                                              
%           WeightMode   -  Indicates how to assign weights for each edge
%                           in the graph. Choices are:
%               'HeatKernel'   - If nodes i and j are connected, put weight
%                                W_ij = exp(-norm(x_i - x_j)/t). This
%                                weight mode can only be used under
%                                'Euclidean' metric and you are required to
%                                provide the parameter t.
%               
%            k         -   The parameter needed under 'KNN' NeighborMode.
%                          Default will be 5.
%            t         -   The parameter needed under 'HeatKernel'
%                          WeightMode. Default will be 1
%
%       
%    
%
%    Written by Bolu Wang, July/2018,
% 
%%  define parameters' Default-value
if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options) 
       error('parameter error!');
   end
end

% define options.Metric=================================================
if ~isfield(options,'Metric')
    options.Metric = 'Euclidean';
end


% define options.NeighborMode =================================================
if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

 if ~isfield(options,'nk')
     options.nk = 5;
 else if options.nk < 1
         options.k = 1;
%         options.nk = size([X;Y],1)-1;
     end
end

% define options.WeightMode =================================================

if ~isfield(options,'t')
    options.t = 1000000;
end


%%  constructB
[nX, ~] = size(X);
[nY, ~] = size(Y);
S  = [X;Y];
tolSmp = size(S, 1);
N = zeros(tolSmp);
%%  constructing the difference in value between two class samples matrix[('Euclidean' and 'Cosine')'s D]
ave_dist=0;   
if strcmpi(options.Metric,'Euclidean')
   % D为所有样本的距离矩阵
   D = zeros(tolSmp);
   for i=1:tolSmp-1
       for j=i+1:tolSmp
           D(i,j) = norm(S(i,:) - S(j,:));
           ave_dist= ave_dist+D(i,j);
       end
   end
   D = D+D';
end

%% constructing the Different Class Neighbor graph  ( XG, YG )
if strcmpi(options.NeighborMode,'knn')
    G = zeros(tolSmp,tolSmp);
    [~, idx] = sort(D, 2); % sort each row 近邻排序
    for i=1:tolSmp
       G(i,idx(i,1:options.nk+1)) = 1;
    end   
    G(nX+1:end,1:nX) = G(nX+1:end,1:nX)*-1;
    G(1:nX,nX+1:end) = G(1:nX,nX+1:end)*-1;
end 
G  = G - diag(diag(G));

%%  constructe Similarity matrices ( XB, YB )
if  strcmpi(options.WeightMode,'HeatKernel')  
    D = exp(-D.^2/ave_dist);
    N = D.*G;
    N = sparse(N);
end
if  strcmpi(options.WeightMode,'sign')  
    N = G;
    N = sparse(N);
end
end
