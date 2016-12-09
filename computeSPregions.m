%compute regions with spectral clustering
function [SPtree,leafMap, nLeaves] = computeSPregions(img)

[m, n, d] = size(img);
sizeLim = 300;

if m>sizeLim || n>sizeLim
    morg = m;
    norg = n;
    img = imresize(img, sizeLim/m);
    [m, n, d] = size(img);
else
    morg = [];
    norg = [];
end

Type = 3;
maxArea = 0.005*(m*n);
diff   = eps;
nLeaves = 1;
spectral = 0;
k = 2;
leafMap = ones(size(img, 1)*size(img,2),1 , 'uint16');
tree = [];
treeLevel = 1;
nodeNum = 1;
if spectral
    cte = 0.01;
else
    cte = 1.6;
end

% convert into list of data points
Data = reshape(img, 1, m * n, []);

if d >= 2
    Data = (squeeze(Data))';
end

% convert to double and normalize to [0,1]
Data = double(Data);
Data = normalizeData(Data);
Data = Data + cte;

% now for the clustering
fprintf('Creating Similarity Graph...\n');

% Compute 4-connected adjacency matrix
I_size = m*n;

% 1-off diagonal elements
V = repmat([ones(m-1,1); 0],n, 1);
V = V(1:end-1); % remove last zero

% n-off diagonal elements
U = ones(m*(n-1), 1);

% get the upper triangular part of the matrix
W = sparse(1:(I_size-1),    2:I_size, V, I_size, I_size)...
    + sparse(1:(I_size-m),(m+1):I_size, U, I_size, I_size);

%add weigths from the density image
W = bsxfun(@times,W,Data);

%make W symmetric
W = W + W';

while(1)
    
    bins = unique(leafMap);
    AsHist = histc(double(leafMap),bins);
    ToSplit = find(AsHist > maxArea);
    
    if ~isempty(ToSplit)
        for j = 1:numel(ToSplit)
            
            ToSplitIdx = leafMap == bins(ToSplit(j));
            Wleaf = W(ToSplitIdx,ToSplitIdx);
            if spectral
                % calculate weighted degree matrix
                degs = sum(Wleaf, 2);
                Dleaf = sparse(1:size(Wleaf, 1), 1:size(Wleaf, 2), degs);
                
                % compute unnormalized Laplacian
                L = Dleaf - Wleaf;
                
                % compute normalized Laplacian if needed
                switch Type
                    case 2
                        % avoid dividing by zero
                        degs(degs == 0) = eps;
                        % calculate inverse of D
                        Dleaf = spdiags(1./degs, 0, size(Dleaf, 1), size(Dleaf, 2));
                        
                        % calculate normalized Laplacian
                        L = Dleaf * L;
                        
                    case 3
                        % avoid dividing by zero
                        degs(degs == 0) = eps;
                        % calculate D^(-1/2)
                        Dleaf = spdiags(1./(degs.^0.5), 0, size(Dleaf, 1), size(Dleaf, 2));
                        
                        % calculate normalized Laplacian
                        L = Dleaf * L * Dleaf;
                end
                
                % compute the eigenvectors corresponding to the k smallest
                % eigenvalues
                [U,~] = eigs(L, k, diff);
                
                % in case of the Jordan-Weiss algorithm, we need to normalize
                % the eigenvectors row-wise
                if Type == 3
                    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
                end
                
                % now use the k-means algorithm to cluster U row-wise
                % C will be a n-by-1 matrix containing the cluster number for
                % each data point
                C = kmeans(U, k, 'start', 'cluster', ...
                    'EmptyAction', 'singleton');
                
                % now convert C to a n-by-k matrix containing the k indicator
                % vectors as columns
                C = sparse(1:size(Dleaf, 1), C, 1);
                
                % convert and restore full size
                regionMap = convertClusterVector(C);
            else
                [regionMap,~]=graclus_mex(1000*Wleaf,nnz(Wleaf),k,0,2,0);
            end
            
            %add the new leafs            
            ToSplitIdx = find(ToSplitIdx == 1);
            
            nodeNum = nodeNum+1;
            tree = [tree ; bins(ToSplit(j)) nodeNum treeLevel];
            leafMap(ToSplitIdx(regionMap == 1)) = nodeNum; %cluster 1
            
            nodeNum = nodeNum+1;
            leafMap(ToSplitIdx(regionMap == 2)) = nodeNum; %cluster 2
            tree = [tree ; bins(ToSplit(j)) nodeNum treeLevel];            
            
            nLeaves = nLeaves + 1;
            
        end
        treeLevel = treeLevel + 1;
    else
        break;
    end
     
end

leafMap = reshape(leafMap,m,n);
nLevels = treeLevel - 1;

%refences; i.e. leaves(i) will now be numbered i.
leaves = unique(leafMap);

%rename leaves
for i = 1:nLeaves
    leafMap(leafMap==leaves(i)) = i;
end

nodesMapping = zeros(nodeNum, 1);
nodesMapping(1:nLeaves) = leaves;
mapPos = nLeaves + 1;
SPtree = zeros(nLeaves-1,3); %Linkage structure
clusterNum = 1; %Used to index linkageStruct
for i = nLevels:-1:1
    nodesInLevel = find(tree(:,3) == i); %Acutally, the rows of Tree in that level
    parents = unique(tree(nodesInLevel,1));
    for j = 1:numel(parents)
        parentsRows = find(tree(:,1) == parents(j));
        pairOfChildren = tree(parentsRows,2);
        pairOfChildrenMap = [find(nodesMapping == pairOfChildren(1));...
            find(nodesMapping == pairOfChildren(2))];
        SPtree(clusterNum,:) = [pairOfChildrenMap' nLevels-i+1];
        nodesMapping(mapPos) = parents(j);
        clusterNum = clusterNum + 1;
        mapPos = mapPos + 1;
    end
    
end

if ~isempty(morg)
    leafMap = imresize(leafMap,[morg norg],'nearest');
end

% centroids = zeros(nLeafs, 2);
% 
% %build linkage tree
% for i = 1:nLeafs
%     leaf = leafMap == i;
%     s = regionprops(leaf,'Centroid','PixelIdxList');
%     centroids(i,:) = s.Centroid;   
% end
% 
% D = squareform(pdist(centroids));
% SPtree = linkage(D);

function normalizedData = normalizeData(Data)
% NORMALIZEDATA Normalized data matrix
%   normalizeData(Data) normalizes the d-by-n matrix Data, so that
%   the minimum value of each dimension and for all data points is 0 and
%   the maximum value respectively is 1.

a = 0;
b = 1;

minData = min(Data, [], 2);
maxData = max(Data, [], 2);

r = (a-b) ./ (minData - maxData);
s = a - r .* minData;

normalizedData = repmat(r, 1, size(Data, 2)) .* Data + repmat(s, 1, size(Data, 2));

function indMatrix = convertClusterVector(M)
% CONVERTCLUSTERVECTOR
%   Converts between row vector with cluster number and indicator vector
%   matrix

if size(M, 2) > 1
    indMatrix = zeros(size(M, 1), 1);
    for ii = 1:size(M, 2)
        indMatrix(M(:, ii) == 1) = ii;
    end
else
    indMatrix = sparse(1:size(M, 1), M, 1);
end

