function leaves = getLeaves(tree,node)
%retrieves the index(es) of the leaf(s) node that make up any no 'node' in 
%a 'tree' with the Linkage structure.

nLeaves = size(tree,1) + 1;
notAllLeaves = false;

if node <= nLeaves
    leaves = node;
else
    children = tree(node-nLeaves,1:2);
    notLeaves = find(children > nLeaves);
    if ~isempty(notLeaves)
        notAllLeaves = true;
    end
    
    while notAllLeaves
        for i = 1:numel(notLeaves)
            newChildren = tree(children(notLeaves(i))-nLeaves,1:2);
            children = [children newChildren];
        end
        children(notLeaves) = [];
        notLeaves = find(children > nLeaves);
        if isempty(notLeaves)
            notAllLeaves = false;
        end
    end
    leaves = children;    
end
