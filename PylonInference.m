function [mask,labels, MSERtree, idMask] = PylonInference(img, prediction, svmTH, sizeMSER, r, additionalU, MSERtree)
%labels is the same size as r. labels(i) = 1 if r(i) has been picked, and 0
%otherwise
largeVal = -5000;
hardUVal = 100;
mask = zeros(size(img),'uint8');
idMask = zeros(size(img));

if nargin < 7
     MSERtree = buildPylonMSER(img,r,sizeMSER);
end

if nargin < 6
    additionalU = 0;
end

%hold on; plot(gt(:,1), gt(:,2),'*b','LineWidth',2)

labels = logical(false(size(r)));
for k = 1:numel(MSERtree.forest)
    treeInfo =  MSERtree.references{k};
    treeInfo = [treeInfo largeVal*ones(size(treeInfo,1),1)];
    %Add predictions
    for i = 1:size(treeInfo,1)
        if treeInfo(i) == 0
            continue;
        else
            elemPrediction = r == treeInfo(i);
            treeInfo(i, 2) = prediction(elemPrediction);
        end
    end
    %For debugging---------------------------------------------
    %              auxMask = uint8(zeros(size(img,1), size(img,2)));
    %asdf = MSERtree.nodesMapping{k};
    %                             sel = vl_erfill(img,treeInfo(asdf(1),1)) ;
    %                             auxMask(sel) = 1;
    %                             auxMask = bwmorph(auxMask, 'close');
    %                             figure, imagesc(auxMask)
    %----------------------------------------------------------
    hardU = [];
    if additionalU
        %if numel(find(treeInfo(1:MSERtree.nLeafs(k),2) > 0)) > 0 %If there is any GT in the leafs in this tree
            hardU = [zeros(1,MSERtree.nLeafs(k)) ; hardUVal+(treeInfo(1:MSERtree.nLeafs(k),2)>0)'];
        %end
    end
    
    if isempty(MSERtree.forest{k})
        
        [response,winner] = max(treeInfo(:,2));
        if response >= svmTH
%             auxMask = uint8(zeros(size(img,1), size(img,2)));
            sel = vl_erfill(img,treeInfo(winner,1)) ;
%             auxMask(sel) = 1;
%             auxMask = bwmorph(auxMask, 'close');
%             sel = find(auxMask == 1);
            mask(sel) = 1;
            correspondance = find(r == treeInfo(winner,1));
            labels(correspondance) = 1;
            idMask(sel) = 1*correspondance;
        end
        
    else
        %                             %---------------------------------------------------------------------------------------
        %                             k = 1;  MSERtree.nLeafs = 6; svmTH = 0;
        %                             MSERtree.forest = [1 2 1;3 4 1;5 7 2;8 6 2;9 10 3];
        %                             treeInfo = [1 0.5 ; 1 0.5 ; 1 -5 ; 1 -5 ; 1 -10 ; 1 10 ; 1 1.1 ; 1 -15 ; 1 3.2 ; 1 1 ; 1 1];
        %                             %            1       2        3      4      5      6      7       8       9     10    11
        %                             V = zeros(3,0);
        %                                 mappedLabels = pylonInference1Class( MSERtree.nLeafs(k), MSERtree.forest, -1*(treeInfo(:,2)-svmTH)', V);
        %                                  mappedLabels = pylonConvertLabels(mappedLabels, MSERtree.forest,  MSERtree.nLeafs(k));
        %                             %-----------------------------------------------------------------------------------------
        V = zeros(3,0);
        mappedLabels = pylonInference1Class( MSERtree.nLeafs(k), MSERtree.forest{k}, -1*(treeInfo(:,2)-svmTH)', V, -1*hardU);
        mappedLabels = pylonConvertLabels(mappedLabels, MSERtree.forest{k},  MSERtree.nLeafs(k));
        %           finalLabels = false(size(mappedLabels));
        %asd = uint8(zeros(size(img,1), size(img,2)));
        for i=1:length(mappedLabels)
            %               finalLabels(i) = mappedLabels(MSERtree.nodesMapping(i));
            
            %---------------
            %                     if k == 3
            %                         if treeInfo(i,1) ~= 0
            %                         auxMask = uint16(zeros(size(img,1), size(img,2)));
            %                             sel = vl_erfill(img,treeInfo(i,1)) ;
            %                             auxMask(sel) = 1;
            %                             auxMask = bwmorph(auxMask, 'close');
            %                             sel = find(auxMask == 1);
            %                             asd(sel) = asd(sel)+10*rand(1);
            %                             %figure, imagesc(asd)
            %                         end
            %                     end
            %--------------
            
            if mappedLabels(i)
                if treeInfo(i,1) ~= 0
%                     auxMask = uint8(zeros(size(img,1), size(img,2)));
                    sel = vl_erfill(img,treeInfo(i,1)) ;
%                     auxMask(sel) = 1;
%                     auxMask = bwmorph(auxMask, 'close');
%                     sel = find(auxMask == 1);
                    mask(sel) = 1;
                    %figure, imagesc(mask)
                    correspondance = find(r == treeInfo(i,1));
                    labels(correspondance) = 1;
                    idMask(sel) = 1*correspondance;
                end
            end
        end
    end
    % figure, imagesc(mask)
end


end