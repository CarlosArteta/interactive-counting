function varargout = InteractiveCountingGUI(varargin)
% INTERACTIVECOUNTINGGUI MATLAB code for InteractiveCountingGUI.fig
%      INTERACTIVECOUNTINGGUI, by itself, creates a new INTERACTIVECOUNTINGGUI or raises the existing
%      singleton*.
%
%      H = INTERACTIVECOUNTINGGUI returns the handle to a new INTERACTIVECOUNTINGGUI or the handle to
%      the existing singleton*.
%
%      INTERACTIVECOUNTINGGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in INTERACTIVECOUNTINGGUI.M with the given input arguments.
%
%      INTERACTIVECOUNTINGGUI('Property','Value',...) creates a new INTERACTIVECOUNTINGGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before InteractiveCountingGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to InteractiveCountingGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help InteractiveCountingGUI

% Last Modified by GUIDE v2.5 12-Jun-2016 16:45:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @InteractiveCountingGUI_OpeningFcn, ...
  'gui_OutputFcn',  @InteractiveCountingGUI_OutputFcn, ...
  'gui_LayoutFcn',  [] , ...
  'gui_Callback',   []);
if nargin && ischar(varargin{1})
  gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
  [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
  gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before InteractiveCountingGUI is made visible.
function varargout = InteractiveCountingGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to InteractiveCountingGUI (see VARARGIN)
global S
set(handles.txt_status, 'String','Initializing');
drawnow;
S = init(handles,S);
% Choose default command line output for InteractiveCountingGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% This sets up the initial plot - only do when we are invisible
% so window can get raised using InteractiveCountingGUI.

if strcmp(get(hObject,'Visible'),'off') %Logo
  p = mfilename('fullpath');
  cd(fileparts(p));
  addpath((genpath(fileparts(p))));
  img = imread('icon.png');
  axes(handles.axes1);
  pp = imshow(img);
  set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
  set(handles.txt_status, 'String','Waiting for User');
end

% UIWAIT makes InteractiveCountingGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = InteractiveCountingGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in btn_process. 
function btn_process_Callback(hObject, eventdata, handles)
% hObject    handle to btn_process (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global S
set(handles.btn_freehand, 'Value', 0);
set(handles.btn_dots, 'Value', 0);
set(handles.btn_stroke, 'Value', 0);
set(handles.btn_diam, 'Value', 0);
axes(handles.axes2);
%---------------------------------------------------------------------Setup
S = setup(handles,S);
gaussRange = -S.gaussWinSize/2:S.gaussWinSize/2;
gaussFilter = 1/(sqrt(2*pi)*S.sigma)*exp(-0.5*gaussRange.^2/(S.sigma^2));
gaussFilter = gaussFilter / sum(gaussFilter);

img = S.img;

%----------------------------------------------------------Compute features
if isempty(S.xd) || S.rescale
  set(handles.txt_status, 'String','Encoding Image');
  drawnow;
  
  img = imresize(img,S.sFactor);
  S.rimg = img;
  
  [xd,S] = computeFeatures(S);
  
  S.xd = xd;
  S.rescale = false;
else
  xd = S.xd;
  img = S.rimg;
end

%---------------------------------------------------------------Annotations
annot = zeros(size(img,1),size(img,2),'single');

if ~isempty(S.dots)
  annotDots = round(S.dots*S.sFactor);
  annot(sub2ind(size(annot), annotDots(:,2), annotDots(:,1))) = 1;
end

if ~isempty(S.strokes)
  for i = 1:numel(S.strokes)
    stroke = S.strokes{i};
    stroke = unique(round(stroke*S.sFactor),'rows');
    annot(sub2ind(size(annot), stroke(:,2), stroke(:,1))) =...
      1/size(stroke,1); %keeps mass = 1
  end
end

ROI = imresize(S.roi,S.sFactor);
density = vl_imsmooth(double(annot),S.sigma);
density = reshape(density,size(density,1)*size(density,2),1);

if ~isempty(S.gt) && S.gtChanged
  S.gt = vl_imsmooth(single(S.gt),S.sigma);
  S.gtChanged = 0;
end

B = regionprops(ROI, 'BoundingBox'); %Bounding Boxes of annotated regions
pad = S.gaussWinSize + 1;
for i=1:numel(B)
  box = B(i).BoundingBox;
  box(1) = round(box(1) - pad);
  if box(1) < 1
    box(1) = 1;
  end
  box(2) = round(box(2) - pad);
  if box(2) < 1
    box(2) = 1;
  end
  box(3) = round(box(3) + 2*pad);
  if box(1) + box(3) > size(ROI,2);
    box(3) = size(ROI,2) - box(1);
  end
  box(4) = round(box(4) +2* pad);
  if box(2) + box(4) > size(ROI,1);
    box(4) = size(ROI,1) - box(2);
  end
  aux = false(size(ROI));
  aux(box(2):box(2)+box(4),box(1):box(1)+box(3)) = true;
  B(i).list = find(aux);
  B(i).BoundingBox = box;
end
clear aux

%Var initializations
sz = [size(xd,1) size(xd,2)];
D = size(xd,3); %dimensionality of the feature vector
As = reshape(xd,sz(1)*sz(2),D);
for i = 1:size(As,2)
  As(:,i) = As(:,i)/max(As(:,i));
end
exmp = 1;

if isempty(S.idx)
  Idx = ones(sz(1)*sz(2),S.nTrees,'uint16');
  nLeaves = ones(1,S.nTrees);
else
  Idx = S.idx;
  nLeaves = S.nLeafs;
end

%------------------------------------------------------------Build codebook
set(handles.txt_status, 'String','Updating Dictionary');
drawnow;
X = cell(1,S.nImages);
Y = cell(1,S.nImages);

%Update dictionary
ROI = reshape(ROI,size(ROI,1)*size(ROI,2),1);

tic
for tr = 1:S.nTrees
  while(1)
    AsHist = hist(double(Idx(ROI,tr)),nLeaves(tr)); %histogram of assignments
    ToSplit = find(AsHist > S.th); %find overpopulated leafs inside ROI
    if ~isempty(ToSplit) %split those leafs
      for j = 1:numel(ToSplit)
        
        if S.splitType == 1
          ToMeasureIdx = Idx(:,tr) == ToSplit(j) & ROI ;
          [~,d] = sort(var( As(ToMeasureIdx,:) ),'descend');
          d = d(1:S.nFeatsSplit);
        else
          d = randi(D,1);
        end
        ToSplitIdx = Idx(:,tr) == ToSplit(j);
          
          if S.firstIt
            %dimTH = graythresh( As(ToSplitIdx ,d(1) ));
            %pointsB = find((As(:,d(1)) < dimTH) & (Idx(:,tr) == ToSplit(j)));
            [~,A] = vl_kmeans( As(ToSplitIdx,d )' ,2);
            pointsB = find( ToSplitIdx );
            pointsB(A==1) = [];
            S.firstIt = false;
          else
            dimTH = median( As(ToSplitIdx & ROI,d(1) ));
            %dimTH = median( As(ToSplitIdx,d(1) ));
            pointsB = find((As(:,d(1)) < dimTH) & (Idx(:,tr) == ToSplit(j)));
          end
                
        %           pointsA = find(As(Idx == j,d) >= mediand);
        
        if isempty(pointsB) %Plan B
          dimTH = mean( As(Idx(:,tr) == ToSplit(j) & ROI,d(1) ));
          pointsB = find((As(:,d(1)) < dimTH) & (Idx(:,tr) == ToSplit(j)));
        end
        
        %points A retain the label
        nLeaves(tr) = nLeaves(tr) + 1;
        Idx(pointsB,tr) = nLeaves(tr);
      end
    else
      break
    end
    %         disp(nLeaves(tr));
    %         disp(num2str(d(1)));
  end
end

disp('Dictionary Updated');
S.idx = Idx;
S.nLeafs = nLeaves;

%densityDists = getLeavesDistibutions(Idx,ROI,density,As);

set(handles.txt_status, 'String','Arranging Samples');
drawnow;
%tic
%Use sparse matrices from now on
  nmax = 0;
  for r = 1:numel(B)
    nmax = nmax+numel(B(r).list);
  end
  code = spalloc(sz(1)*sz(2),sum(nLeaves),nmax*sum(nLeaves)); %full size
  for r = 1:numel(B)%for each annotated region
    box = B(r).BoundingBox;
    coderoi = sparse(zeros((box(4)+1)*(box(3)+1),sum(nLeaves)));
    li = numel(B(r).list);
    s = 1;
    for tr = 1:S.nTrees
      scode = sparse(1:li,double(Idx(B(r).list,tr)),ones(1,li),li,sum(nLeaves(tr)));
      %scode is the size of the BB
      sl = 1;
      e = s + nLeaves(tr) - 1;
      for l = s:e
        sscode = reshape(scode(:,sl), box(4)+1, box(3)+1);
        sscode = sconv2(sscode,gaussFilter,'same');
        coderoi(:,l) = reshape(sscode,(box(4)+1)*(box(3)+1),1);
        sl = sl + 1;
      end
      s = s + nLeaves(tr);
    end
    code(B(r).list,:) = coderoi;
    
  end
  clear scode sscode coderoi
  
toc

%----------------------------------------------------------------Regression

set(handles.txt_status, 'String','Learning Regression');
drawnow;
%toc
%Use only patches inside ROI for regression
code(~ROI,:) = [];
density(~ROI) = [];
X{exmp} = code;
Y{exmp} = density;

X = cell2mat(X');
Y = cell2mat(Y');

%add constant term
X = [X ones(size(X,1),1)];
negs = 1;

while numel(negs) > 0 %enforcing non-negativity
  b = (X'*X + speye(size(X,2)))\(X'*Y);
  negs = find(b<0);
  disp(['Neg. elem.: ' num2str(numel(negs)) '. Fit: ' num2str(sum(X*b)) '/' num2str(sum(Y))]);
  X(:,negs) = 0;
  
end
clear X Y                        

code = zeros(sz(1),sz(2),S.nTrees,'uint16');
offset = 0;
for tr = 1:S.nTrees
  code(:,:,tr) = reshape(Idx(:,tr),sz(1),sz(2))+offset;
  offset = offset + nLeaves(tr);
end
densityEst = sum(b(code),3)+b(end);

densitySmooth = vl_imsmooth(densityEst,S.targScale/3);
densityMap = uint8(255*mat2gray(densitySmooth));
%S.densityRaw = uint8(255*mat2gray(densityEst));
S.densityRaw = densityMap;


%------------------------------------------------------------Visualizations
img = densityMap;
numPixels = size(img,1)*size(img,2);

set(handles.txt_status, 'String','Computing Regions');
drawnow;

if S.regType == 1 %MSER visualization
  
  %Compute MSERs
  [r,ell] = vl_mser(img,'MaxArea',S.maxPixels/numPixels,'MinArea',...
    S.minPixels/numPixels,'MaxVariation',0.2,'MinDiversity',0.2,...
    'Delta',1, 'BrightOnDark',1, 'DarkOnBright',0);
  
  %Encode MSERs
  nFeatures = 2;
  lambda = -1;
  X = zeros(length(r), nFeatures);%feature Vector
  sizeMSER = zeros(length(r), 1);
  additionalU = 1;
  
  for k = 1:length(r)
    sel = vl_erfill(img,r(k)) ;
    sizeMSER(k) = numel(sel);
    if isempty(sel) %|| numel(sel) < minPixels/2;
      X(k,:) = zeros(1,nFeatures);
    else
      X(k,1) = sum(densitySmooth(sel));
      %X(k,2) = var(densitySmooth(sel));
    end
  end
  I = round(X(:,1)); %estimated class of the region
  scores = (1 - (X(:,1) - I)).^2 + lambda; %scoring function
  scores(round(X)==0) = -inf; %discard regions that approximate to 0
  scores(I>S.ERregionTH) = -inf;
  
  MSERtree = buildPylonMSER(img,r,sizeMSER);
  
  [mask,labels,~,idMask] = PylonInference(img, scores, 0, sizeMSER, r, additionalU, MSERtree);
  
  mask = logical(mask);
  X(~labels,1) = 0;
  regions = regionprops(mask, 'Centroid','PixelList','PixelIdxList');
  nRegions = numel(regions);
  class = zeros(length(r),1);
  classMask = zeros(size(img,1), size(img,2),'uint8');
  classText = cell(1,max(I));
  
  S.outDots = [];
  
  for i = 1:nRegions
    class(i) = I(idMask(regions(i).PixelList(1,2),regions(i).PixelList(1,1)));
    if class(i) == 0
      continue;
    end
    classMask(regions(i).PixelIdxList) = class(i);
    classText{class(i)} = [classText{class(i)} ; round(regions(i).Centroid)];
    if class(i) == 1
      S.outDots = [S.outDots ; round(regions(i).Centroid)];
    else
      pixels = single(regions(i).PixelList);
      pixels = [pixels single(img(sub2ind(size(img),pixels(:,2),pixels(:,1))))/-255+1 ];
      %pixels = [pixels img(sub2ind(size(img),pixels(:,2), pixels(:,1)))];
      if size(pixels,1) >= class(i)
        dots = round(vl_kmeans(pixels', class(i)));
        dots = dots(1:2,:)';
      else
        dots = pixels(:,1:2)';
      end
      S.outDots = [S.outDots ; dots];
    end
  end
  
else %compute regions with Spectral Clustering
  tic
  [SPtree,leafMap, nLeaves] = computeSPregions(img);
  toc
  
  %Encode Regions
  nFeatures = 2;
  lambda = -1.1;
  nRegions = max(max(SPtree(:,1:2)))+1;
  X = zeros(nRegions, nFeatures);%feature Vector
  
  for k = 1:nLeaves %these are leafs
    sel = leafMap == k;
    X(k,1) = sum(densitySmooth(sel));
    %X(k,2) = var(densitySmooth(sel));
  end
  
  for k = nLeaves+1:nRegions %these are not leafs
    leaves = getLeaves(SPtree,k);
    %sel = ismember(leafMap,leaves);
    sel = ismembc(leafMap,uint16(leaves)); %much faster
    X(k,1) = sum(densitySmooth(sel));
    %X(k,2) = var(densitySmooth(sel));
  end
  I = round(X(:,1)); %estimated class of the region
  scores = ((1 - (X(:,1) - I)).^2) + lambda;%scoring function
  
  %scores(round(X)==0) = -1; %discard regions that approximate to 0
  
  %hard limit for the coverage of the regions
  scores(I>S.SCregionTH) = -inf;
  scores(I == 0) = 20;
  
  V = zeros(3,0);
  hardU = [zeros(1,nLeaves);1000*ones(1,nLeaves)];
  labels = pylonInference1Class( nLeaves, SPtree, -1*scores', V, -1*hardU);
  labels = pylonConvertLabels(labels, SPtree,  nLeaves);
  
  regions.labels = find(labels);
  nRegions = numel(regions.labels);
  regions.centroids = zeros(nRegions,2);
  
  classMask = zeros(size(img,1),size(img,2),'uint16');
  classText = cell(1,max(I));
  class = zeros(nRegions,1);
  
  for i = 1:nRegions
    leaves = getLeaves(SPtree,regions.labels(i));
    class(i) = I(regions.labels(i));
    sel = ismember(leafMap,leaves);
    sel = bwmorph(sel,'erode');
    classMask(sel) = class(i);
    if class(i) > 0
      cents = regionprops(sel,'Centroid');
      regions.centroids(i,:) = cents.Centroid;
      classText{class(i)} = [classText{class(i)} ; round(regions.centroids(i,:))];
    end
  end
  
  S.outDots = [];
end

%-----------------------------------------------------------Plot and finish
S.outReg = classMask;
S.nClasses = max(class);
S.classText = classText;

%plot
plot_results(handles, densitySmooth, S);
%profile viewer
%save dictionary values for re-use

%S.densityImg = PadIm(img,cropSize);
S.densityImg = img;
%S.density = PadIm(densitySmooth,cropSize);
S.density = densitySmooth;

set(handles.btn_view, 'Enable','on');
set(handles.btn_inspect, 'Enable','on');
set(handles.txt_status, 'String','Waiting for User');

% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
[file, path] = uigetfile( ...
  {'*.png;*.jpg;*.bmp;*.pgm',...
  'All Image Files (*.png, *.jpg, *.bmp, *.pgm)'}, 'Pick an image');
if ~isequal(file, 0)
  S = loadIm(handles, fullfile(path,file), S);
end

% --------------------------------------------------------------------
function PrintMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to PrintMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
printdlg(handles.figure1)

% --------------------------------------------------------------------
function CloseMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to CloseMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
selection = questdlg(['Close ' get(handles.figure1,'Name') '?'],...
  ['Close ' get(handles.figure1,'Name') '...'],...
  'Yes','No','Yes');
if strcmp(selection,'No')
  return;
end

delete(handles.figure1)
clear -global;


% --- Executes on button press in btn_freehand.
function btn_freehand_Callback(hObject, eventdata, handles)
% hObject    handle to btn_freehand (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if get(handles.btn_freehand, 'Value');
  set(handles.btn_dots, 'Value', 0);
  set(handles.btn_diam, 'Value', 0);
  set(handles.btn_inspect, 'Value', 0);
  set(handles.btn_stroke, 'Value', 0);
end

% --- Executes on button press in btn_dots.
function btn_dots_Callback(hObject, eventdata, handles)
% hObject    handle to btn_dots (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(handles.btn_dots, 'Value');
  set(handles.btn_freehand, 'Value', 0);
  set(handles.btn_diam, 'Value', 0);
  set(handles.btn_inspect, 'Value', 0);
  set(handles.btn_stroke, 'Value', 0);
end

% --- Executes on button press in btn_diam.
function btn_diam_Callback(hObject, eventdata, handles)
% hObject    handle to btn_diam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(handles.btn_diam, 'Value');
  set(handles.btn_freehand, 'Value', 0);
  set(handles.btn_dots, 'Value', 0);
  set(handles.btn_inspect, 'Value', 0);
  set(handles.btn_stroke, 'Value', 0);
end

% --- Executes on button press in btn_stroke.
function btn_stroke_Callback(hObject, eventdata, handles)
% hObject    handle to btn_stroke (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(handles.btn_stroke, 'Value');
  set(handles.btn_freehand, 'Value', 0);
  set(handles.btn_dots, 'Value', 0);
  set(handles.btn_diam, 'Value', 0);
  set(handles.btn_inspect, 'Value', 0);
end


% --- Executes on button press in btn_load.
function btn_load_Callback(hObject, eventdata, handles)
% hObject    handle to btn_load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global S
set(handles.txt_status, 'String','Loading Image');
drawnow;
set(handles.txt_neib, 'Enable','off');
% Move to basefolder
% [path,user_cance] = imgetfile();
% if user_cance
%     msgbox(sprintf('Error'), 'Error', 'Error');
%     return
% end
%FOR DEVELOPMENT STAGE
path = 'images/Cells.bmp';
S = loadIm(handles, path, S);


% --- Executes on button press in btn_reset.
function btn_reset_Callback(hObject, eventdata, handles)
% hObject    handle to btn_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S

axes(handles.axes1);

colormapDisp = 1;%get(handles.color_map,'Value');
switch colormapDisp
  case 1 %RGB
    pp = imshow(S.img);
  case 2 %Gray
    if numel(S.channels) == 3
      pp = imshow(rgb2gray(S.img));
    elseif numel(S.channels) == 1
      pp = imshow(S.img(:,:,S.channels));
    end
  case 3 %LAB
    pp = imshow(S.img); %should be LAB
end

set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
S.roi = false(size(S.img,1),size(S.img,2));
S.strokes = {};
S.dots = [];
S.idx = [];
S.nLeafs = [];
S.xd = [];
S.rad = [];
S.diam = [];

% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
axes(handles.axes1);
redraw = 0;

clickType = get(gcbf,'SelectionType');

if strcmp(clickType,'normal')
  colormapDisp = 1;%get(handles.color_map,'Value');
  switch colormapDisp
    case 1 %RGB
      im = S.img;
    case 2 %Gray
      if numel(S.channels) == 3
        im = rgb2gray(S.img);
      elseif numel(S.channels) == 1
        im = S.img(:,:,S.channels);
      end
      im = cat(3,im,im,im);
    case 3 %LAB
      im = S.img; %should be LAB
  end
  
  switch true
    case get(handles.btn_freehand, 'Value')
      ROI = imfreehand(gca);
      ROI = createMask(ROI);
      S.roi = S.roi | ROI;
      roiShow = cat(3,uint8(~S.roi).*0.*ones(size(S.roi),'uint8'),uint8(~S.roi).*100.*ones(size(S.roi),'uint8'),...
        uint8(~S.roi).*50.*ones(size(S.roi),'uint8'));
      pp = imshow(im + roiShow);
      set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
    case get(handles.btn_dots, 'Value')
      DOT = get(gca,'CurrentPoint');
      S.dots = [S.dots ; round(DOT(1,1)) round(DOT(1,2))];
    case get(handles.btn_diam, 'Value')
      if ~isempty(S.diam)
        redraw = 1;
      end
      LINE = imline(gca);
      S.diam = wait(LINE);
      S.rad = pdist(S.diam)/2;
      S.rescale = true;
    case get(handles.btn_inspect, 'Value');
      ROI = imfreehand(gca);
      ROI = createMask(ROI);
      if ~get(handles.btn_view, 'Value')
        ROI = imresize(ROI,S.targScale/S.rad);
      end
      stat = regionprops(ROI,'Centroid');
      ROI = find(ROI);
      areaSum = sum(S.density(ROI));
      if ~get(handles.btn_view, 'Value')
        stat.Centroid = stat.Centroid*S.rad/S.targScale;
      end
      text(stat.Centroid(1), stat.Centroid(2),num2str(areaSum,'%2.2f'),'color','r','FontSize',15,'FontWeight','bold');
    case get(handles.btn_stroke, 'Value')
      strokes = imfreehand(gca,'Closed',false);
      strokes = getPosition(strokes);
      S.strokes = [S.strokes ; round(strokes)];
  end
  
  if ~get(handles.btn_view, 'Value')
    
    if redraw
      roiShow = cat(3,uint8(~S.roi).*0.*ones(size(S.roi),'uint8'),uint8(~S.roi).*100.*ones(size(S.roi),'uint8'),...
        uint8(~S.roi).*50.*ones(size(S.roi),'uint8'));
      pp = imshow(im + roiShow);
      set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
    end
    
    if ~isempty(S.dots)
      hold on, plot(S.dots(:,1),S.dots(:,2),...
        '*r','markersize',8,'linewidth',2,'HitTest','off');
      hold off;
    end
    
    if ~isempty(S.diam)
      hold on, plot(S.diam(:,1),S.diam(:,2),...
        '-y','linewidth',5,'HitTest','off');
      hold off;
    end
    
    if ~isempty(S.strokes)
      hold on;
      for i = 1:numel(S.strokes)
        stroke = S.strokes{i};
        line(stroke(:,1), stroke(:,2),...
          'linewidth',4,'Color',[1 0 0]);
      end
      hold off;
    end
    
  else
    
    if redraw
      roi = imresize(S.roi,S.targScale/S.rad);
      roiShow = cat(3,uint8(roi).*0.*ones(size(roi),'uint8'),uint8(~roi).*50.*ones(size(roi),'uint8'),...
        uint8(~roi).*25.*ones(size(roi),'uint8'));
      pp = imshow(cat(3,S.densityImg,S.densityImg,S.densityImg) + roiShow);
      set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
    end
    
    if ~isempty(S.dots)
      dots = S.dots*S.targScale/S.rad;
      hold on, plot(dots(:,1),dots(:,2),...
        '*r','markersize',8,'linewidth',2,'HitTest','off');
      hold off;
    end
    
    if ~isempty(S.diam)
      diam = S.diam*S.targScale/S.rad;
      hold on, plot(diam(:,1),diam(:,2),...
        '-y','linewidth',5,'HitTest','off');
      hold off;
    end
    
    if ~isempty(S.strokes)
      hold on;
      for i = 1:numel(S.strokes)
        stroke = S.strokes{i};
        line(stroke(:,1), stroke(:,2),...
          'linewidth',4,'Color',[1 0 0]);
      end
      hold off;
    end
    
  end
end
drawnow
refresh(gcf)
drawnow

function txt_nTrees_Callback(hObject, eventdata, handles)
% hObject    handle to txt_nTrees (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_nTrees as text
%        str2double(get(hObject,'String')) returns contents of txt_nTrees as a double


% --- Executes during object creation, after setting all properties.
function txt_nTrees_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_nTrees (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


function txt_splitTH_Callback(hObject, eventdata, handles)
% hObject    handle to txt_splitTH (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_splitTH as text
%        str2double(get(hObject,'String')) returns contents of txt_splitTH as a double


% --- Executes during object creation, after setting all properties.
function txt_splitTH_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_splitTH (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end



function txt_neib_Callback(hObject, eventdata, handles)
% hObject    handle to txt_neib (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_neib as text
%        str2double(get(hObject,'String')) returns contents of txt_neib as a double


% --- Executes during object creation, after setting all properties.
function txt_neib_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_neib (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_view.
function btn_view_Callback(hObject, eventdata, handles)
% hObject    handle to btn_view (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
axes(handles.axes1);

if ~get(handles.btn_view, 'Value') %Show image
  %set(handles.btn_view,'String','Density View');
  set(handles.txt_view, 'String','Annotation view');
  colormapDisp = 1;%get(handles.color_map,'Value');
  switch colormapDisp
    case 1 %RGB
      im = S.img;
    case 2 %Gray
      if numel(S.channels) == 3
        im = rgb2gray(S.img);
      elseif numel(S.channels) == 1
        im = S.img(:,:,S.channels);
      end
      im = cat(3,im,im,im);
    case 3 %LAB
      im = S.img; %should be LAB
  end
  
  roiShow = cat(3,uint8(~S.roi).*0.*ones(size(S.roi),'uint8'),...
    uint8(~S.roi).*100.*ones(size(S.roi),'uint8'),...
    uint8(~S.roi).*50.*ones(size(S.roi),'uint8'));
  pp = imshow(im + roiShow);
  set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
  if ~isempty(S.dots)
    hold on, plot(S.dots(:,1),S.dots(:,2),...
      '*r','markersize',8,'linewidth',2,'HitTest','off');
    hold off;
  end
  
  if ~isempty(S.diam)
    hold on, plot(S.diam(:,1),S.diam(:,2),...
      '-y','linewidth',5,'HitTest','off');
    hold off;
  end
  
  if ~isempty(S.strokes)
    hold on;
    for i = 1:numel(S.strokes)
      stroke = S.strokes{i};
      line(stroke(:,1), stroke(:,2),...
        'linewidth',4,'Color',[1 0 0]);
    end
    hold off;
  end
  
else %show density map
  %set(handles.btn_view,'String','Original View');
  set(handles.txt_view, 'String','Density view');
  roi = imresize(S.roi,S.targScale/S.rad);
  roiShow = cat(3,uint8(roi).*0.*ones(size(roi),'uint8'),...
    uint8(~roi).*50.*ones(size(roi),'uint8'),...
    uint8(~roi).*25.*ones(size(roi),'uint8'));
  pp = imshow(cat(3,S.densityImg,S.densityImg,S.densityImg) + roiShow);
  set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});  
end
drawnow
refresh(gcf)
drawnow

% --- Executes on button press in btn_inspect.
function btn_inspect_Callback(hObject, eventdata, handles)
% hObject    handle to btn_inspect (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(handles.btn_inspect, 'Value');
  set(handles.btn_freehand, 'Value', 0);
  set(handles.btn_diam, 'Value', 0);
  set(handles.btn_dots, 'Value', 0);
end

function S = loadIm(handles, path, S)

S = init(handles,S);
S.path = path;
img = imread(path);
[pathstr,name,ext] = fileparts(path);
if exist([pathstr '/' name '_GT' ext],'file')
  gt = imread([pathstr '/' name '_GT' ext]);
  for i = 1:size(gt,3)
    gtChannel = gt(:,:,i);
    if any(gtChannel(:))
      gt = gtChannel;
      break;
    end
  end
  S.gt = bwmorph(gt>=1,'shrink');
end

%quick test to see if image will produce errors
for i = 1:size(img,3)
  channel = single(img(:,:,i));
  if median(channel(:)) > 0
    S.channels = [S.channels i];
  end
end

if size(img,3) == 1
  img = cat(3,img,img,img);
end
S.img = img;
axes(handles.axes1);
colormapDisp = 1;%get(handles.color_map,'Value');
switch colormapDisp
  case 1 %RGB
    pp = imshow(img);
  case 2 %Gray
    if numel(S.channels) == 3
      pp = imshow(rgb2gray(img));
    elseif numel(S.channels) == 1
      pp = imshow(img(:,:,S.channels));
    end
  case 3 %LAB
    pp = imshow(img); %should be LAB
end
set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
%im = img;
S.roi = false(size(img,1),size(img,2));
S.strokes = {};
S.dots = [];
set(handles.txt_status, 'String','Waiting for User');

% --- Executes on selection change in split_dim.
function split_dim_Callback(hObject, eventdata, handles)
% hObject    handle to split_dim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns split_dim contents as cell array
%        contents{get(hObject,'Value')} returns selected item from split_dim


% --- Executes during object creation, after setting all properties.
function split_dim_CreateFcn(hObject, eventdata, handles)
% hObject    handle to split_dim (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_save_test.
function btn_save_test_Callback(hObject, eventdata, handles)
% hObject    handle to btn_save_test (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
suffix = get(handles.txt_test_name,'String');
roi = S.roi;
diam = S.diam;
dots = S.dots;
strokes = S.strokes;
[pathstr,name] = fileparts(S.path);
save([pathstr '/' name '_' suffix '.mat'],'roi','diam','dots','strokes');


function txt_test_name_Callback(hObject, eventdata, handles)
% hObject    handle to txt_test_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of txt_test_name as text
%        str2double(get(hObject,'String')) returns contents of txt_test_name as a double


% --- Executes during object creation, after setting all properties.
function txt_test_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to txt_test_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_load_test.
function btn_load_test_Callback(hObject, eventdata, handles)
% hObject    handle to btn_load_test (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
[pathstr,name] = fileparts(S.path);
cd(pathstr);
[file, path] = uigetfile( ...
  {[name '*.mat'], 'Mat Files (*.mat)'}, 'Pick a file for selected image');
if ~isequal(file, 0)
  test = load(fullfile(path,file));
  S.roi = test.roi;
  S.diam = test.diam;
  S.dots = test.dots;
  S.strokes = test.strokes;
  S.rad = pdist(S.diam)/2;
  S.rescale = true;
  axes(handles.axes1);
  
  colormapDisp = 1;%get(handles.color_map,'Value');
  switch colormapDisp
    case 1 %RGB
      im = S.img;
    case 2 %Gray
      if numel(S.channels) == 3
        im = rgb2gray(S.img);
      elseif numel(S.channels) == 1
        im = S.img(:,:,S.channels);
      end
      im = cat(3,im,im,im);
    case 3 %LAB
      im = S.img; %should be LAB
  end
  
  roiShow = cat(3,uint8(~S.roi).*0.*ones(size(S.roi),'uint8'),...
    uint8(~S.roi).*100.*ones(size(S.roi),'uint8'),...
    uint8(~S.roi).*50.*ones(size(S.roi),'uint8'));
  pp = imshow(im + roiShow);
  set(pp, 'ButtonDownFcn', {@axes1_ButtonDownFcn, handles});
  if ~isempty(S.dots)
    hold on, plot(S.dots(:,1),S.dots(:,2),...
      '*r','markersize',8,'linewidth',2,'HitTest','off');
    hold off;
  end
  
  if ~isempty(S.diam)
    hold on, plot(S.diam(:,1),S.diam(:,2),...
      '-y','linewidth',5,'HitTest','off');
    hold off;
  end
  
  if ~isempty(S.strokes)
    hold on;
    for i = 1:numel(S.strokes)
      stroke = S.strokes{i};
      line(stroke(:,1), stroke(:,2),...
        'linewidth',4,'Color',[1 0 0]);
    end
    hold off;
  end
end


% --- Executes on button press in btn_result_view.
function btn_result_view_Callback(hObject, eventdata, handles)
% hObject    handle to btn_result_view (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of btn_result_view
global S
plot_results(handles, S);

function plot_results(handles,densitySmooth,S)

axes(handles.axes2);
set(handles.txt_status, 'String','Plotting');
drawnow;

colors = 'gbmyckrw';
orgImg = S.rimg;

visCount = 0;
imshow(orgImg);

colormapDisp = get(handles.color_map,'Value');
switch colormapDisp
  case 1 %RGB
    imshow(orgImg);
  case 2 %Gray
    if numel(S.channels) == 3
      imshow(rgb2gray(orgImg));
    elseif numel(S.channels) == 1
      imshow(orgImg(:,:,S.channels));
    end
  case 3 %LAB
    imshow(orgImg); %should be LAB
end

hold on;
%if ~get(handles.btn_result_view, 'Value')
  for class = 1:S.nClasses
    B = bwboundaries(S.outReg == class);
    if class > 8
      color = 'w';
    else
      color = colors(class);
    end
    for i=1:numel(B)
      line(B{i}(:,2),B{i}(:,1),'Color',color,'LineWidth',3, 'LineStyle','-','marker','.');
    end
    visCount = visCount + class*numel(B); %count from the visualization
  end
  
  if S.plotText
    for class = 2:S.nClasses
      if ~isempty(S.classText{class})
        xy = S.classText{class};
        h = text(xy(:,1), xy(:,2),num2str(class),'color',[0.5 0 0],'FontSize',28,'FontWeight', 'demi');
        set(h,'Clipping','on');
      end
    end
  end
% else
%   if ~isempty(S.outDots)
%     plot(S.outDots(:,1),S.outDots(:,2),'*r','markersize',8,'linewidth',2);
%   end
% end

hold off;

drawnow
refresh(gcf)

%show numbers on txt_results
globalCount = sum(densitySmooth(:));
if ~isempty(S.gt)
  gtCount = sum(S.gt(:));
else
  gtCount = NaN;
end

%results = ['Ground Truth: ' num2str(gtCount,'%02.2f') ' - Global Count: ' num2str(globalCount,'%02.2f')];
results = ['Global Count: ' num2str(globalCount,'%02.2f')];
disp(['Ground Truth: ' num2str(gtCount,'%02.2f') ' - Global Count: ' num2str(globalCount,'%02.2f')...
  ' - Vis Count: ' num2str(visCount) ' - Relative Error: ' num2str(100*(gtCount-globalCount)/gtCount) '%']);
set(handles.txt_results, 'String',results);
set(handles.txt_status, 'String','Waiting for User');

drawnow
refresh(gcf)
drawnow


% --- Executes on selection change in region_type.
function region_type_Callback(hObject, eventdata, handles)
% hObject    handle to region_type (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns region_type contents as cell array
%        contents{get(hObject,'Value')} returns selected item from region_type


% --- Executes during object creation, after setting all properties.
function region_type_CreateFcn(hObject, eventdata, handles)
% hObject    handle to region_type (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_saveIm_in.
function btn_saveIm_in_Callback(hObject, eventdata, handles)
% hObject    handle to btn_saveIm_in (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
axes(handles.axes1);
suffix = get(handles.txt_test_name,'String');
[pathstr,name] = fileparts(S.path);
im = export_fig(gca,'-transparent','-q110','-m1.5','-a2');
imwrite(im,[pathstr '/' name '_' suffix '_in.png'],'png');
imwrite(S.densityRaw,[pathstr '/' name '_' suffix '_dens.png'],'png');

% --- Executes on button press in btn_saveIm_out.
function btn_saveIm_out_Callback(hObject, eventdata, handles)
% hObject    handle to btn_saveIm_out (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global S
%save annotations image
axes(handles.axes1);
[pathstr,name] = fileparts(S.path);
im = export_fig(gca,'-transparent','-q100');
imwrite(im,[pathstr '/' name '_in.png'],'png');

%save density image
imwrite(S.densityRaw,[pathstr '/' name '_dens.png'],'png');

%save output image
axes(handles.axes2);
[pathstr,name] = fileparts(S.path);
im = export_fig(gca,'-transparent','-q100');
imwrite(im,[pathstr '/' name '_out.png'],'png');

%save count
globalCount = sum(S.density(:));
fileID = fopen([pathstr '/' name '_count.txt'],'w');
fprintf(fileID,'%f\n',globalCount);
fclose(fileID);



% --- Executes on selection change in color_map.
function color_map_Callback(hObject, eventdata, handles)
% hObject    handle to color_map (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns color_map contents as cell array
%        contents{get(hObject,'Value')} returns selected item from color_map


% --- Executes during object creation, after setting all properties.
function color_map_CreateFcn(hObject, eventdata, handles)
% hObject    handle to color_map (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end




%-----------------------------------------------------------Other callbacks

function S = init(handles,S)
%Initialize S structure.

S.img = [];
S.path = [];
S.rimg = [];
S.gt = [];
S.gtChanged = 1;
S.rescale = false;
S.targScale = 8;
S.roi = [];
S.dots = [];
S.idx = [];
S.strokes = {};
S.nLeafs = [];
S.xd = [];
S.rad = [];
S.diam = [];
S.density = [];
S.densityImg = [];
S.densityRaw = [];
S.channels = [];
S.outReg = [];
S.outDots = [];
S.nClasses = [];
S.classText = [];
S.plotText = 1;
S.firstIt = true;
S.maxResSize = 512;
S.globalCount = [];

%features
S.textPatches = 0;
S.gabor = 1;
S.dog = 1;
S.colorChannels = 1;

%visualizations
S.minPixels = (S.targScale/16)^2;
S.maxPixels = 2000;%numPixels;
S.ERregionTH = 7;
S.SCregionTH = 9;

set(handles.btn_dots, 'Value', 0);
set(handles.btn_freehand, 'Value', 0);

function S = setup(handles,S)
%Setup algorithm parameters.

S.nImages = 1;
S.nFeatsSplit = 3;
S.nFeat = 0;

if S.colorChannels
  S.nFeat_cc =  numel(S.channels);
else
  S.nFeat_cc = 0;
end

if S.textPatches
  S.neib = str2double(get(handles.txt_neib,'String'));
  S.cropSize = ceil(S.neib/2);
  S.orientBins = 16;
  S.nFeat_tp = S.neib^2 + 1;
else
  S.nFeat_tp = 0;
end

if S.gabor
  S.nOrient = 4;
  S.nScales = 5;
  S.minWaveLength = S.targScale/3;
  S.mult = 1.7;
  S.sigmaOnf = 0.65;
  S.dThetaOnSigma = 1.3;
  S.Lnorm = 2;
  S.feedback = 1;
  S.nFeat_gb = S.nScales;
else
  S.nFeat_gb = 0;
end

if S.dog
  S.sigmaRange = [S.targScale/6 S.targScale/2];
  S.nPerOctave = 3;
  S.sigmas = power(2,log2(S.sigmaRange(1))-(0.5/S.nPerOctave):...
    (1.0/S.nPerOctave):log2(S.sigmaRange(2))+(0.5/S.nPerOctave));
  S.nFeat_dog = numel(S.sigmas) - S.nPerOctave;
else
  S.nFeat_dog = 0;
end

S.th = 200;%str2double(get(handles.txt_splitTH,'String')); %number of words that a leaf must hold to be split
S.splitType = 1;%get(handles.split_dim,'Value');
if S.splitType == 1
  %set(handles.txt_nTrees,'String',1);
  S.nTrees = 1;
  S.nFeatSplit = 1; %TODO: Implement in interface
else
  %S.nTrees = str2double(get(handles.txt_nTrees,'String'));
end

if ~isempty(S.strokes)
  S.sigma = S.targScale/10;
else
  S.sigma = S.targScale/1.6;
end
S.regType = 1;%get(handles.region_type,'Value'); %1 Extremal Regions, 2 Spectral Clustering
if isempty(S.rad) && ~isempty(S.strokes)
  S.rad = zeros(1,numel(S.strokes));
  for i = 1:numel(S.strokes)
    stroke = S.strokes{i};
    rad(i) = size(stroke,1);
  end
  S.rad = median(rad)/3;
elseif isempty(S.rad)
  error('Diameter must be provided for dot annotations');
end
S.sFactor = S.targScale/S.rad;

%Gaussian kernel for feature smoothing
S.gaussWinSize = 12;

function [xd,S] = computeFeatures(S)
% Computes the features according to the data and parameters on S.

img = S.rimg;
xd = [];
if numel(S.channels) == 3
  %convert to LAB
  imlab = vl_xyz2lab(vl_rgb2xyz(img)) ;
  l = single(imlab(:,:,1));
  a = single(imlab(:,:,2));
  b = single(imlab(:,:,3));
else
  l = single(img(:,:,S.channels));
end

l = anisodiff(l,5,20,.25,1);

if S.colorChannels
  xcc = zeros(size(img,1),size(img,2),S.nFeat_cc,'single');
  xcc(:,:,1) = single(l/max(l(:)));
  if numel(S.channels) == 3
    xcc(:,:,2) = single(a/max(a(:)));
    xcc(:,:,3) = single(b/max(b(:)));
  end
  xd = cat(3,xd,xcc);
  clear xcc
end

if S.textPatches
  xtp = zeros(size(img,1),size(img,2),S.nFeat_tp,'single');
  xtp(:,:,1:S.nFeat_tp)=...
    EncodePatchBased(l,S.neib,S.cropSize,1,single(l/max(l(:))),0,S.orientBins);
  xd = cat(3,xd,xtp);
  clear xtp
end

if S.gabor
  xgb = zeros(size(img,1),size(img,2),S.nFeat_gb,'single');
  xgb_or = zeros(size(img,1),size(img,2),S.nOrient,'single');
  EO = gaborconvolve(l, S.nScales, S.nOrient,  S.minWaveLength, S.mult, ...
    S.sigmaOnf, S.dThetaOnSigma, S.Lnorm, S.feedback);
  for i = 1:S.nScales
    for j=1:S.nOrient
      %xgb(:,:,j+nOrient*(i-1)) = single(abs(EO{i,j}));
      xgb_or(:,:,j) = single(abs(EO{i,j}));
    end
    xgb(:,:,i) = sum(xgb_or,3);
    xgb(:,:,i) = xgb(:,:,i)/max(max(xgb(:,:,i)));
  end
  xd = cat(3,xd,xgb);
  clear E0 xgb
end

if S.dog
  l_n = single(l-median(l(:)));
  imG = zeros([size(l) numel(S.sigmas)],'single');
  
  for i = 1:size(imG,3)
    imG(:,:,i) = vl_imsmooth(l_n,S.sigmas(i),'Padding','continuity');
  end
  xdog = zeros([size(img,1),size(img,2),numel(S.sigmas)-S.nPerOctave],'single');
  %sigma = zeros(1,numel(sigmas)-nPerOctave);
  for i = 1:size(imG,3)-S.nPerOctave
    %   sigma(i) = sqrt(sigmas(i)*sigmas(i+nPerOctave));
    xdog(:,:,i) = imG(:,:,i)-imG(:,:,i+S.nPerOctave);
  end
  xd = cat(3,xd,xdog);
  clear img_n imG xdog
end
