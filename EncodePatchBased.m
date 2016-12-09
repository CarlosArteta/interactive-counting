function imf = EncodePatchBased(im,neighSize, cropSize, normPatch, normalizedIm,addEnt,nAngleBins)
%Encoding of image using normalized raw intensities in neighbourhood
%image = input image
%neighSize = size of the neighbourhood centered at each pixel
%cropSize = number of rows and columns to crop on each side of the output
%normPatch = choose to contrast-normalized each patch 
%normalizedIm = image to extract absolute intensity feature

if nargin < 4
    normPatch = 1;
end
if nargin < 5
    normalizedIm = im;
    display('Intensity feature taken from non-normalized image');
end
if nargin < 6
    addEnt = 0;
end
if nargin < 7
    nAngleBins = 32;
end
% im = imadjust(im, [0.12 0.8], [] );
if addEnt
    e1 = entropyfilt(im, true(5));
    e2 = entropyfilt(im, true(9));
    e3 = entropyfilt(im, true(15));
    e4 = entropyfilt(im, true(21));
    e5 = entropyfilt(im, true(25));
end

im = single(im);
normalizedIm = single(normalizedIm);
%im = (im-mean(im(:)))/norm(im,1);
centre = [size(im,2)/2+.5 , size(im,1)/2+.5]; %Reference centre of corrdinates for rotations
%im = im/norm(im,1);

halfPatch = (neighSize-1)/2; 
delta = (halfPatch/2+1)/2;

%Compute image gradient to get orientation of patches for steering
imG = vl_imsmooth(im,delta,'Padding','continuity');
[gx gy] = gradient(imG);
intImGx = cumsum(cumsum(gx),2); %integral images
intImGy = cumsum(cumsum(gy),2);

%nAngleBins = 32; %patch rotations will be quantized into nAngleBins values
anglesBins = -180+360/nAngleBins:360/nAngleBins:180;
imRotations = cell(nAngleBins,1);
sizeIm = zeros(nAngleBins,2,'single');

%handle for contrast normalization function
%contrastNorm = @(vector) vector.*( log(1+norm(vector,2)/0.03 ) )/norm(vector,2);

%filter to compute areas over the integral images using conv2
window = zeros(neighSize+2, neighSize+2); 
window(2,2) = 1; window(2,end) = -1;  window(end,2) = -1; window(end,end) = 1;

%compute the gradient in x and y for all pixels based on ist neighbours 
gxPatch = conv2(intImGx, window, 'same');
gyPatch = conv2(intImGy, window, 'same');
z = gxPatch + gyPatch*1i; 
angZ = angle(z)*180/pi; %compute the orientation angle in degrees
angZ = reshape(angZ, size(z,1)*size(z,2),1);
[NU,indexAngle] = min(abs(bsxfun(@minus,angZ',anglesBins'))); %quantize with angleBins
angZ = anglesBins(indexAngle);

%calculate the mapping between each pixel in the original image and the
%correspondent pixel in the rotate image 
%first, create the structure with the original rows and cols
rotPixel = zeros(size(im,1), size(im,2),2,'single'); 
[row col] = ind2sub(size(rotPixel),find(rotPixel(:,:,1) == 0)); 
rotPixel(:,:,1) = reshape(col, size(rotPixel,1), size(rotPixel,2));
rotPixel(:,:,2) = reshape(row, size(rotPixel,1), size(rotPixel,2));
clear row col

%now put them with respect to the new coordinate system centred in the
%centre of the image
rotPixel = reshape(rotPixel,size(im,1)*size(im,2),2);
rotPixel(:,1) = rotPixel(:,1) - centre(1);
rotPixel(:,2) = centre(2) - rotPixel(:,2); 

%convert to polar to make the rotations easier
[theta,rho] = cart2pol(rotPixel(:,1),rotPixel(:,2));

%rotate each pixel by the angle dictated by its neighbourhood
theta = theta + (angZ')*pi/180; 

%and back to cartesian
[rotPixel(:,1), rotPixel(:,2)] = pol2cart(theta,rho);

%compute all image rotations considered in anglesBins
for i = 1:nAngleBins
    imRotations{i} = imrotate(im,anglesBins(i));
    %all rotations will have different sizes. Let's save them in sizeIm
    sizeIm(i,1) = size(imRotations{i},1);
    sizeIm(i,2) = size(imRotations{i},2);
end

%adjust the rotated pixels to take into account the change in image size
rotPixel(:,1)= round(rotPixel(:,1) + centre(1) + (sizeIm(indexAngle,2)-size(im,2))/2);
rotPixel(:,2)= round(centre(2) - rotPixel(:,2) + (sizeIm(indexAngle,1)-size(im,1))/2);

imf = zeros(size(im,1)*size(im,2), neighSize^2,'single'); %Output structure

for i = 1:nAngleBins %les get the correspondent patches from each of the rotated images
    rotIm = imRotations{i}; %get the rotated image
    indInOrig = find(indexAngle == i); %get the pixels that have their correspondance in this image
    if isempty(indInOrig)
        continue;
    end
    col = rotPixel(indInOrig,1); %get their row and col number in the rotated image 
    row = rotPixel(indInOrig,2);
    
    %create the structures to save all pixels from all patches found in the
    %image.
    patchPixelsRow = zeros(length(row), neighSize^2);
    patchPixelsCol = zeros(length(col), neighSize^2);

    %get all those pixels per patch
    j = 1;
    for k = 1:neighSize
        for l = 1:neighSize
            patchPixelsRow(:,j) = row' - halfPatch  - 1 + k;
            patchPixelsCol(:,j) = col' - halfPatch - 1 + l;
            j = j + 1;
        end
    end
    
    %Validate limits for each patch
    patchPixelsRow(patchPixelsRow < 1) = 1;
    patchPixelsRow(patchPixelsRow > size(rotIm,1)) = size(rotIm,1);
    patchPixelsCol(patchPixelsCol < 1) = 1;
    patchPixelsCol(patchPixelsCol > size(rotIm,2)) = size(rotIm,2);
    
    %convert the subscripts of each pixel into linear indexes    
    patchPixelIdx = sub2ind(size(rotIm), patchPixelsRow, patchPixelsCol);
    
    %get all the patches and store them on each row
    patches = rotIm(patchPixelIdx);
    
    %contrast-normalize each patch individually 
%     if normPatch
%         patches = num2cell(patches,2);
%         patches = cellfun(contrastNorm,patches,'UniformOutput', false);
%         patches = cell2mat(patches);
%     end

    %save in the output structure
    imf(indInOrig,:) = patches;   
end

%contrast-normalize each patch individually 
if normPatch 
    normCts = (log(1+sqrt(sum(abs(imf).^2,2))/0.03 ))./sqrt(sum(abs(imf).^2,2));
    imf = bsxfun(@times,imf,normCts);
end

imf = reshape(imf, size(im,1), size(im,2), neighSize^2);
imG = vl_imsmooth(normalizedIm,delta,'Padding','continuity');

if addEnt
    imf = cat(3,imf,imG,e1,e2,e3,e4,e5);
else
    imf = cat(3,imf,imG);
end


% imf(1:cropSize,:,:)=[];
% imf(end-cropSize+1:end,:,:)=[];
% imf(:,1:cropSize,:)=[];
% imf(:,end-cropSize+1:end,:)=[];

end

