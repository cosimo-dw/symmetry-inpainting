filename = '~/Downloads/inpainting/texture.png';
img = double(imread(filename));
mask = abs(255 - img(:,:,1)) < 10 & abs(img(:,:,2) - 0) < 10  & abs(img(:,:,3) - 0) < 10;
mask = imdilate(mask,strel('square',3));
figure;imshow(mask);
datacost = zeros(800*800,2);
datacost(:,1) = mask(:)*1000;
datacost(:,2) = reshape(mask(:,end:-1:1)*999 + 1,[],1);

revimg = img(:,end:-1:1,:);
idxs = reshape(1:(800*800),800,800);
% idxs(:,1:end-1) -> idxs(:,2:end)
weightH = sum(abs(reshape(img(:,1:end-1,:) - revimg(:,1:end-1,:),[],3)),2) + ...
          sum(abs(reshape(img(:,2:end,:) - revimg(:,2:end,:),[],3)),2);

weightV = sum(abs(reshape(img(1:end-1,:,:) - revimg(1:end-1,:,:),[],3)),2) + ...
          sum(abs(reshape(img(2:end,:,:) - revimg(2:end,:,:),[],3)),2);

neighbor = sparse([reshape(idxs(:,1:end-1),[],1); reshape(idxs(1:end-1,:),[],1)],...
                  [  reshape(idxs(:,2:end),[],1);   reshape(idxs(2:end,:),[],1)],...
                  [                      weightH;                       weightV],...
                  800*800,800*800);

neighbor = round(neighbor / 5);

% GCO_Delete(h);
h = GCO_Create(800*800,2);
GCO_SetDataCost(h,datacost.');
GCO_SetNeighbors(h,neighbor);

GCO_Expansion(h);
GCO_Swap(h);
GCO_Expansion(h);
GCO_Swap(h);

[E D S] = GCO_ComputeEnergy(h);
label = reshape(GCO_GetLabeling(h),800,800);
tmp = reshape(img,[],3);
texture(label(:)==1,:) = tmp(label(:)==1,:);
tmp = reshape(revimg,[],3);
texture(label(:)==2,:) = tmp(label(:)==2,:);
figure;imshow(label,[])
figure;imshow(uint8(reshape(texture,800,800,3)))

[Lh,Lv] = imgrad(img);
[Gh,Gv] = imgrad(revimg);
Lh = reshape(Lh,[],3);
Lv = reshape(Lv,[],3);
Gh = reshape(Gh,[],3);
Gv = reshape(Gv,[],3);

Fh = Lh;
Fv = Lv;
newmask = (label==2); %mask & (label==2);
Fh(newmask(:),:) = Gh(newmask(:),:);
Fv(newmask(:),:) = Gv(newmask(:),:);
dmask = ~imdilate(mask,strel('square',5));
Fh(dmask(:),:) = Lh(dmask(:),:);
Fv(dmask(:),:) = Lv(dmask(:),:);
Fh(mask & (label~=2),:) = 0;
Fv(mask & (label~=2),:) = 0;

Fh = reshape(Fh,800,800,3);
Fv = reshape(Fv,800,800,3);
tic;
Y = PoissonGaussSeidel( reshape(texture,800,800,3), Fh, Fv, repmat(label==2,1,1,3) );
toc
figure;imshow(uint8(Y))

% nmask = abs(255 - Y(:,:,1)) < 10 & abs(Y(:,:,2) - 0) < 10  & abs(Y(:,:,3) - 0) < 10;
% nmask = imdilate(nmask,strel('square',3));
% nmask = repmat(nmask,1,1,3);
% Fh = Lh;
% Fv = Lv;
% Fh = reshape(Fh,800,800,3);
% Fv = reshape(Fv,800,800,3);
% Fh(nmask) = 0;
% Fv(nmask) = 0;
% tic;
% Z = PoissonGaussSeidel( Y, Fh, Fv, nmask );
% toc
% figure;imshow(uint8(Z))