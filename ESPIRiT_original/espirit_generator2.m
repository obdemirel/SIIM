function [sense_maps] = espirit_generator(kspace,acs,ncalib,ksize,thr)
%% ESPIRiT Maps Demo
% This is a demo on how to generate ESPIRiT maps. It is based on the paper
% Uecker et. al, MRM 2013 DOI 10.1002/mrm.24751. ESPIRiT is a method that
% finds the subspace of multi-coil data from a calibration region in
% k-space using a series of eigen-value decompositions in k-space and image
% space. 

%%
% Set parameters

nocoils = size(kspace,3);
for slice_abc = 1:size(acs,4)
tic
disp(['ESPIRiT maps of Slice #: ',num2str(slice_abc),' is processing'])
run setPath

kspace = ones(size(kspace,1),size(kspace,2),nocoils);
DATA = kspace;

[sx,sy,Nc] = size(DATA);
%ncalib = 24; % use 24 calibration lines to compute compression
%ksize = [6,6]; % kernel size


% Threshold for picking singular vercors of the calibration matrix
% (relative to largest singlular value.

eigThresh_1 = thr;%0.02; %0.004 ORIGINAL 0.02

% threshold of eigen vector decomposition in image space. 
eigThresh_2 = 0.98;%0.985;

% crop a calibration area
%calib = crop(DATA,[ncalib,ncalib,Nc]);

calib = squeeze(acs(:,:,:,slice_abc));
%%
% Display coil images: 
im = ifft2c(DATA);

% figure, imshow3(abs(im),[],[1,30]); 
% title('magnitude of physical coil images');
% colormap((gray(256))); colorbar;
% 
% figure, imshow3(angle(im),[],[1,30]); 
% title('phase of physical coil images');
% colormap('default'); colorbar;

%% Compute ESPIRiT EigenVectors
% Here we perform calibration in k-space followed by an eigen-decomposition
% in image space to produce the EigenMaps. 

% compute Calibration matrix, perform 1st SVD and convert singular vectors
% into k-space kernels

[k,S] = dat2Kernel(calib,ksize);
idx = max(find(S >= S(1)*eigThresh_1));

%% 
% This shows that the calibration matrix has a null space as shown in the
% paper. 

%kdisp = reshape(k,[ksize(1)*ksize(2)*Nc,ksize(1)*ksize(2)*Nc]);
%figure, subplot(211), plot([1:ksize(1)*ksize(2)*Nc],S,'LineWidth',2);
%hold on, 
%plot([1:ksize(1)*ksize(2)*Nc],S(1)*eigThresh_1,'r-','LineWidth',2);
%plot([idx,idx],[0,S(1)],'g--','LineWidth',2)
%legend('signular vector value','threshold')
%title('Singular Vectors')
%subplot(212), imagesc(abs(kdisp)), colormap(gray(256));
%xlabel('Singular value #');
%title('Singular vectors')


%%
% crop kernels and compute eigen-value decomposition in image space to get
% maps
[M,W] = kernelEig(k(:,:,:,1:idx),[sx,sy]);

%%
% show eigen-values and eigen-vectors. The last set of eigen-vectors
% corresponding to eigen-values 1 look like sensitivity maps

% figure, imshow3(abs(W),[],[1,30]); 
% title('Eigen Values in Image space');
% colormap((gray(256))); colorbar;
% 
% figure, imshow3(abs(M),[],[30,30]); 
% title('Magnitude of Eigen Vectors');
% colormap(gray(256)); colorbar;
% 
% figure, imshow3(angle(M),[],[30,30]); 
% title('Magnitude of Eigen Vectors');
% colormap(jet(256)); colorbar;


%%
% project onto the eigenvectors. This shows that all the signal energy
% lives in the subspace spanned by the eigenvectors with eigenvalue 1.
% These look like sensitivity maps. 


% alternate way to compute projection is:
% ESP = ESPIRiT(M);
% P = ESP'*im;

P = sum(repmat(im,[1,1,1,Nc]).*conj(M),3);
%figure, imshow3(abs(P),[],[1,30]); 
% title('Magnitude of Eigen Vectors');
% colormap(sqrt(gray(256))); colorbar;
% 
% figure, imshow3(angle(P),[],[1,30]); 
% title('Magnitude of Eigen Vectors');
% colormap((jet(256))); colorbar;



%%
% crop sensitivity maps 
%maps = M(:,:,:,end);
maps = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]);

weights = W(:,:,end-1:end) ;
weights = (weights - eigThresh_2)./(1-eigThresh_2).* (W(:,:,end-1:end) > eigThresh_2);
weights = -cos(pi*weights)/2 + 1/2;


%figure, imshow3(abs(maps),[],[1,nocoils]); 
%title('Absolute sensitivity maps');
%colormap((gray(256))); colorbar;

%figure, imshow3(angle (maps),[],[1,nocoils]); 
%title('Phase of sensitivity maps');
%colormap((jet(256))); colorbar;

sense_maps(:,:,:,slice_abc) = maps;
toc
end
% nIterCG = 20;
% 
% DATAc = DATA;
% SNS = ESPIRiT(maps(:,:,:));
% % SNS = ESPIRiT(maps(:,:,:),weights);
% tic;[reskSENSE, resSENSE1] = cgESPIRiT(DATAc, SNS, nIterCG,0 ,DATAc*1);toc
% %figure, imshow(abs(resSENSE1),[])
% figure, imshow(abs(circshift((resSENSE1),[84 115])),[])
% %figure, imshow(abs(circshift(sos(resSENSE1),[80 76])),[])
%save('sense_maps','sense_maps','-v7.3')

end

