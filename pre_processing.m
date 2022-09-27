clear all
clc
addpath utils

rssq = @(x) squeeze(sum(abs(x).^2,3)).^(1/2); % root-sum-of-squares function
pre_proc =1;
spsg_recon = 1; % Split slice-GRAPPA recon based on:
%Cauley, Stephen F., et al. "Interslice leakage artifact reduction technique for simultaneous multislice acquisitions." Magnetic resonance in medicine 72.1 (2014): 93-102.

load('DATA2.mat') % k-space and ACS data for RV and LV time-frames of the myocardial perfusion dataset
%kspace = [208,192,34,2] %[RO,PE,Channel,time-frames]
%acs = [208,64,34,3] %[RO,PE,Channel,slices], calibration scans for the SMS
%there is asymmetric echo and partial Fourier included in the k-space
%acs data is already CAIPIRINHA shifted

slice_R = 3; PE_R= 4; center_region=0;  center_locs = [29,160,61,85];%center_locs are dummy here
%in case you have a scan with fully-sampled center region, please use
%center_region=1 with appropriate center_locs


%%%%% ESPIRiT MAP GENERATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%First start with the ESPIRiT map generation using:
%Uecker, Martin, et al. "ESPIRiT�an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA." Magnetic resonance in medicine 71.3 (2014): 990-1001.
sense_maps_all = zeros(size(kspace,1),size(kspace,2),size(kspace,3),size(acs,4),'single');
acsi = acs(104-12:104+12-1,32-12:32+12-1,:,:); %only using center 24x24 region

for sli = 1:size(acs,4)
    cd ESPIRiT_original
    sense_maps = espirit_generator(zeros(size(kspace,1),size(kspace,2),size(kspace,3)),acsi(:,:,:,sli),24,[6,6],0.02);
    cd ..
    sense_maps_all(:,:,:,sli) = single(sense_maps);
end
sense_orig = sense_maps_all;


sense_maps_all_PF = sense_maps_all;

%%% crop the sense maps to no PF no asymetric echo
k2 = 172; k1 = 164; %% k2 is for PE, k1 is for asymmetric echo,these two can be seen from: figure, imshow(log(kspace(:,:,1,1)),[])
ds = sense_maps_all;
dsn = zeros(k1,k2,size(acs,3),size(acs,4),'single');
for aa = 1:size(acs,4)
    for bb = 1:size(acs,3)
        a = fft2c(squeeze(sense_maps_all(:,:,bb,aa)));
        a = squeeze(a(208-k1+1:208,1:k2));
        dsn(:,:,bb,aa) = ifft2c(a);
    end
end
sense_maps_all = dsn;

%since k-space is in k-space, we can directly cut those regions without
%additional fft operations
kspace = squeeze(kspace(208-k1+1:208,1:k2,:,:)); % same PF and asymmetric echo cutting applied to k-space

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%% Split Slice-GRAPPA RECON%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(spsg_recon==1)
    [spsg_res_all,spsg_kspace_all] = split_slice_GRAPPA_main(kspace,acs,slice_R,PE_R,center_region,center_locs,sense_maps_all);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%% SIGNAL INTENSITY INFORMED MAP GENERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%
filter_size = 24;

for slis = 1:size(spsg_kspace_all,1)
    for aa = 1:size(kspace,4)
        
        cenkter_k2 = size(sense_maps,2)/2;
        cenkter_k1 = size(sense_maps_all,1) - size(sense_maps,1)/2;
        [sli,m,n,no_c,imas] = size(spsg_kspace_all); 
        
        % select the center region of the k-space
        kspacei = squeeze(spsg_kspace_all(slis,cenkter_k1 - filter_size/2 : cenkter_k1 + filter_size/2 -1,cenkter_k2 - filter_size/2 : cenkter_k2 + filter_size/2 -1,:,aa));
        kspacei = kspacei ./ max(abs(kspacei(:)));
        
        % Low Resolution map generation
        bigm = size(sense_maps,1);
        bign = size(sense_maps,2);
        sense_coil_set1 = zeros(bigm,bign,no_c,'single');
        [m,n,no_c] = size(sense_coil_set1);
        
        %generate the blackman filter for all channels
        filt = blackman(filter_size) * blackman(filter_size).';
        filt = repmat(filt, [1 1 no_c]);
        
        %apply he blackman filter to filter size center region
        sense_coil_set1(bigm/2 - filter_size/2 : bigm/2 + filter_size/2 -1,bign/2 - filter_size/2 : bign/2 + filter_size/2 -1,:) = kspacei.* filt;

        %time to generate signal intensity images for each time-frame     
        sense_im_coils = sum((conj(squeeze(sense_orig(:,:,:,slis))).*ifft2c(sense_coil_set1)),3);
        im1 = fft2c(sense_im_coils);
        im2 = ifft2c(im1(208-k1+1:208,1:k2));
        sense_maps_intensity = abs(im2);
        
        
        %if you want to see how does the signal intensity informed maps
        %look like use the following line.
        %We only need the intenstiy values to multiply with ESPIRiT maps
        sense_maps_espirit_unnormalized_all(:,:,:,slis,aa) = sense_maps_all(:,:,:,slis).*sense_maps_intensity;
        sense_maps_intensity_all(:,:,slis,aa) = sense_maps_intensity;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%% PG-DL RECONSTRUCTION FILE GENERATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mask_all = []; %masking
padded_all = [];

%%% acquired - non acquired %%%%%%%%%
for dyn_c = 1:size(kspace,4)
    non_acq_p(:,:,:) = squeeze(kspace(:,:,:,dyn_c))==0;
    acq_p = ones(size(kspace,1),size(kspace,2),size(kspace,3),'single')-non_acq_p; 
    mask_all(:,:,:,dyn_c) = acq_p;
end

kspace_all = kspace;


%%SPSG needs re-adding asymmetric echo and PF regions to match with DL
temp = spsg_res_all;
spsg_res_all = zeros(208,192,size(spsg_res_all,3),size(spsg_res_all,4),'single');
for ims = 1:size(spsg_res_all,4)
    for slice = 1:size(spsg_res_all,3)
        new_im = zeros(208,192,'single');
        new_im(45:end,1:172) = circshift(fft2c(temp(:,:,slice,ims)),[-22 10]);
        spsg_res_all(:,:,slice,ims) = ifft2c(new_im);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save('subject_2','kspace_all','sense_maps_all','sense_maps_intensity_all','spsg_res_all','mask_all','-v7.3')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
