function [spsg_res_all,finalrecon] = split_slice_GRAPPA_main(kspace,acs,slice_R,PE_R,center_region,center_locs,sense_maps_all)

kspace_allm = kspace;
acs_spsg = acs;
cd spsg
[m,n,no_c,ims]= size(kspace_allm);
kspacei = kspace_allm;
%normalizing the k-space - same as in PG-DL deep learning
for aa = 1:size(kspace_allm,4)
    max_val(aa) = max(max(max(abs(kspace_allm(:,:,:,aa)))));
    kspacei(:,:,:,aa) = kspace_allm(:,:,:,aa)./max(max(max(abs(kspace_allm(:,:,:,aa)))));
end

disp('Split slice-GRAPPA Kernels are processing...') %SMS entanglement kernel calibration
tic
[data_ak_ind,kernel_r,kernel_c] = sg_kernel_main_sp(slice_R,PE_R,kspacei,acs_spsg,[5,5],0,1);
toc

if(center_region ==1) %Skipping this part since no fully-sampled ACS region is available
    tic
    [data_ak_small,kernel_rs,kernel_cs] = sg_kernel_mini_sp(slice_R,PE_R,kspacei,acs_spsg,[5,5],0,1);
    toc
end
disp('Split slice-GRAPPA Kernels are Ready!')

disp('Split slice-GRAPPA SMS Entanglement Part is processing...') %SMS entanglement part
tic
if(center_region ==0)
    data_ak_small = 0; kernel_rs = 0; kernel_cs = 0;
end
sgrecon = sg_rec(slice_R,PE_R,ims,data_ak_ind,kernel_r,kernel_c,kspacei,center_locs,data_ak_small,kernel_rs,kernel_cs,center_region,0,1);
toc
disp('Split slice-GRAPPA SMS Entanglement Part is Ready!')

disp('Split slice-GRAPPA in-plane reconstruction is processing...') %In-plane kernel calibration and final reconstruction
tic
if(PE_R==1)
    finalrecon = permute(sgrecon,[4 1 2 3 5]);
else
    finalrecon = sg_fin(slice_R,PE_R,ims,sgrecon,acs_spsg,kspacei,[5,4],0,1);
    toc
    disp('Split slice-GRAPPA in-plane reconstruction is Ready!')
end


for aa = 1:size(kspace_allm,4)
    finalrecon(:,:,:,:,aa) = finalrecon(:,:,:,:,aa).*max_val(aa);
end

%SENSE-1 reconstruction for GRAPPA reconstruced images using ESPIRiT
[sense1_images] = sense1_maker(finalrecon,sense_maps_all,ims);
spsg_res_all = sense1_images;
cd ..

end

