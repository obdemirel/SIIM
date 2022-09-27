function [mod_kspace] = sg_rec(slice_R,PE_R,ims,data_ak_ind,kernel_r,kernel_c,new_kspace,center_locs,data_ak_small,kernel_rs,kernel_cs,center_region,gui_on,par_on)



mod_kspace = zeros(size(new_kspace,1),size(new_kspace,2),size(new_kspace,3),slice_R,ims,'single');
mod_kspace_mid = zeros(size(center_locs(1):center_locs(2),2),size(center_locs(3):center_locs(4),2),size(new_kspace,3),slice_R,ims,'single');
only_acq = zeros(size(new_kspace,1),size(new_kspace,2),size(new_kspace,3),slice_R,size(new_kspace,4),'single');

if(par_on==1)
    
    for slice = 1:slice_R;
        if(gui_on==1)
            progressbar(slice/(slice_R+1))
        end
        ksb = size(new_kspace,1);
        n = size(new_kspace,2);
        coil_n = size(new_kspace,3);
        parfor ss = 1:ims
            ak_ind = squeeze(data_ak_ind(slice,:,:));
            mb_kspce = squeeze(new_kspace(:,:,:,ss));
            end_kspace = conv_op_ind_sg_55(PE_R,ak_ind,mb_kspce,kernel_r,kernel_c,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
            mb_kspace_new = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
            
            mod_kspace(:,:,:,slice,ss) = mb_kspace_new;
            
            if(center_region==1)
                ak_ind = squeeze(data_ak_small(slice,:,:));
                mb_kspce = squeeze(new_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,ss));
                %mb_kspce = squeeze(new_kspace(:,center_locs(3):center_locs(4),:,ss));
                end_kspace = conv_op_ind_sg_55(1,ak_ind,mb_kspce,kernel_rs,kernel_cs,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                
                mod_kspace_mid(:,:,:,slice,ss) = mb_kspace_new;
                %mod_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
            end
            
            %if(center_region==1)
            %    only_acq(:,1:PE_R:end,:) = mod_kspace(:,1:PE_R:end,:,slice,ss);
            %    mod_kspace(:,:,:,slice,ss) = 0;
            %    mod_kspace(:,1:PE_R:end,:,slice,ss) = only_acq(:,1:PE_R:end,:);
            %    mod_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
            %    %mod_kspace(:,center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
            %end
        end
            if(center_region==1)
                only_acq(:,1:PE_R:end,:,slice,:) = mod_kspace(:,1:PE_R:end,:,slice,:);
                mod_kspace(:,:,:,slice,:) = 0;
                mod_kspace(:,1:PE_R:end,:,slice,:) = only_acq(:,1:PE_R:end,:,slice,:);
                mod_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,slice,:)...
                    = squeeze(mod_kspace_mid(:,:,:,slice,:));
                %mod_kspace(:,center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
            end
        disp(['Slice: ',num2str(slice),' is ready'])
        
    end
    
else
    for slice = 1:slice_R;
        if(gui_on==1)
            progressbar(slice/(slice_R+1))
        end
        ksb = size(new_kspace,1);
        n = size(new_kspace,2);
        coil_n = size(new_kspace,3);
        for ss = 1:ims
            ak_ind = squeeze(data_ak_ind(slice,:,:));
            mb_kspce = squeeze(new_kspace(:,:,:,ss));
            end_kspace = conv_op_ind_sg_55(PE_R,ak_ind,mb_kspce,kernel_r,kernel_c,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
            mb_kspace_new = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
            
            mod_kspace(:,:,:,slice,ss) = mb_kspace_new;
            
            if(center_region==1)
                ak_ind = squeeze(data_ak_small(slice,:,:));
                mb_kspce = squeeze(new_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,ss));
                %mb_kspce = squeeze(new_kspace(:,center_locs(3):center_locs(4),:,ss));
                end_kspace = conv_op_ind_sg_55(1,ak_ind,mb_kspce,kernel_rs,kernel_cs,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                
                mod_kspace_mid = mb_kspace_new;
                %mod_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
            end
            if(center_region==1)
                only_acq(:,1:PE_R:end,:,ss) = mod_kspace(:,1:PE_R:end,:,slice,ss);
                mod_kspace(:,:,:,slice,ss) = 0;
                mod_kspace(:,1:PE_R:end,:,slice,ss) = only_acq(:,1:PE_R:end,:,ss);
                mod_kspace(center_locs(1):center_locs(2),center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
                %mod_kspace(:,center_locs(3):center_locs(4),:,slice,ss) = mod_kspace_mid;
            end
        end
        disp(['Slice: ',num2str(slice),' is ready'])
        
    end
end
if(gui_on==1)
    progressbar(1)
end

end

