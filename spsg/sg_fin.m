function [hf_kspace] = sg_fin(slice_R,PE_R,ims,sg_kspace,acs,new_kspace,kernels,gui_on,par_on)


hf_kspace = zeros(slice_R,size(new_kspace,1),size(new_kspace,2),size(new_kspace,3),ims,'single');

for klm=1:slice_R
    
    kspace = sg_kspace;
    
    conc_acs_fregion = squeeze(acs(:,:,:,klm));
    
    reg = 0;
    
    %%% KERNEL %%%%%%%%%
    % x0x0x0x        x0x0x0x
    % x0x0x0x        x0x0x0x
    % x0x0x0x    ->  x0x*x0x
    % x0x0x0x        x0x0x0x
    % x0x0x0x        x0x0x0x
    %%%%%%%%%%%%%%%%%%%%
    
    kernel_row = kernels(1);  %3
    kernel_col = kernels(2);  %3
    
    kernel_dim = kernel_row*kernel_col;
    
    acq_cols = 1:PE_R:size(conc_acs_fregion,2);
    col_adder = acq_cols(kernel_col)-acq_cols(1);
    
    acq_cols_back = size(conc_acs_fregion,2):-PE_R:1;
    
    
    MA = zeros((size(conc_acs_fregion,1)-(kernel_row-1))*(size(conc_acs_fregion,2)-col_adder),kernel_dim*size(kspace,3),'single');
    
    %% MA matrix filling
    for coil_selec = 1:size(conc_acs_fregion,3)
        selected_acs = conc_acs_fregion(:,:,coil_selec);
        row_count = 1;
        for col = 1:size(selected_acs,2)-col_adder
            for row = 1:size(selected_acs,1)-(kernel_row-1)
                neighbors = selected_acs(row:row+(kernel_row-1),col:PE_R:col+col_adder);
                neighbors = neighbors(:).';
                MA(row_count,(coil_selec-1)*(kernel_dim) +1:coil_selec*(kernel_dim)) = neighbors;
                row_count = row_count+1;
            end
        end
    end
    
    
    row_start = ceil(kernel_row/2);
    row_end = size(conc_acs_fregion,1)-floor(kernel_row/2);
    
    if(PE_R==2)
        Mk = zeros([size(MA) PE_R-1],'single');
        %% Mk vectors filling
        for outer = 1:PE_R-1
            for coil_selec = 1:size(conc_acs_fregion,3)
                if(outer==1)
                    selected_acs = conc_acs_fregion(row_start:row_end,mean(acq_cols(1:kernel_col)):mean(acq_cols_back(1:kernel_col)),coil_selec);
                end
                Mk(:,coil_selec,outer) = selected_acs(:);
            end
        end
    elseif(PE_R==3)
        Mk = zeros([size(MA) PE_R-1],'single');
        %% Mk vectors filling
        for outer = 1:PE_R-1
            for coil_selec = 1:size(conc_acs_fregion,3)
                if(outer==1)
                    selected_acs = conc_acs_fregion(row_start:row_end,ceil(mean(acq_cols(1:kernel_col)))-1:ceil(mean(acq_cols_back(1:kernel_col)))-1,coil_selec);
                elseif(outer==2)
                    selected_acs = conc_acs_fregion(row_start:row_end,ceil(mean(acq_cols(1:kernel_col))):ceil(mean(acq_cols_back(1:kernel_col))),coil_selec);
                elseif(outer==3)
                    selected_acs = conc_acs_fregion(row_start:row_end,mean(acq_cols(1:kernel_col))+1:mean(acq_cols_back(1:kernel_col))+1,coil_selec);
                end
                Mk(:,coil_selec,outer) = selected_acs(:);
            end
        end
    elseif(PE_R==4)
        Mk = zeros([size(MA) PE_R-1],'single');
        %% Mk vectors filling
        for outer = 1:PE_R-1
            for coil_selec = 1:size(conc_acs_fregion,3)
                if(outer==1)
                    selected_acs = conc_acs_fregion(row_start:row_end,mean(acq_cols(1:kernel_col))-1:mean(acq_cols_back(1:kernel_col))-1,coil_selec);
                elseif(outer==2)
                    selected_acs = conc_acs_fregion(row_start:row_end,mean(acq_cols(1:kernel_col)):mean(acq_cols_back(1:kernel_col)),coil_selec);
                elseif(outer==3)
                    selected_acs = conc_acs_fregion(row_start:row_end,mean(acq_cols(1:kernel_col))+1:mean(acq_cols_back(1:kernel_col))+1,coil_selec);
                end
                Mk(:,coil_selec,outer) = selected_acs(:);
            end
        end
    end
    
    %disp('Mk is ready!')
    
    
    %reg = norm(MA,'fro')/sqrt(size(MA,1))*reg;
    ak = zeros(kernel_dim*size(kspace,3),size(kspace,3),PE_R-1,'single');
    
    pre_cal_A = pinv(MA'*MA);
    if(par_on==1)
        parfor outer = 1:PE_R-1
            pre_cal_B(:,:,outer) = MA'*squeeze(Mk(:,:,outer));
        end
    else
        for outer = 1:PE_R-1
            pre_cal_B(:,:,outer) = MA'*squeeze(Mk(:,:,outer));
        end
    end
    
    for outer = 1:PE_R-1
        if(par_on==1)
            parfor coil_selec = 1:size(conc_acs_fregion,3)
                ak(:,coil_selec,outer) = pre_cal_A*squeeze(pre_cal_B(:,coil_selec,outer));
                %disp(['Coil ' num2str(coil_selec) ' weights are ready!'])
            end
        else
            for coil_selec = 1:size(conc_acs_fregion,3)
                ak(:,coil_selec,outer) = pre_cal_A*squeeze(pre_cal_B(:,coil_selec,outer));
                %disp(['Coil ' num2str(coil_selec) ' weights are ready!'])
            end
        end
    end
    
    clear MA
    clear Mk
    
    data_ak_ind(klm,:,:,:) = ak;
    
    %%%% trial
    
end
display('2nd set of kernels are readys!')



sg = squeeze(sg_kspace(:,:,:,1,1));
%%% sampling points%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
non_acq_p = squeeze(sg(:,:,1))==0;
acq_p = ones(size(sg_kspace,1),size(sg_kspace,2),size(sg_kspace,3),'single')-non_acq_p;
non_acq_p = logical(non_acq_p);
loc_mask = logical(acq_p);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for klm=1:slice_R
    if(gui_on==1)
        progressbar(klm/(slice_R+1))
    end
    if(par_on==1)
        parfor selected_cyc =1:ims
            
            sg = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
            
            if(PE_R==2)
                %% mid
                ak_ind = squeeze(data_ak_ind(klm,:,:,1));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new2 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                %% right
                mb_kspace_new = mb_kspace_new2;
            elseif(PE_R==3)
                %% left
                ak_ind = squeeze(data_ak_ind(klm,:,:,1));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new1 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new1 = circshift(mb_kspace_new1,[0 0 0]);
                %% mid
                ak_ind = squeeze(data_ak_ind(klm,:,:,2));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new2 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new2 = circshift(mb_kspace_new2,[0 +1 0]);
                
                mb_kspace_new = mb_kspace_new1+mb_kspace_new2;
            elseif(PE_R==4)
                %% left
                ak_ind = squeeze(data_ak_ind(klm,:,:,1));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new1 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new1 = circshift(mb_kspace_new1,[0 -1 0]);
                %% mid
                ak_ind = squeeze(data_ak_ind(klm,:,:,2));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new2 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new2 = circshift(mb_kspace_new2,[0 0 0]);
                %% right
                ak_ind = squeeze(data_ak_ind(klm,:,:,3));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new3 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new3 = circshift(mb_kspace_new3,[0 1 0]);
                
                mb_kspace_new = mb_kspace_new1+mb_kspace_new2+mb_kspace_new3;
            end
            
            res2 = zeros(size(new_kspace,1),size(new_kspace,2),size(new_kspace,3),'single');
            res2 = mb_kspace_new;
            
            res2(loc_mask) = sg(loc_mask);
            
            hf_kspace(klm,:,:,:,selected_cyc) = res2;
            
        end
    else
        for selected_cyc =1:ims
            
            sg = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
            
            if(PE_R==2)
                %% mid
                ak_ind = squeeze(data_ak_ind(klm,:,:,1));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new2 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                %% right
                mb_kspace_new = mb_kspace_new2;
            elseif(PE_R==3)
                %% left
                ak_ind = squeeze(data_ak_ind(klm,:,:,1));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new1 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new1 = circshift(mb_kspace_new1,[0 0 0]);
                %% mid
                ak_ind = squeeze(data_ak_ind(klm,:,:,2));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new2 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new2 = circshift(mb_kspace_new2,[0 +1 0]);
                
                mb_kspace_new = mb_kspace_new1+mb_kspace_new2;
            elseif(PE_R==4)
                %% left
                ak_ind = squeeze(data_ak_ind(klm,:,:,1));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new1 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new1 = circshift(mb_kspace_new1,[0 -1 0]);
                %% mid
                ak_ind = squeeze(data_ak_ind(klm,:,:,2));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new2 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new2 = circshift(mb_kspace_new2,[0 0 0]);
                %% right
                ak_ind = squeeze(data_ak_ind(klm,:,:,3));
                mb_kspce = squeeze(sg_kspace(:,:,:,klm,selected_cyc));
                end_kspace = conv_op_ind_sg2_54(PE_R,ak_ind,mb_kspce,kernel_row,kernel_col,size(mb_kspce,1),size(mb_kspce,2),size(mb_kspce,3));
                mb_kspace_new3 = reshape(end_kspace,[size(mb_kspce,1) size(mb_kspce,2) size(mb_kspce,3)]);
                mb_kspace_new3 = circshift(mb_kspace_new3,[0 1 0]);
                
                mb_kspace_new = mb_kspace_new1+mb_kspace_new2+mb_kspace_new3;
            end
            
            res2 = zeros(size(new_kspace,1),size(new_kspace,2),size(new_kspace,3),'single');
            res2 = mb_kspace_new;
            
            res2(loc_mask) = sg(loc_mask);
            
            hf_kspace(klm,:,:,:,selected_cyc) = res2;
            
        end
    end
    disp(['Slice: ',num2str(klm),' is ready'])
end
if(gui_on==1)
    progressbar(1)
end

end

