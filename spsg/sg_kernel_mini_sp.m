function [data_ak_small,kernel_row,kernel_col] = sg_kernel_mini_sp(slice_R,PE_R,new_kspace,acs,kernels,gui_on,par_on)

kspace = new_kspace;
kernel_row = kernels(1);
kernel_col = kernels(2);

kernel_dim = kernel_row*kernel_col;
conc_acs_fregion = squeeze(acs(:,:,:,1)); % dummy for sizes only

acq_cols = 1:PE_R:size(conc_acs_fregion,2);
col_adder = acq_cols(kernel_col)-acq_cols(1);

acq_cols_back = size(conc_acs_fregion,2):-PE_R:1;

MA = zeros((size(conc_acs_fregion,1)-(kernel_row-1))*(size(conc_acs_fregion,2)-col_adder),kernel_dim*size(kspace,3),'single');
MA_big = zeros(slice_R*size(MA,1),size(MA,2),'single');

for klm=1:slice_R
    
    reg = 0;
    
    %%% KERNEL %%%%%%%%%
    % xxxxx        xxxxx
    % xxxxx        xxxxx
    % xxxxx    ->  xx*xx
    % xxxxx        xxxxx
    % xxxxx        xxxxx
    %%%%%%%%%%%%%%%%%%%%
    
    conc_acs_fregion = squeeze(acs(:,:,:,klm));
    %% MA matrix filling
    for coil_selec = 1:size(conc_acs_fregion,3)
        selected_acs = conc_acs_fregion(:,:,coil_selec);
        row_count = 1;
        for col = 1:size(selected_acs,2)-(kernel_col-1)
            for row = 1:size(selected_acs,1)-(kernel_row-1)
                neighbors = selected_acs(row:row+(kernel_row-1),col:col+(kernel_col-1));
                neighbors = neighbors(:).';
                MA(row_count,(coil_selec-1)*(kernel_dim) +1:coil_selec*(kernel_dim)) = neighbors;
                row_count = row_count+1;
            end
        end
    end
    
    MA_big((klm-1)*size(MA,1) + 1 :klm*size(MA,1),:) = MA;
    
end

pre_cal_A = pinv(MA_big'*MA_big);

for klm=1:slice_R
    
    conc_acs_fregion = squeeze(acs(:,:,:,klm));
    
    row_start = ceil(kernel_row/2);
    row_end = size(conc_acs_fregion,1)-floor(kernel_row/2);
    col_start = ceil(kernel_col/2);
    col_end = size(conc_acs_fregion,2)-floor(kernel_col/2);
    
    %Mk = zeros(size(MA,1),size(conc_acs_fregion,3),'single');
    %% Mk vectors filling
    for coil_selec = 1:size(conc_acs_fregion,3)
        selected_acs = conc_acs_fregion(row_start:row_end,col_start:col_end,coil_selec);
        Mk(:,coil_selec) = selected_acs(:);
    end
    
    
    
    new_Mk = zeros(size(MA_big,1),size(acs,3),'single');
    new_Mk((klm-1)*size(MA,1) + 1 :klm*size(MA,1),:) = Mk;
    
    
    ak = zeros(kernel_dim*size(kspace,3),size(kspace,3),2,'single');
    
    pre_cal_B = MA_big'*new_Mk;
    
    if(par_on==1)
        parfor coil_selec = 1:size(conc_acs_fregion,3)
            ak(:,coil_selec) = pre_cal_A*pre_cal_B(:,coil_selec);
            %disp(['Coil ' num2str(coil_selec) ' weights are ready!'])
        end
    else
        for coil_selec = 1:size(conc_acs_fregion,3)
            if(gui_on==1)
                progressbar(coil_selec/(size(conc_acs_fregion,3)+1))
            end
            ak(:,coil_selec) = pre_cal_A*pre_cal_B(:,coil_selec);
            %disp(['Coil ' num2str(coil_selec) ' weights are ready!'])
        end
        if(gui_on==1)
            progressbar(1)
        end
    end
    
    data_ak_small(klm,:,:,:) = ak;
end

end

