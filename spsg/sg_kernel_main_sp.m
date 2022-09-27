function [data_ak_ind,kernel_row,kernel_col] = sg_kernel_main_sp(slice_R,PE_R,new_kspace,acs,kernels,gui_on,par_on)

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

for klm =1:slice_R
    
    
    reg = 0;
    
    %%% KERNEL %%%%%%%%% example of R2
    % x0x0x0x0x        x0x0x0x0x
    % x0x0x0x0x        x0x0x0x0x
    % x0x0x0x0x    ->  x0x0*0x0x
    % x0x0x0x0x        x0x0x0x0x
    % x0x0x0x0x        x0x0x0x0x
    %%%%%%%%%%%%%%%%%%%%
    
    conc_acs_fregion = squeeze(acs(:,:,:,klm));
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
    
    MA_big((klm-1)*size(MA,1) + 1 :klm*size(MA,1),:) = MA;
end

pre_cal_A = pinv(MA_big'*MA_big);

for klm=1:slice_R
    
    
    conc_acs_fregion = squeeze(acs(:,:,:,klm));
    row_start = ceil(kernel_row/2);
    row_end = size(conc_acs_fregion,1)-floor(kernel_row/2);
    
    
    %Mk = zeros(size(MA,1),size(conc_acs_fregion,3),'single');
    %% Mk vectors filling
    for coil_selec = 1:size(conc_acs_fregion,3)
        selected_acs = conc_acs_fregion(row_start:row_end,mean(acq_cols(1:kernel_col)):mean(acq_cols_back(1:kernel_col)),coil_selec); %5,60 for 5x5
        Mk(:,coil_selec) = selected_acs(:);
    end
    %disp('Mk is ready!')
    
    
    new_Mk = zeros(size(MA_big,1),size(acs,3),'single');
    new_Mk((klm-1)*size(MA,1) + 1 :klm*size(MA,1),:) = Mk;
    
    ak = zeros(kernel_dim*size(kspace,3),size(kspace,3),'single');
    
    pre_cal_B = MA_big'*new_Mk;
    
    if(par_on==1)
        parfor coil_selec = 1:size(conc_acs_fregion,3)
            % tic
            ak(:,coil_selec) = pre_cal_A*pre_cal_B(:,coil_selec);
            % toc
        end
    else
        for coil_selec = 1:size(conc_acs_fregion,3)
            if(gui_on==1)
                progressbar(coil_selec/(size(conc_acs_fregion,3)+1))
            end
            % tic
            ak(:,coil_selec) = pre_cal_A*pre_cal_B(:,coil_selec);
            % toc
        end
        if(gui_on==1)
            progressbar(1)
        end
    end
    
    data_ak_ind(klm,:,:) = ak;
end

end

