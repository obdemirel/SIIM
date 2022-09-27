function [sense1_images] = sense_maker(hf_kspace,sense_maps,ims)


kspace_to_im = @(x) ifft2c(x);% * sqrt(size(x,1) * size(x,2));
sense1_images = zeros(size(hf_kspace,2),size(hf_kspace,3),size(sense_maps,4),ims,'single');

for asd1 = 1:size(hf_kspace,5)
    
    for ii = 1:size(hf_kspace,1)
        a1 = squeeze(hf_kspace(ii,:,:,:,asd1));
        a1 = squeeze((sum(conj(squeeze(sense_maps(:,:,:,ii))) .* kspace_to_im(a1),3)));
        sense1_images(:,:,ii,asd1) = a1;
    end
end

end