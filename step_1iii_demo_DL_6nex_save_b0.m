% Save the b0 images for each brain 6-nex case.



filepaths = { ...
    '/bmrNAS/people/yuxinh/ms_20170824/24Aug17_Ex12345_Ser5/'; ... %6nex data
    '/bmrNAS/people/yuxinh/DTI_20191211/11Dec19_Ex7181_Ser2/'; ... %6nex data
}

NX = 256;
NY = 256;
for k = 1 : length(filepaths)
    clear smap klb_cc k0 p
    filepath = filepaths{k};
    
    load([filepath, 'k1.mat'])
    k0(:,2:2:end,:,:,:) = -k0(:,2:2:end,:,:,:); % only for 1st, to shift fov
    % sensitivity map loading
    for s = 1 : size(k0, 5)
        load([filepath, 'llr_0002_corr/b0', num2str(s), '.mat'])
        smap(:,:,:,s) = sens;
    end
    
    % normalize and coil compress b0 data
    k0 = zero_pad(k0(:,:,:,1,:), [NX, NY, p.nc, 1, size(k0, 5)]);

    
    k0 = k0 / p.scale;
    for s = 1 : size(k0, 5)
        [klb_cc(:,:,:,s),~] = GCC(squeeze(k0(:,:,:,1,s)), k0(:,:,:,1,s), p.v, [40,1]);
    end
    b0 = squeeze(sum(ifft2c(klb_cc) .* conj(smap),3));
    save([filepath, 'llr_0002_corr/smap.mat'], 'smap', 'p');
    save([filepath, 'llr_0002_corr/b0.mat'], 'b0');
end
