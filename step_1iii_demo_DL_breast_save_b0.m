% Save the b0 images for each breast case.
clear all
filepaths = {...
'17Oct18_Ex4743_Ser7',...
'24Oct18_Ex4819_Ser6',...
'24Oct18_Ex4821_Ser7',...
'25Oct18_Ex4827_Ser6',...
'26Oct18_Ex4837_Ser6',...
'26Oct18_Ex4840_Ser6',...
'26Oct18_Ex4841_Ser6',...
'26Oct18_Ex4842_Ser6',...
'30Oct18_Ex4866_Ser6',...
'26Feb19_Ex5864_Ser6',...
'26Feb19_Ex5864_Ser7',...
'04Mar19_Ex5908_Ser6',...
};

NX = 360;
NY = 360;

%%

for k = 1 : length(filepaths)
    clear smap k0 p
    filepath = ['/bmrNAS/people/yuxinh/breast_bw2/',  filepaths{k}];
    savepath = ['/bmrNAS/people/yuxinh/breast_bw2/',  filepaths{k}, '/prep_llr_nocp_bart_0002'];
    % sensitivity map loading
    load([filepath, '/k1.mat'])

    for s = 1 : size(k0, 5)
        load([savepath, '/b0', num2str(s), '.mat'])
        smap(:,:,:,s) = sens;
    end
    
    % normalize and coil compress b0 data
    k0 = zero_pad(k0(:,:,:,1,:), [NX, NY, p.nc, 1, size(k0,5)]);

    
    k0 = squeeze(k0) / p.scale;

    b0 = squeeze(sum(ifft2c(k0) .* conj(smap),3));
    save([savepath, '/smap.mat'], 'smap', 'p');
    save([savepath, '/b0.mat'], 'b0');
end
