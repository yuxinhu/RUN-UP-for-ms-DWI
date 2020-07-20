% This demo loads and savesthe deep learning reconstruction results.
nx = 256; 
ny = 256; 
nc = 8; % Number of coils
ns = 4; % Number of shots
total = 1016; % Total number of files

% output = zeros(nx, ny, total);
% target = zeros(nx, ny, total);

filepath = ['/bmrNAS/people/yuxinh/DL_val_cfls_breast_res/ki_enhanced_iter8_nolastunet']; % where the results are saved.
for i = 1 : total
    output(:,:,i)  = sos(readcfl([filepath '/images/tr' num2str(i) '-outputs']))';
    target(:,:,i)  = sos(readcfl([filepath '/images/tr' num2str(i) '-targets']))';
end

save([filepath,'/res.mat'],'output','target')

%% Compared the results with the targe (PSNR, NRMSE, SSIM)
[p,p2,p3] = cal_value(output, target);
