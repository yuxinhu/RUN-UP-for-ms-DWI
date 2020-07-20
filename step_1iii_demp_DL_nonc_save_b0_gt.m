% Part1: save the b0 images for each case.
% Part2: reload the joint reconstruction results, and save them into one file for each slice and direction.
addpath(genpath(pwd))


ns = [12 12 30 25 30 25 24 18 20 15 12 12 18]; % #slices
nd = [30 30 60 60 60 60 45 45 30 30 30 30 30]; % #directions

filepaths = {'/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3/'; ...
    '/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3_nex2/'; ...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser3/';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser4/';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser9/';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser10/';...
    '/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser4/';...
    '/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser5/';...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser4/';...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser5/';...
    '/bmrNAS/people/yuxinh/highres_brain/Exam5908/30Apr19_Ex5908_Ser6/'; ... % validation data
    '/bmrNAS/people/yuxinh/revision_nonc/04Jun19_Ex6080_Ser3/'; ... % test data
    '/bmrNAS/people/yuxinh/DTI_20191211/11Dec19_Ex7181_Ser3/'; ...
}


%% Part 1
NX = 256;
NY = 256; % Zero-padding size, should be 360 for breast data

for k = 1 : length(filepaths)
    clear smap klb_cc p
    filepath = filepaths{k};
    
    % sensitivity map loading
    for s = 1 : ns(k)
        load([filepath, 'SENSE_gcc8/b0', num2str(s), '.mat'])
        smap(:,:,:,s) = sens;
    end
    
    % normalize and coil compress b0 data
    load([filepath, 'k1.mat'])
    k0 = zero_pad(k0(:,:,:,1,:), [NX, NY, p.nc, 1, ns(k)]);
    if size(k0,5) ~= ns(k)
        disp([num2str(k) ' filepath #slices wrong'])
    end
    
    k0 = k0 / p.scale;
    for s = 1 : ns(k)
        [klb_cc(:,:,:,s),~] = GCC(squeeze(k0(:,:,:,1,s)), k0(:,:,:,1,s), p.v, [40,1]);
    end
    b0 = squeeze(sum(ifft2c(klb_cc) .* conj(smap),3));
    save([filepath, 'llr_0002_corr/smap.mat'], 'smap', 'p');
    save([filepath, 'llr_0002_corr/b0.mat'], 'b0');
end


% Part 2
% constructs all "gt" files which includes all joint reconstruction results
% from the joint reconstruction results for the deep learning reconstruction


for k = 1 : length(filepaths)
    filepath = filepaths{k};
    com = ['rm -rf ', filepath, 'llr_0002_corr/gt*mat'];
    system(com);
    for s = 1 : ns(k)
        load([filepath, 'gcc8_m01p0i100_sense/s', num2str(s), '.mat'])
        for d = 1 : nd(k)
            im1 = mag(:,:,d);
            phase1 = phase(:,:,:,d);
            index = (d-1) * ns(k) + s;
            save([filepath, 'llr_0002_corr/gt', num2str(index), '.mat'], 'im1', 'phase1')
        end
    end
end