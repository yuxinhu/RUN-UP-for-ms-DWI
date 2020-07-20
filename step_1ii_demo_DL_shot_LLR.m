% Demo of applying shot-LLR reconstruction for different cases. 
% Saving the k-space data (for future deep learning reconstruction) and shot-LLR results (for comparison) for each direction and slice.

clear all
addpath(genpath(pwd))
% Maybe also need to include BART path.

pdirname = {...
'/ms_20170718/17Jul17_Ex12110_Ser2',...
'/ms_20170718/17Jul17_Ex12110_Ser8',...
'/ms1_20170714/14Jul17_Ex12091_Ser3',...
'/ms1_20170714/14Jul17_Ex12091_Ser9',...
'/ms1_20170714/14Jul17_Ex12091_Ser13',...
'/ms2_20170714/14Jul17_Ex12092_Ser8',...
'/ms2_20170714/14Jul17_Ex12092_Ser12',...
'/ms_20170719/19Jul17_Ex12124_Ser2',...
'/ms_20170719/19Jul17_Ex12124_Ser8',...
'/MS_20170816/16Aug17_Ex12279_Ser2',...
'/MS_20170816/16Aug17_Ex12279_Ser4',...
'/MS_20170816/16Aug17_Ex12279_Ser8',...
'/ms_20170824/24Aug17_Ex12345_Ser5',...
};

NX = 256; % Size for zero-filling
NY = 256; % Size for zero-filling

% NX = 360;
% NY = 360; % for breast

p.ndir = 1; % Number of diffusion-encoding directions
p.lambda = 0.0008; % Regularization parameter for shot-LLR
p.gcc = true; % whether or not to use coil compresssion
p.v = 8;  % Number of virtual coils to be compressed
p.reconmethod = 'LLR'; % Reconstruction method to be used.
p.b0 = []; % This is a list storing all the non-diffusion-weighted scans (not including the first b=0 scan). Usually it is empty.
% If using Qiyuan's tensor file for acquisition, set this to "1:15:p.ndir", since the first of every fifteen directions would be the non-diffusion-weighted scan.


%%
for n = 1 : length(pdirname)
    p.filename = ['/bmrNAS/people/yuxinh', pdirname{n}];
    p.savepath = [p.filename,'/prep_llr_bart_00008'];
    if(~exist(p.savepath))
        mkdir([p.savepath]);
    end
    
    image_index = 1;
    for file_index = 1 : p.ndir
        p.dir = file_index;
        disp(['loading file from ',num2str(file_index),':',p.filename])
        DL_LLRini;
    end
end

