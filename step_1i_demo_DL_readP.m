clear all;
% This demo loads the kspace data from the Pfiles of multiple scans with modified Orchestra (Matlab, SDK 1.7-1).

addpath(genpath('/bmrNAS/people/yuxinh/bart/orchestra-sdk-1.7-1.matlab')) % Add installed Orchestra into the path. You probably need to change this.
addpath(genpath(pwd))



pdirname = {...
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

for n = 1 : length(pdirname)
    dirname1 = ['/bmrNAS/people/yuxinh/breast_bw2/', pdirname{n}]; % % where pfiles (and vrgf and ref files) are saved
    filenames=dir(dirname1); 
    p = findPfile(filenames);
    EpiDiffusionRecon_yuxin([dirname1,'/',p,'.7'], dirname1);
end


