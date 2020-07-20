% This demo is actually pretty short, just as the last step in part (i), which finally save cfls as the training data.
% It is mainly based on function DL_savecfl_nonc_withb0, which are called multiple times for different datasets.

addpath(genpath(pwd))

filepaths = {'/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3/llr_0002_corr'; ...
    '/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3_nex2/llr_0002_corr'; ...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser3/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser4/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser9/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser10/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser4/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser5/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser4/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser5/llr_0002_corr';...
}

Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/training_nonc']; % 10 DTI scans

flag_nonc = 1;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);
% 
% generating file 1~360 from 1:/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3/llr_0002_corr
% generating file 361~720 from 2:/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3_nex2/llr_0002_corr
% generating file 721~2520 from 3:/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser3/llr_0002_corr
% generating file 2521~4020 from 4:/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser4/llr_0002_corr
% generating file 4021~5820 from 5:/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser9/llr_0002_corr
% generating file 5821~7320 from 6:/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser10/llr_0002_corr
% generating file 7321~8400 from 7:/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser4/llr_0002_corr
% generating file 8401~9210 from 8:/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser5/llr_0002_corr
% generating file 9211~9810 from 9:/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser4/llr_0002_corr
% generating file 9811~10260 from 10:/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser5/llr_0002_corr

%%
filepaths = {'/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3/llr_0002_corr'; ...
    '/bmrNAS/people/yuxinh/DTI_20170816/16Aug17_Ex12278_Ser3_nex2/llr_0002_corr'; ...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser3/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser4/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser9/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180513/13May18_Ex3787_Ser10/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser4/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180603/03Jun18_Ex3966_Ser5/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser4/llr_0002_corr';...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser5/llr_0002_corr';...
};
Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/training_llr']; % 10 DTI scans

flag_nonc = 0;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);

%% validation, test dti data (3 cases)

filepaths = { ...
    '/bmrNAS/people/yuxinh/highres_brain/Exam5908/30Apr19_Ex5908_Ser6/llr_0002_corr'; ... % validation data
    '/bmrNAS/people/yuxinh/revision_nonc/04Jun19_Ex6080_Ser3/llr_0002_corr'; ... % test data
    '/bmrNAS/people/yuxinh/DTI_20191211/11Dec19_Ex7181_Ser3/llr_0002_corr'; ...
}

Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/test_nonc']; % 10 DTI scans
flag_nonc = 1;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);


Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/test_llr'];
flag_nonc = 0;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);

% 
% generating file 1~360 from 1:/bmrNAS/people/yuxinh/highres_brain/Exam5908/30Apr19_Ex5908_Ser6/llr_0002_corr
% generating file 361~720 from 2:/bmrNAS/people/yuxinh/revision_nonc/04Jun19_Ex6080_Ser3/llr_0002_corr
% generating file 721~1260 from 3:/bmrNAS/people/yuxinh/DTI_20191211/11Dec19_Ex7181_Ser3/llr_0002_corr
%% 6nex data (2 cases)
filepaths = { ...
    '/bmrNAS/people/yuxinh/ms_20170824/24Aug17_Ex12345_Ser5/llr_0002_corr'; ... %6nex data
    '/bmrNAS/people/yuxinh/DTI_20191211/11Dec19_Ex7181_Ser2/llr_0002_corr'; ... %6nex data
}

Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/test_6nex_llr'];
flag_nonc = 0;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);
% 
% generating file 1~54 from 1:/bmrNAS/people/yuxinh/ms_20170824/24Aug17_Ex12345_Ser5/llr_0002_corr
% generating file 55~246 from 2:/bmrNAS/people/yuxinh/DTI_20191211/11Dec19_Ex7181_Ser2/llr_0002_corr

%% breast test data (12 cases)

filepaths = {...
'/bmrNAS/people/yuxinh/breast_bw2/17Oct18_Ex4743_Ser7/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/24Oct18_Ex4819_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/24Oct18_Ex4821_Ser7/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/25Oct18_Ex4827_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4837_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4840_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4841_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4842_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/30Oct18_Ex4866_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/26Feb19_Ex5864_Ser6/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/26Feb19_Ex5864_Ser7/prep_llr_nocp_bart_0002',...
'/bmrNAS/people/yuxinh/breast_bw2/04Mar19_Ex5908_Ser6/prep_llr_nocp_bart_0002',...
};


Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/test_breast_llr'];
flag_nonc = 0;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);

% 
% generating file 1~84 from 1:/bmrNAS/people/yuxinh/breast_bw2/17Oct18_Ex4743_Ser7/prep_llr_nocp_bart_0002
% generating file 85~168 from 2:/bmrNAS/people/yuxinh/breast_bw2/24Oct18_Ex4819_Ser6/prep_llr_nocp_bart_0002
% generating file 169~252 from 3:/bmrNAS/people/yuxinh/breast_bw2/24Oct18_Ex4821_Ser7/prep_llr_nocp_bart_0002
% generating file 253~336 from 4:/bmrNAS/people/yuxinh/breast_bw2/25Oct18_Ex4827_Ser6/prep_llr_nocp_bart_0002
% generating file 337~420 from 5:/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4837_Ser6/prep_llr_nocp_bart_0002
% generating file 421~504 from 6:/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4840_Ser6/prep_llr_nocp_bart_0002
% generating file 505~588 from 7:/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4841_Ser6/prep_llr_nocp_bart_0002
% generating file 589~672 from 8:/bmrNAS/people/yuxinh/breast_bw2/26Oct18_Ex4842_Ser6/prep_llr_nocp_bart_0002
% generating file 673~756 from 9:/bmrNAS/people/yuxinh/breast_bw2/30Oct18_Ex4866_Ser6/prep_llr_nocp_bart_0002
% generating file 757~844 from 10:/bmrNAS/people/yuxinh/breast_bw2/26Feb19_Ex5864_Ser6/prep_llr_nocp_bart_0002
% generating file 845~932 from 11:/bmrNAS/people/yuxinh/breast_bw2/26Feb19_Ex5864_Ser7/prep_llr_nocp_bart_0002
% generating file 933~1016 from 12:/bmrNAS/people/yuxinh/breast_bw2/04Mar19_Ex5908_Ser6/prep_llr_nocp_bart_0002


%%
filepaths = { ...
    '/bmrNAS/people/yuxinh/DTI_20180609/09Jun18_Ex4010_Ser5/llr_0002_corr';...
}

Outpath = ['/bmrNAS/people/yuxinh/DL_data_revision/training_small_nonc']; % 10 DTI scans

flag_nonc = 1;
DL_savecfl_nonc_withb0(filepaths, Outpath, 0, flag_nonc);
% 