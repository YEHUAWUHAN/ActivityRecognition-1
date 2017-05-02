%% influence of the pose-stip combination
clear all;
clc;
addpath(genpath('SVM-chi-square-master'));
addpath(genpath('fcl-master/matlab/kmeans'));
addpath(genpath('tSNE'));

%% KTH dataset
fprintf('========Evaluation on KTH dataset=======\n');
%%% get features and options
disp('-loading data...');
load('KTH_FeatureData_all.mat');
option = V2_GetDefaultConfig('KTH'); 

disp('-learn visual words...'); 
%%% learn visual words
[codebook_stip,SUMD_stip,opts_stip,running_info_stip,mu_stip,sigma_stip]...
    = V2_CreateCodebook(stip_data_S,option,'stip');
eval_res.CodebookLearning.codebook_stip = codebook_stip;
eval_res.CodebookLearning.SUMD_stip = SUMD_stip;
eval_res.CodebookLearning.opts_stip = opts_stip;
eval_res.CodebookLearning.running_info_stip = running_info_stip;
eval_res.RawFeatureStandardization.mu_stip = mu_stip;
eval_res.RawFeatureStandardization.sigma_stip = sigma_stip;

[codebook_hra,SUMD_hra,opts_hra,running_info_hra,mu_hra,sigma_hra]...
    = V2_CreateCodebook(stip_data_S,option,'skeleton.HeadRightArm');
eval_res.CodebookLearning.codebook_hra = codebook_hra;
eval_res.CodebookLearning.SUMD_hra = SUMD_hra;
eval_res.CodebookLearning.opts_hra = opts_hra;
eval_res.CodebookLearning.running_info_hra = running_info_hra;
eval_res.RawFeatureStandardization.mu_hra = mu_hra;
eval_res.RawFeatureStandardization.sigma_hra = sigma_hra;

[codebook_hla,SUMD_hla,opts_hla,running_info_hla,mu_hla,sigma_hla]...
    = V2_CreateCodebook(stip_data_S,option,'skeleton.HeadLeftArm');
eval_res.CodebookLearning.codebook_hla = codebook_hla;
eval_res.CodebookLearning.SUMD_hla = SUMD_hla;
eval_res.CodebookLearning.opts_hla = opts_hla;
eval_res.CodebookLearning.running_info_hla = running_info_hla;
eval_res.RawFeatureStandardization.mu_hla = mu_hla;
eval_res.RawFeatureStandardization.sigma_hla = sigma_hla;


[codebook_trl,SUMD_trl,opts_trl,running_info_trl,mu_trl,sigma_trl]...
    = V2_CreateCodebook(stip_data_S,option,'skeleton.TorsoRightLeg');
eval_res.CodebookLearning.codebook_trl = codebook_trl;
eval_res.CodebookLearning.SUMD_trl = SUMD_trl;
eval_res.CodebookLearning.opts_trl = opts_trl;
eval_res.CodebookLearning.running_info_trl = running_info_trl;
eval_res.RawFeatureStandardization.mu_trl = mu_trl;
eval_res.RawFeatureStandardization.sigma_trl = sigma_trl;

[codebook_tll,SUMD_tll,opts_tll,running_info_tll,mu_tll,sigma_tll]...
    = V2_CreateCodebook(stip_data_S,option,'skeleton.TorsoLeftLeg');
eval_res.CodebookLearning.codebook_tll = codebook_tll;
eval_res.CodebookLearning.SUMD_tll = SUMD_tll;
eval_res.CodebookLearning.opts_tll = opts_tll;
eval_res.CodebookLearning.running_info_tll = running_info_tll;
eval_res.RawFeatureStandardization.mu_tll = mu_tll;
eval_res.RawFeatureStandardization.sigma_tll = sigma_tll;

%%% loop over different coding methods
fprintf('-evaluation...');

stip_S_encoded = V2_LocalFeatureEncoding(stip_data_S,eval_res,option);
option.features.type = 'accumulated';
stip_T_encoded = V2_LocalFeatureEncoding(stip_data_T,eval_res,option);

eval_res.stip_S_encoded = stip_S_encoded;
eval_res.stip_T_encoded = stip_T_encoded;

[model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);

eval_res.svm.model = model;
eval_res.svm.Yt = Yt;
eval_res.svm.Ytp = Ytp;
eval_res.svm.prob_estimates = prob_estimates;
eval_res.svm.meta_res = meta_res;
eval_res.svm.test_labels = test_labels;    

save(option.fileIO.eval_res_file,'eval_res');
save(option.fileIO.option_file,'option');

