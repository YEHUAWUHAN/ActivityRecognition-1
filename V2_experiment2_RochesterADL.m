%% influence of the pose-stip combination
clear all;
clc;
addpath(genpath('SVM-chi-square-master'));
addpath(genpath('fcl-master/matlab/kmeans'));
addpath(genpath('tSNE'));


%% Rochester ADL dataset
clear all;
clc;
disp('=============== evaluation on RochesterADL dataset===========');
option = V2_GetDefaultConfig('RochesterADL');

%%% this script implements the pipeline of bag of features for action
%%% recognition in RochesterADL
subjects = 5; % in the implementation, we perform leave-one-subject-out.
eval_res = {};
stip_data_S = {};
stip_data_T = {};


for ss = subjects
    fprintf('- leave subject %i out...\n',ss);
    disp('-- loading data...\n');
    load(sprintf('data_RochesterADL/stip_data_%i.mat',ss));% load pre-stored data
    
    disp('-- learning visual word...');
    % learn visual word
    [codebook_stip,SUMD_stip,opts_stip,running_info_stip,mu_stip,sigma_stip]...
        = V2_CreateCodebook(stip_data_S{ss},option,'stip');
    eval_res{ss}.CodebookLearning.codebook_stip = codebook_stip;
    eval_res{ss}.CodebookLearning.SUMD_stip = SUMD_stip;
    eval_res{ss}.CodebookLearning.opts_stip = opts_stip;
    eval_res{ss}.CodebookLearning.running_info_stip = running_info_stip;
    eval_res{ss}.RawFeatureStandardization.mu_stip = mu_stip;
    eval_res{ss}.RawFeatureStandardization.sigma_stip = sigma_stip;

    [codebook_hra,SUMD_hra,opts_hra,running_info_hra,mu_hra,sigma_hra]...
        = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.HeadRightArm');
    eval_res{ss}.CodebookLearning.codebook_hra = codebook_hra;
    eval_res{ss}.CodebookLearning.SUMD_hra = SUMD_hra;
    eval_res{ss}.CodebookLearning.opts_hra = opts_hra;
    eval_res{ss}.CodebookLearning.running_info_hra = running_info_hra;
    eval_res{ss}.RawFeatureStandardization.mu_hra = mu_hra;
    eval_res{ss}.RawFeatureStandardization.sigma_hra = sigma_hra;

    [codebook_hla,SUMD_hla,opts_hla,running_info_hla,mu_hla,sigma_hla]...
        = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.HeadLeftArm');
    eval_res{ss}.CodebookLearning.codebook_hla = codebook_hla;
    eval_res{ss}.CodebookLearning.SUMD_hla = SUMD_hla;
    eval_res{ss}.CodebookLearning.opts_hla = opts_hla;
    eval_res{ss}.CodebookLearning.running_info_hla = running_info_hla;
    eval_res{ss}.RawFeatureStandardization.mu_hla = mu_hla;
    eval_res{ss}.RawFeatureStandardization.sigma_hla = sigma_hla;


    [codebook_trl,SUMD_trl,opts_trl,running_info_trl,mu_trl,sigma_trl]...
        = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.TorsoRightLeg');
    eval_res{ss}.CodebookLearning.codebook_trl = codebook_trl;
    eval_res{ss}.CodebookLearning.SUMD_trl = SUMD_trl;
    eval_res{ss}.CodebookLearning.opts_trl = opts_trl;
    eval_res{ss}.CodebookLearning.running_info_trl = running_info_trl;
    eval_res{ss}.RawFeatureStandardization.mu_trl = mu_trl;
    eval_res{ss}.RawFeatureStandardization.sigma_trl = sigma_trl;


    [codebook_tll,SUMD_tll,opts_tll,running_info_tll,mu_tll,sigma_tll]...
        = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.TorsoLeftLeg');
    eval_res{ss}.CodebookLearning.codebook_tll = codebook_tll;
    eval_res{ss}.CodebookLearning.SUMD_tll = SUMD_tll;
    eval_res{ss}.CodebookLearning.opts_tll = opts_tll;
    eval_res{ss}.CodebookLearning.running_info_tll = running_info_tll;
    eval_res{ss}.RawFeatureStandardization.mu_tll = mu_tll;
    eval_res{ss}.RawFeatureStandardization.sigma_tll = sigma_tll;

    disp('-- evaluation...');
    option.features.type = 'accumulated';
    stip_S_encoded = V2_LocalFeatureEncoding(stip_data_S{ss},eval_res{ss},option);
    option.features.type = 'accumulated';
    stip_T_encoded = V2_LocalFeatureEncoding(stip_data_T{ss},eval_res{ss},option);

    eval_res{ss}.stip_S_encoded = stip_S_encoded;
    eval_res{ss}.stip_T_encoded = stip_T_encoded;

    %%% multi-class svm train for snippets
    [model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
       V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);

    eval_res{ss}.svm.model = model;
    eval_res{ss}.svm.Yt = Yt;
    eval_res{ss}.svm.Ytp = Ytp;
    eval_res{ss}.svm.prob_estimates = prob_estimates;
    eval_res{ss}.svm.meta_res = meta_res;        
    eval_res{ss}.svm.test_labels = test_labels;
    clear stip_data_S stip_data_T;

end

save(option.fileIO.eval_res_file,'eval_res');
save(option.fileIO.option_file,'option');
