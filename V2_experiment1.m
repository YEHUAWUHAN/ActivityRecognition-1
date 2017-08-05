%% influence of the pose-stip combination
clear all;
clc;
addpath(genpath('SVM-chi-square-master'));
addpath(genpath('fcl-master/matlab/kmeans'));
addpath(genpath('vlfeat-0.9.20/toolbox'));
run('vlfeat-0.9.20/toolbox/vl_setup')

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
weightCandidates = 0:10;
fprintf('-evaluation...');

for aa = weightCandidates    
    option.features.pose_stip_weight = aa/10;
%     fprintf('- feature encoding..\n');
    stip_S_encoded = V2_LocalFeatureEncoding(stip_data_S,eval_res,option);
    stip_T_encoded = V2_LocalFeatureEncoding(stip_data_T,eval_res,option);
%     fprintf('- feature encoding end.\n');
    
    eval_res.stip_S_encoded{aa+1} = stip_S_encoded;
    eval_res.stip_T_encoded{aa+1} = stip_T_encoded;
    
    [model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
    V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);

    eval_res.svm.model{aa+1} = model;
    eval_res.svm.Yt{aa+1} = Yt;
    eval_res.svm.Ytp{aa+1} = Ytp;
    eval_res.svm.prob_estimates{aa+1} = prob_estimates;
    eval_res.svm.meta_res{aa+1} = meta_res;
    
    clear stip_S_encoded stip_T_encoded
end

save(option.fileIO.eval_res_file,'eval_res','-v7.3');
save(option.fileIO.option_file,'option','-v7.3');

%% Rochester ADL dataset
% clear all;
% clc;
% disp('=============== evaluation on RochesterADL dataset===========');
% option = V2_GetDefaultConfig('RochesterADL');
% 
% this script implements the pipeline of bag of features for action
% recognition in RochesterADL
% subjects = 1:5; % in the implementation, we perform leave-one-subject-out.
% eval_res = {};
% stip_data_S = {};
% stip_data_T = {};
% weightCandidates = 0:10;
% 
% 
% for ss = subjects
%     fprintf('- leave subject %i out...\n',ss);
%     disp('-- loading data...\n');
%     load(sprintf('stip_data_%i.mat',ss));% load pre-stored data
%     
%     disp('-- learning visual word...');
%     learn visual word
%     [codebook_stip,SUMD_stip,opts_stip,running_info_stip,mu_stip,sigma_stip]...
%         = V2_CreateCodebook(stip_data_S{ss},option,'stip');
%     eval_res{ss}.CodebookLearning.codebook_stip = codebook_stip;
%     eval_res{ss}.CodebookLearning.SUMD_stip = SUMD_stip;
%     eval_res{ss}.CodebookLearning.opts_stip = opts_stip;
%     eval_res{ss}.CodebookLearning.running_info_stip = running_info_stip;
%     eval_res{ss}.RawFeatureStandardization.mu_stip = mu_stip;
%     eval_res{ss}.RawFeatureStandardization.sigma_stip = sigma_stip;
% 
%     [codebook_hra,SUMD_hra,opts_hra,running_info_hra,mu_hra,sigma_hra]...
%         = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.HeadRightArm');
%     eval_res{ss}.CodebookLearning.codebook_hra = codebook_hra;
%     eval_res{ss}.CodebookLearning.SUMD_hra = SUMD_hra;
%     eval_res{ss}.CodebookLearning.opts_hra = opts_hra;
%     eval_res{ss}.CodebookLearning.running_info_hra = running_info_hra;
%     eval_res{ss}.RawFeatureStandardization.mu_hra = mu_hra;
%     eval_res{ss}.RawFeatureStandardization.sigma_hra = sigma_hra;
% 
%     [codebook_hla,SUMD_hla,opts_hla,running_info_hla,mu_hla,sigma_hla]...
%         = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.HeadLeftArm');
%     eval_res{ss}.CodebookLearning.codebook_hla = codebook_hla;
%     eval_res{ss}.CodebookLearning.SUMD_hla = SUMD_hla;
%     eval_res{ss}.CodebookLearning.opts_hla = opts_hla;
%     eval_res{ss}.CodebookLearning.running_info_hla = running_info_hla;
%     eval_res{ss}.RawFeatureStandardization.mu_hla = mu_hla;
%     eval_res{ss}.RawFeatureStandardization.sigma_hla = sigma_hla;
% 
% 
%     [codebook_trl,SUMD_trl,opts_trl,running_info_trl,mu_trl,sigma_trl]...
%         = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.TorsoRightLeg');
%     eval_res{ss}.CodebookLearning.codebook_trl = codebook_trl;
%     eval_res{ss}.CodebookLearning.SUMD_trl = SUMD_trl;
%     eval_res{ss}.CodebookLearning.opts_trl = opts_trl;
%     eval_res{ss}.CodebookLearning.running_info_trl = running_info_trl;
%     eval_res{ss}.RawFeatureStandardization.mu_trl = mu_trl;
%     eval_res{ss}.RawFeatureStandardization.sigma_trl = sigma_trl;
% 
% 
%     [codebook_tll,SUMD_tll,opts_tll,running_info_tll,mu_tll,sigma_tll]...
%         = V2_CreateCodebook(stip_data_S{ss},option,'skeleton.TorsoLeftLeg');
%     eval_res{ss}.CodebookLearning.codebook_tll = codebook_tll;
%     eval_res{ss}.CodebookLearning.SUMD_tll = SUMD_tll;
%     eval_res{ss}.CodebookLearning.opts_tll = opts_tll;
%     eval_res{ss}.CodebookLearning.running_info_tll = running_info_tll;
%     eval_res{ss}.RawFeatureStandardization.mu_tll = mu_tll;
%     eval_res{ss}.RawFeatureStandardization.sigma_tll = sigma_tll;
% 
%     disp('-- evaluation...');
%     for aa = weightCandidates
%         option.features.pose_stip_weight = aa/10;
%         fprintf('- feature encoding..\n');
%         stip_S_encoded = V2_LocalFeatureEncoding(stip_data_S{ss},eval_res{ss},option);
%         stip_T_encoded = V2_LocalFeatureEncoding(stip_data_T{ss},eval_res{ss},option);
%         fprintf('- feature encoding end.\n');
%         eval_res{ss}.stip_S_encoded{aa+1} = stip_S_encoded;
%         eval_res{ss}.stip_T_encoded{aa+1} = stip_T_encoded;
% 
% 
%         multi-class svm train for snippets
%         [model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
%            V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);
% 
%         eval_res{ss}.svm.model{aa+1} = model;
%         eval_res{ss}.svm.Yt{aa+1} = Yt;
%         eval_res{ss}.svm.Ytp{aa+1} = Ytp;
%         eval_res{ss}.svm.prob_estimates{aa+1} = prob_estimates;
%         eval_res{ss}.svm.meta_res{aa+1} = meta_res;        
%         clear stip_S_encoded stip_T_encoded
%     end
%     
%     clear stip_data_S stip_data_T;
% 
% end
% 
% save(option.fileIO.eval_res_file,'eval_res');
% save(option.fileIO.option_file,'option');


%% SenseEmotion3_Searching dataset
% clear all;
% clc;
% disp('=============== evaluation on SenseEmotion3_Searching dataset===========');
% option = V2_GetDefaultConfig('SenseEmotion3_Searching');
% 
% this script implements the pipeline of bag of features for action
% recognition in SenseEmotion3_Searching
% groups = 1:4; % in the implementation, we perform leave-one-group-out.
% eval_res = {};
% stip_data_S = {};
% stip_data_T = {};
% weightCandidates = 0:10;
% 
% 
% for ss = groups
%     fprintf('- leave group %i out...\n',ss);
%     disp('-- loading data...\n');
%     load(sprintf('option.fileIO.dataset_name_leave_%i.mat',ss));% load pre-stored data
%     
%     disp('-- learning visual word...');
%     learn visual word
%     [codebook_stip,SUMD_stip,opts_stip,running_info_stip,mu_stip,sigma_stip]...
%         = V2_CreateCodebook(stip_data_S,option,'stip');
%     eval_res{ss}.CodebookLearning.codebook_stip = codebook_stip;
%     eval_res{ss}.CodebookLearning.SUMD_stip = SUMD_stip;
%     eval_res{ss}.CodebookLearning.opts_stip = opts_stip;
%     eval_res{ss}.CodebookLearning.running_info_stip = running_info_stip;
%     eval_res{ss}.RawFeatureStandardization.mu_stip = mu_stip;
%     eval_res{ss}.RawFeatureStandardization.sigma_stip = sigma_stip;
% 
%     [codebook_hra,SUMD_hra,opts_hra,running_info_hra,mu_hra,sigma_hra]...
%         = V2_CreateCodebook(stip_data_S,option,'skeleton.HeadRightArm');
%     eval_res{ss}.CodebookLearning.codebook_hra = codebook_hra;
%     eval_res{ss}.CodebookLearning.SUMD_hra = SUMD_hra;
%     eval_res{ss}.CodebookLearning.opts_hra = opts_hra;
%     eval_res{ss}.CodebookLearning.running_info_hra = running_info_hra;
%     eval_res{ss}.RawFeatureStandardization.mu_hra = mu_hra;
%     eval_res{ss}.RawFeatureStandardization.sigma_hra = sigma_hra;
% 
%     [codebook_hla,SUMD_hla,opts_hla,running_info_hla,mu_hla,sigma_hla]...
%         = V2_CreateCodebook(stip_data_S,option,'skeleton.HeadLeftArm');
%     eval_res{ss}.CodebookLearning.codebook_hla = codebook_hla;
%     eval_res{ss}.CodebookLearning.SUMD_hla = SUMD_hla;
%     eval_res{ss}.CodebookLearning.opts_hla = opts_hla;
%     eval_res{ss}.CodebookLearning.running_info_hla = running_info_hla;
%     eval_res{ss}.RawFeatureStandardization.mu_hla = mu_hla;
%     eval_res{ss}.RawFeatureStandardization.sigma_hla = sigma_hla;
% 
% 
%     [codebook_trl,SUMD_trl,opts_trl,running_info_trl,mu_trl,sigma_trl]...
%         = V2_CreateCodebook(stip_data_S,option,'skeleton.TorsoRightLeg');
%     eval_res{ss}.CodebookLearning.codebook_trl = codebook_trl;
%     eval_res{ss}.CodebookLearning.SUMD_trl = SUMD_trl;
%     eval_res{ss}.CodebookLearning.opts_trl = opts_trl;
%     eval_res{ss}.CodebookLearning.running_info_trl = running_info_trl;
%     eval_res{ss}.RawFeatureStandardization.mu_trl = mu_trl;
%     eval_res{ss}.RawFeatureStandardization.sigma_trl = sigma_trl;
% 
% 
%     [codebook_tll,SUMD_tll,opts_tll,running_info_tll,mu_tll,sigma_tll]...
%         = V2_CreateCodebook(stip_data_S,option,'skeleton.TorsoLeftLeg');
%     eval_res{ss}.CodebookLearning.codebook_tll = codebook_tll;
%     eval_res{ss}.CodebookLearning.SUMD_tll = SUMD_tll;
%     eval_res{ss}.CodebookLearning.opts_tll = opts_tll;
%     eval_res{ss}.CodebookLearning.running_info_tll = running_info_tll;
%     eval_res{ss}.RawFeatureStandardization.mu_tll = mu_tll;
%     eval_res{ss}.RawFeatureStandardization.sigma_tll = sigma_tll;
% 
%     disp('-- evaluation...');
%     for aa = weightCandidates
%         option.features.pose_stip_weight = aa/10;
%         fprintf('- feature encoding..\n');
%         stip_S_encoded = V2_LocalFeatureEncoding(stip_data_S,eval_res{ss},option);
%         stip_T_encoded = V2_LocalFeatureEncoding(stip_data_T,eval_res{ss},option);
%         fprintf('- feature encoding end.\n');
%         eval_res{ss}.stip_S_encoded{aa+1} = stip_S_encoded;
%         eval_res{ss}.stip_T_encoded{aa+1} = stip_T_encoded;
% 
% 
%         multi-class svm train for snippets
%         [model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
%            V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);
% 
%         eval_res{ss}.svm.model{aa+1} = model;
%         eval_res{ss}.svm.Yt{aa+1} = Yt;
%         eval_res{ss}.svm.Ytp{aa+1} = Ytp;
%         eval_res{ss}.svm.prob_estimates{aa+1} = prob_estimates;
%         eval_res{ss}.svm.meta_res{aa+1} = meta_res;        
%         clear stip_S_encoded stip_T_encoded
%     end
%     
%     clear stip_data_S stip_data_T;
% 
% end
% 
% save(option.fileIO.eval_res_file,'eval_res');
% save(option.fileIO.option_file,'option');
    

