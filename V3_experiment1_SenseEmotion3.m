%% influence of the pose-stip combination. 
%%% SenseEmotion3_Searching dataset, no pose feature.

clear all;
clc;
addpath(genpath('SVM-chi-square-master'));
addpath(genpath('fcl-master/matlab/kmeans'));
addpath(genpath('vlfeat-0.9.20/toolbox'));
run('vlfeat-0.9.20/toolbox/vl_setup')



disp('=============== evaluation on SenseEmotion3_Searching dataset===========');
option = V3_GetDefaultConfig('SenseEmotion3_Searching');
feature_data = V3_ReadingData('SenseEmotion3_Searching',option);

% this script implements the pipeline of bag of features for action
% recognition in SenseEmotion3_Searching
groups = 1:4; % in the implementation, we perform leave-one-group-out.
eval_res = {};
scales = option.features.stip.scales; % can be automatically determined
feature_type_list = option.features.stip.feature_type_list;


for gg = groups
    stip_data_S = {};
    stip_data_T = {};

    fprintf('- leave group %i out...\n',gg);
    groups_training = groups(groups~=gg);
    for ii = groups_training
        stip_data_S = [stip_data_S feature_data{ii}];
    end
    stip_data_T = feature_data{gg};
    
    

    for ss = 1:length(scales)
        %learn GMM for each individual scale
        fprintf('-- learning visual word for scale %f...\n', scales(ss));
        [gmm_means, gmm_covar, gmm_prior,mu,sigma]...
            = V3_CreateCodebook(stip_data_S,option,scales(ss));
        eval_res{gg}.CodebookLearning{ss}.scale = scales(ss);
        eval_res{gg}.CodebookLearning{ss}.gmm_means = gmm_means;
        eval_res{gg}.CodebookLearning{ss}.gmm_covar = gmm_covar;
        eval_res{gg}.CodebookLearning{ss}.gmm_prior = gmm_prior;
        eval_res{gg}.CodebookLearning{ss}.RawFeatureStandardization.mu = mu;
        eval_res{gg}.CodebookLearning{ss}.RawFeatureStandardization.sigma = sigma;
    end
    
    
    disp('-- evaluation...');
    fprintf('- feature encoding..\n');
    stip_S_encoded = V3_LocalFeatureEncoding(stip_data_S,eval_res{gg},option);
    stip_T_encoded = V3_LocalFeatureEncoding(stip_data_T,eval_res{gg},option);
    fprintf('- feature encoding end.\n');
    eval_res{gg}.stip_S_encoded = stip_S_encoded;
    eval_res{gg}.stip_T_encoded = stip_T_encoded;


%         multi-class svm train for snippets
    [model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
       V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);

    eval_res{gg}.svm.model = model;
    eval_res{gg}.svm.Yt = Yt;
    eval_res{gg}.svm.Ytp = Ytp;
    eval_res{gg}.svm.prob_estimates = prob_estimates;
    eval_res{gg}.svm.meta_res = meta_res;        
    clear stip_S_encoded stip_T_encoded;  
    clear stip_data_S stip_data_T;

end

save(option.fileIO.eval_res_file,'eval_res');
save(option.fileIO.option_file,'option');
    

