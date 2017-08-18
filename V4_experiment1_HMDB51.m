%%% script for Fisher vector encoding and mini-batch training of the
%%% bilinear max-margin classifier
clear all;
clc;
addpath(genpath('SVM-chi-square-master'));
addpath(genpath('fcl-master/matlab/kmeans'));
addpath(genpath('vlfeat-0.9.20/toolbox'));
run('vlfeat-0.9.20/toolbox/vl_setup')

%%
disp('=============== evaluation on HMDB dataset===========');
option = V3_GetDefaultConfig('HMDB51');

%%% split to three train/test groups, final result would be their average.
groups = 1:3; % in the implementation, we perform leave-one-group-out.
eval_res = {};
scales = option.features.stip.scales; % can be automatically determined
feature_type_list = option.features.stip.feature_type_list;
split_path = option.fileIO.split_path;
act_list = option.fileIO.act_list;
for gg = groups
    
    %%% train multi-scale GMM
    stip_data_S = V4_ReadingData(option,0);
    
%     for aa = 1:length(act_list)
%         split_list = importdata([split_path,'/', ...
%             act_list{aa},'_test_split',num2str(aa),'.txt']);
%         for vv = 1:length(split_list.data)
%             video = split_list.textdata{vv};
%             idx = find(strcmp(videos_in_feature_data,video));
%             flag = split_list.data(vv);
%             if flag == 1
%                 stip_data_S = [stip_data_S feature_data{idx}];
%             elseif flag == 2
%                 stip_data_T = [stip_data_T feature_data{idx}];
%             else 
%                 continue;
%             end
%         end
%     end
%     
    
    for ss = 1:length(scales)
        %learn GMM for each individual scale
        fprintf('-- learning visual word for scale %f...\n', scales(ss));
        [gmm_means, gmm_covar, gmm_prior,mu,D,U]...
            = V3_CreateCodebook(stip_data_S,option,scales(ss));
        eval_res{gg}.CodebookLearning{ss}.scale = scales(ss);
        eval_res{gg}.CodebookLearning{ss}.gmm_means = gmm_means;
        eval_res{gg}.CodebookLearning{ss}.gmm_covar = gmm_covar;
        eval_res{gg}.CodebookLearning{ss}.gmm_prior = gmm_prior;
        eval_res{gg}.CodebookLearning{ss}.RawFeatureWhitening.mu = mu;
        eval_res{gg}.CodebookLearning{ss}.RawFeatureWhitening.D = D;
        eval_res{gg}.CodebookLearning{ss}.RawFeatureWhitening.U = U;
    end
    stip_data_S = {};
    
    disp('-- Evaluation.(training in a mini-batch scheme)..');
    idx_mini_batch = 1;
    option.fileIO.readingmode = 'mini_batch';
    option.fileIO.batch_size_per_class = 2;
    while(1)
        fprintf('-- used training samples per class: %d\n',idx_min_batch);
        stip_data_S = V4_ReadingData(option,idx_mini_batch);
        if isempty(stip_data_S)
            disp('-- Training finishes.');
            break;
        end
        fprintf('--- feature encoding..\n');
        stip_S_encoded = V3_LocalFeatureEncoding(stip_data_S,eval_res{gg},option);
%         stip_T_encoded = V3_LocalFeatureEncoding(stip_data_T,eval_res{gg},option);
        
%         eval_res{gg}.stip_S_encoded = stip_S_encoded;
%         eval_res{gg}.stip_T_encoded = stip_T_encoded;


    %         multi-class svm train for snippets
        [model,Yt,Ytp,prob_estimates,meta_res,test_labels]=...
           V2_ActivityRecognition(stip_S_encoded,stip_T_encoded,option);
       
        clear stip_S_encoded stip_T_encoded;  
        clear stip_data_S stip_data_T;
        idx_mini_batch = idx_mini_batch + option.fileIO.batch_size_per_class;
    end
    
    eval_res{gg}.svm.model = model;
    eval_res{gg}.svm.Yt = Yt;
    eval_res{gg}.svm.Ytp = Ytp;
    eval_res{gg}.svm.prob_estimates = prob_estimates;
    eval_res{gg}.svm.meta_res = meta_res; 
end

save(option.fileIO.eval_res_file,'eval_res');
save(option.fileIO.option_file,'option');
    

