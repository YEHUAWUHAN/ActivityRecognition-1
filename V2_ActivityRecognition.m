function [model,Yt,cls,prob_estimates,meta_res,test_labels] = V2_ActivityRecognition(...
    stip_data_train,stip_data_test,option)

act_list = option.fileIO.act_list;

Xs = [];
Ys = [];
Xt = [];
Yt = [];
NN = length(stip_data_train);
NNt = length(stip_data_test);
test_labels = {};

for ii = 1:NN  
    n_snippets = size(stip_data_train{ii}.feature,1);
    %vis = stip_data_train{ii}.visible;
    %if option.voting.excluding_empty_detection
    %    idx = find(vis==1);
    %    n_snippets = length(idx);
    %else
    %    idx = 1:n_snippets;
    %end
    idx = 1:n_snippets;
    switch option.fileIO.dataset_name
        case 'RochesterADL'
            label = find(strcmp(act_list, stip_data_train{ii}.video(1:end-4)));
            Ys = [Ys;label*ones(n_snippets,1)];
        case 'KTH'
            ss = strsplit(stip_data_train{ii}.video,'_');
            label = find(strcmp(act_list, ss{2}));
            Ys = [Ys;label*ones(n_snippets,1)];
        case 'Weizmann'
            ss = strsplit(stip_data_train{ii}.video,'_');
            label = find(strcmp(act_list, ss{2}));
            Ys = [Ys;label*ones(n_snippets,1)];
        case 'HMDB51'
            label = find(strcmp(act_list, stip_data_train{ii}.video));
            Ys = [Ys;label*ones(n_snippets,1)];
        case 'SenseEmotion3_Searching'
            Ys = [Ys;stip_data_train{ii}.label*ones(n_snippets,1)];
        otherwise
            error('no othe option.');
            return;
    end
    Xs = [Xs;stip_data_train{ii}.feature(idx,:)];
%     if ~option.hyperfeatures.multilayerfeature
%         Xs(ii,:) = Encoding(stip_data_train{ii}.features(:,dd:end),codebook,...
%             mu,sigma,option);
%     else
%         last = Encoding(stip_data_train{ii}.features(:,dd:end),codebook,...
%             mu,sigma,option);
%         for ll = 1:option.hyperfeatures.num_layers-1
%             last = [last stip_data_train{ii}.globalfeatures{ll}];
%         end
%         
%         Xs(ii,:) = last ./ (sum(last)); % l1 -normalizing
%     end
    stip_data_train{ii} = [];
end


for ii = 1:NNt
    n_snippets = size(stip_data_test{ii}.feature,1);
    
    %vis = stip_data_test{ii}.visible;
    %if option.voting.excluding_empty_detection
    %    idx = find(vis==1);
    %    n_snippets = length(idx);
    %else
    %    idx = 1:n_snippets;
    %end
    idx = 1:n_snippets;
    switch option.fileIO.dataset_name
        case 'RochesterADL'
            label = find(strcmp(act_list, stip_data_test{ii}.video(1:end-4)));
            Yt = [Yt;label*ones(n_snippets,1)];
        case 'KTH'
            ss = strsplit(stip_data_test{ii}.video,'_');
            label = find(strcmp(act_list, ss{2}));
            Yt = [Yt;label*ones(n_snippets,1)];
        case 'Weizmann'
            ss = strsplit(stip_data_test{ii}.video,'_');
            label = find(strcmp(act_list, ss{2}));
            Yt = [Yt;label*ones(n_snippets,1)];
        case 'HMDB51'
            label = find(strcmp(act_list, stip_data_test{ii}.video));
            Yt = [Yt;label*ones(n_snippets,1)];
        case 'SenseEmotion3_Searching'
            label = stip_data_test{ii}.label;
            Yt = [Yt;label*ones(n_snippets,1)];

        otherwise
            error('no othe option.');
            return;
    end
    Xt = [Xt;stip_data_test{ii}.feature(idx,:)];
%     if ~option.hyperfeatures.multilayerfeature
%         Xs(ii,:) = Encoding(stip_data_train{ii}.features(:,dd:end),codebook,...
%             mu,sigma,option);
%     else
%         last = Encoding(stip_data_train{ii}.features(:,dd:end),codebook,...
%             mu,sigma,option);
%         for ll = 1:option.hyperfeatures.num_layers-1
%             last = [last stip_data_train{ii}.globalfeatures{ll}];
%         end
%         
%         Xs(ii,:) = last ./ (sum(last)); % l1 -normalizing
%     end
    test_labels{ii}.label = stip_data_test{ii}.label;
    test_labels{ii}.video = stip_data_test{ii}.video;
    test_labels{ii}.n_snippets = n_snippets;
    stip_data_test{ii} = [];

end


%%% re-arrange the labels so as to avoid relabeling issue in libsvm
[Ys,idx_sort] = sort(Ys, 'descend');
Xs = Xs(idx_sort,:);

[Yt,idx_sort] = sort(Yt, 'descend');
Xt = Xt(idx_sort,:);


%%% feature PCA dimension reduction
target_dim = option.svm.pca_whitening_target_dim;
[Xs,Xt] = V3_pca_dr(Xs,Xt, target_dim,'reduce');

%%% classification
fprintf('-- training and testing \n');
[model,prob_estimates,cls,meta_res] = TrainSVM(Xs,Ys,Xt,Yt,option);
end


function [bestmodel,prob_estimates,class_label,meta_res] = TrainSVM(Xs,Ys,Xt,Yt,option)
%%% X is a struct containing feature vectors in a context hierarchy.
%%% Y is the action label.

% opt.nFold = option.svm.n_fold_cv; %%% rochester and weizmann, 5-fold cv; KTH 10-foldsave
% opt.kernel = option.svm.kernel;
% opt.useWeight = option.svm.useWeight;
% % Xtrain = CCV_normalize(Xs,1);
% % Xtest = CCV_normalize(Xt,1);
% [acc, prob_estimates, bestmodel, class_label, meta_res]...
%         =Fu_direct_SVM2(Xs, Xt, Ys,Yt,opt);



%%% In the following, we implement our own multi-class classification
%%% problem, so as to avoid relabeling problem within svm. Here we applied
%%% the embedded functions in Matlab.

svm_template = templateSVM('KernelFunction', option.svm.kernel);
rng default;
Mdl = fitcecoc(Xs,Ys,'Learners',svm_template, 'OptimizeHyperparameters', 'auto',...
    'HyperparameterOptimizationOptions',...
    struct('Kfold',option.svm.n_fold_cv, 'ShowPlots',false,'Verbose',0));
class_label = predict(Mdl,Xt);
bestmodel = Mdl;
prob_estimates = [];
meta_res = [];
acc_rate = sum(class_label == Yt)/length(Yt);
fprintf('-- recognition rate = %f\n',acc_rate);

end











