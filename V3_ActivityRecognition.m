function [model_new] = structure_bilinearSVM_train(...
    stip_data_train,option,model_old)

%%% train a bilinear max-margin classifier using alternating optimization
%%% scheme. The training is running in mini-batch mode due to large scale.
act_list = option.fileIO.act_list;

Xs = [];
Ys = [];
NN = length(stip_data_train);

for ii = 1:NN  
    n_snippets = size(stip_data_train{ii}.feature,1);
    
    idx = 1:n_snippets;
    switch option.fileIO.dataset_name        
        case 'HMDB51'
            label = find(strcmp(act_list, stip_data_train{ii}.video));
            Ys = [Ys;label*ones(n_snippets,1)];
        case 'UCF101'
            label = find(strcmp(act_list, stip_data_train{ii}.video));
            Ys = [Ys;label*ones(n_snippets,1)];
        otherwise
            error('no othe option.');
            return;
    end
    Xs = [Xs;stip_data_train{ii}.feature(idx,:)];
    stip_data_train{ii} = [];
end



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











