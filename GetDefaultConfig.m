function option = GetDefaultConfig(dataset)

%% files io
%%% specify the dataset path
timer = datestr(fix(clock),'yyyy-mm-dd-HH-MM');
option.fileIO.time = timer;
option.fileIO.dataset_path = sprintf('~/Videos/Dataset_%s',dataset);
option.fileIO.dataset_name = dataset;

%%% read the action_list
switch dataset
    case 'RochesterADL'
        act_list = importdata('annotation/Dataset_RochesterADL/activity_list.txt');
    case 'Weizmann'
        act_list = {'bend','jack','jump','pjump','run','side','skip','walk','wave1','wave2'};
    case 'KTH'
        act_list = {'boxing','handclapping','handwaving','jogging','running','walking'};
    case 'HMDB51'
        data_path = option.fileIO.dataset_path;
        split_path = [dataset_path '/testTrainMulti_7030_splits'];
        aa = dir(data_path);
        act_list = arrayfun( @(x) x.name, aa(3:end),'UniformOutput',false );
        act_list(strcmp(act_list,'testTrainMulti_7030_splits'))=[];
        act_list(strcmp(act_list,'uncompress.sh'))=[];
    otherwise
        error('no other datasets so far...');
end
option.fileIO.act_list = act_list;
%%% normally it is 2.0. But HMBD51 used 1.0 version.
option.fileIO.stip_file_version = '2.0'; 

%%% the file to store all results: codebook,svm and eval results.
option.fileIO.eval_res_file = sprintf('%s_EvaluationResults_%s.mat',dataset,timer);
option.fileIO.option_file = sprintf('%s_option_%s.mat',dataset,timer);
%% low-level stip features. Here we assume that you already
%%% the local feature vector does not include scales
option.stip_features.including_scale = false; 
option.stip_features.standardization = true;
%% codebook generation
option.codebook.maxsamples = 100000; %%% uplimit of samples for clustering.
option.codebook.NC = 4000; %%% number of clusters to obtain
option.codebook.encoding_method = 'hard_voting';
%% svm classification
option.svm.kernel = 'linear'; %%% svm kernel, can be linear, RBF or Chi-Square
option.svm.n_fold_cv = 5; %%% n-fode cross-validation for parameter selection

