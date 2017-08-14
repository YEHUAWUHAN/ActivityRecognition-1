function option = V3_GetDefaultConfig(dataset)
%% raw features
switch dataset
    case 'KTH'
        option.features.skeleton.fullbody = true;
        option.features.stip.type = 'STIP';
    case 'RochesterADL'
        option.features.stip.type = 'STIP';
        option.features.skeleton.fullbody = false;
    case 'SenseEmotion3_Searching'
        option.features.skeleton.fullbody = false;
        option.features.stip.type = 'iDT';
        option.features.stip.scales = 2.^(0:0.5:2.5); % can be automatically determined
        option.features.stip.feature_type_list = {'var','mbh'}; % var,traj,hog,hof,mbh
    case 'HMDB51'
        option.features.stip.type = 'iDT';
        option.features.stip.scales = 2.^(0:0.5:2.5); % can be automatically determined
        option.features.stip.feature_type_list = {'var','mbh'}; % var,traj,hog,hof,mbh
    otherwise
        error('no other dataset considered so far.');
        return;
end
option.features.whitening_dim = 0.5; % if >1, =target_dim; else ratio of dimension to preserve.
option.features.time_window = 30; % successive frames wrt.subsampling
option.features.stride = 30; % sliding time window by this stride
option.features.whitening = 0; % 1 for kmeans
% option.voting.excluding_empty_detection = true;

% 'batch':one feature per video; 
% 'accumulated':features accumulated over each snippets
% 'snippets': features are from each individual snippets
option.features.type = 'batch';
% option.features.type = 'accumulated';

% %% hyperfeatures architecture
% % option.hyperfeatures.use_trained_codebook = 1;
% option.hyperfeatures.encoding_W = 100; % receptive field window
% option.hyperfeatures.encoding_S = 10; % stride
% option.hyperfeatures.scaling = 1;  % In next layer, nc = nc/rfscaling
% option.hyperfeatures.num_layers = 1; % number of layer
% option.hyperfeatures.multilayerfeature = true; % combining global features from all layers
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
        option.fileIO.stip_file_version = '2.0'; 
    case 'Weizmann'
        act_list = {'bend','jack','jump','pjump','run','side','skip','walk','wave1','wave2'};
        option.fileIO.subject_list = {'daria','denis','eli','ido','ira','lena','lyova','moshe','shahar'};
        option.fileIO.stip_file_version = '2.0'; 
    case 'KTH'
        act_list = {'boxing','handclapping','handwaving','jogging','running','walking'};
        option.fileIO.stip_file_version = '2.0'; 
    case 'SenseEmotion3_Searching'
        act_list = {'regular', 'irregular1', 'irregular2'};
        option.fileIO.stip_file_version = '2.0'; 
    case 'HMDB51'
        dataset_path = option.fileIO.dataset_path;
        option.fileIO.split_path = [dataset_path '/testTrainMulti_7030_splits'];
        aa = dir(dataset_path);
        act_list = arrayfun( @(x) x.name, aa(3:end),'UniformOutput',false );
        act_list(strcmp(act_list,'testTrainMulti_7030_splits'))=[];
        act_list(strcmp(act_list,'DT_features'))=[];
    otherwise
        error('no other datasets so far...');
end
option.fileIO.act_list = act_list;
%%% normally it is 2.0. But HMBD51 used 1.0 version.

%%% the file to store all results: codebook,svm and eval results.
option.fileIO.eval_res_file = sprintf('%s_EvaluationResults_%s.mat',dataset,timer);
option.fileIO.option_file = sprintf('%s_option_%s.mat',dataset,timer);

%% codebook generation
option.codebook.maxsamples = 100000; %%% uplimit of samples for clustering.
option.codebook.NC_gmm = 64; %%% number of clusters to obtain 4000-Kmeans
option.codebook.encoding_method = 'Fisher'; %%% 'VLAD', 'Fisher','hard_voting'
option.codebook.visualize = 0;

option.codebook.type = 'Kmeans'; %%% Fisher vector 
if strcmp(option.codebook.encoding_method, 'Fisher')
    option.codebook.type = 'GMM'; %%% Fisher vector 
end
%% svm classification
option.svm.kernel = 'linear'; %%% svm kernel, can be linear, RBF or chi-square
option.svm.n_fold_cv = 5; %%% n-fode cross-validation for parameter selection
option.svm.useWeight = true;
option.svm.pca_whitening_target_dim = 400;

