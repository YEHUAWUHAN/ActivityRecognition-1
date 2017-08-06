function option = V2_GetDefaultConfig(dataset)
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
    otherwise
        error('no other dataset considered so far.');
        return;
end
option.features.stip.including_scale = 1;
option.features.time_window = 30; % successive frames wrt.subsampling
option.features.stride = 30; % sliding time window by this stride
option.features.pose_stip_weight = 1; % feature =( w*posefeature, (1-w)*stipfeature)
option.features.standardization = 1; % 1 for kmeans
option.voting.excluding_empty_detection = true;

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
        split_path = [dataset_path '/testTrainMulti_7030_splits'];
        aa = dir(dataset_path);
        act_list = arrayfun( @(x) x.name, aa(3:end),'UniformOutput',false );
        act_list(strcmp(act_list,'testTrainMulti_7030_splits'))=[];
        act_list(strcmp(act_list,'uncompress.sh'))=[];
        option.fileIO.stip_file_version = '1.0'; 
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
option.codebook.NC_stip = 4000; %%% number of clusters to obtain 4000-Kmeans
option.codebook.NC_pose = 50; %%% number of clusters to obtain 50-Kmeans
option.codebook.encoding_method = 'hard_voting'; %%% 'VLAD', 'Fisher','hard_voting'
option.codebook.visualize = 0;

option.codebook.type = 'Kmeans'; %%% Fisher vector 
if strcmp(option.codebook.encoding_method, 'Fisher')
    option.codebook.type = 'GMM'; %%% Fisher vector 
end
%% svm classification
option.svm.kernel = 'linear'; %%% svm kernel, can be linear, RBF or chi-square
option.svm.n_fold_cv = 5; %%% n-fode cross-validation for parameter selection
option.svm.useWeight = true;



