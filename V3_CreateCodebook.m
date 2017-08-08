function [gmm_means, gmm_covar, gmm_priors,mu,sigma] = V3_CreateCodebook(stip_data,option,scale)
%%% This function creates the vocabulary for LEAVE-ONE-SUBJECT-OUT, the leaved subject is passed in the argument.
%%% This function returns a matrix, whose rows denote vocabularies and columns denote features.
%%% V1.0.1 In this version, the dataset and subject arguments are only for
%%% log files.

n_videos = length(stip_data);
   
%%% select features depending on type
features = [];
idx_range_feature = [];
feature_type_list = option.features.stip.feature_type_list;
for tt = 1:length(feature_type_list)
    type = feature_type_list{tt};
    idx_range_feature = [idx_range_feature feature_type_to_index(type)];
end

%%% select samples depending on scale
for n = 1:n_videos
    scales_one_video = stip_data{n}.feature(:,7);
    idx = (abs(scales_one_video-scale)<=10e-5);
    features = [features; stip_data{n}.feature(idx,idx_range_feature)];
end

%%% Due to complexity, we randomly choose N features, if the #features > N.
N = option.codebook.maxsamples; % in previous settings, we set N = 200000 hdmb51, we wet 100000
rng default;
if size(features,1)>N
   kk = randperm(size(features,1));
   features = features(kk(1:N),:);
end

fprintf('-- n_samples=%d..\n',size(features,1));
%%% data standardization
if option.features.standardization
    mu = mean(features,1);
    sigma = std(features,1);
else
    mu = zeros(1,size(features,2));
    sigma = ones(1,size(features,2));
end
features = (features-repmat(mu,size(features,1),1))...
    ./repmat(sigma+1e-6,size(features,1),1);

%%% learn GMM
features = features';
NC = option.codebook.NC_gmm;
%%% initialize using kmeans
fprintf('-- clustering (Gaussian Mixture Model)....\n');
fprintf('--- clustering (Gaussian Mixture Model) init....\n');

opts.seed = 0;                  % change starting position of clustering
opts.algorithm = 'kmeans_optimized';     % change the algorithm to 'kmeans_optimized'
opts.init = 'kmeans++';           % use kmeans++ as initialization
opts.no_cores = 7;              % number of cores to use. for scientific experiments always use 1! -1 means using all
opts.max_iter = 3;             % stop after 100 iterations
opts.tol = 1e-5;                % change the tolerance to converge quicker
opts.silent = true;             % do not output anything while clustering
opts.remove_empty = true;       % remove empty clusters from resulting cluster center matrix
opts.additional_params.bv_annz = 0.125;
[ IDX, init_mean,SUMD,running_info ] = fcl_kmeans(sparse(features), NC, opts);
init_cov = zeros(size(features,1), NC);
init_prior = zeros(1,NC);
IDX = IDX+1;
for i=1:NC
    data_k = features(:,IDX==i);
    init_prior(i) = size(data_k,2) / NC;

    if size(data_k,1) == 0 || size(data_k,2) == 0
        init_cov(:,i) = diag(cov(features'));
    else
        init_cov(:,i) = diag(cov(data_k'));
    end
end


fprintf('--- clustering (Gaussian Mixture Model) EM....\n');
[gmm_means, gmm_covar, gmm_priors] = ...
    vl_gmm(features, NC, 'initialization','custom',...
    'InitMeans',full(init_mean),'InitCovariances',init_cov,'InitPriors',init_prior);
features = [];

end



function idx_range = feature_type_to_index(type)
switch type
    case 'var'
        idx_range = 4:5;
    case 'traj'
        idx_range = 11:40;
    case 'hog'
        idx_range = 41:136;
    case 'hof'
        idx_range = 137:244;
    case 'mbh'
        idx_range = 245:436;
    otherwise
        fprintf('-other features are not implemented.');
end
end
        


