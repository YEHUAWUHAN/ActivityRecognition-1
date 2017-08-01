function [codebook,SUMD,opts,running_info,mu,sigma] = V2_CreateCodebook(stip_data,option,flag)
%%% This function creates the vocabulary for LEAVE-ONE-SUBJECT-OUT, the leaved subject is passed in the argument.
%%% This function returns a matrix, whose rows denote vocabularies and columns denote features.
%%% V1.0.1 In this version, the dataset and subject arguments are only for
%%% log files.



switch flag
    case 'stip' 

        NN = length(stip_data);
        features = [];

        if option.features.stip.including_scale
            dd = 8;
        else
            dd = 10;
        end

        for ii = 1:NN
            features = [features;stip_data{ii}.features(:,dd:end)];
        end
        NC = option.codebook.NC_stip;
    case 'skeleton.HeadRightArm'
        NN = length(stip_data);
        features = [];
        
        % excluding incorrect pose estimate
        for ii = 1:NN
            ff = stip_data{ii}.skeleton.HeadRightArm;
            iidx = find(sum(ff,2)==0 | sum(ff,2)> 10e3);
            ff(iidx,:) = [];
            features = [features;ff];
        end
        NC = option.codebook.NC_pose;
    case 'skeleton.HeadLeftArm'
        NN = length(stip_data);
        features = [];

        for ii = 1:NN
            ff = stip_data{ii}.skeleton.HeadLeftArm;
            %%% filter out incorrect estimation. Check each row in ff
            %%% if sum(ff(i,:))==0 or sum(ff(i,:))> 10e6, excluding it
            iidx = find(sum(ff,2)==0 | sum(ff,2)> 10e3);
            ff(iidx,:) = [];
            features = [features;ff];
            
        end
        NC = option.codebook.NC_pose;
    case 'skeleton.TorsoRightLeg'
        if ~option.features.skeleton.fullbody
            fprintf('-fullbody feature is disabled');
            codebook=[];
            SUMD=[];
            opts=[];
            running_info=[];
            mu=[];
            sigma=[];
            return;
        end
        
        NN = length(stip_data);
        features = [];

        for ii = 1:NN
            ff = stip_data{ii}.skeleton.TorsoRightLeg;
            iidx = find(sum(ff,2)==0 | sum(ff,2)> 10e3);
            ff(iidx,:) = [];
            features = [features;ff];
        end
        NC = option.codebook.NC_pose;
    case 'skeleton.TorsoLeftLeg'
        if ~option.features.skeleton.fullbody
            fprintf('-fullbody feature is disabled');
            codebook=[];
            SUMD=[];
            opts=[];
            running_info=[];
            mu=[];
            sigma=[];
            return;
        end
        
        NN = length(stip_data);
        features = [];

        for ii = 1:NN
            ff = stip_data{ii}.skeleton.TorsoLeftLeg;
            iidx = find(sum(ff,2)==0 | sum(ff,2)> 10e3);
            ff(iidx,:) = [];
            features = [features;ff];
        end
        NC = option.codebook.NC_pose;
    otherwise
        fprintf('-other features are not implemented.');
end
        
        
fprintf('-- #features=%i, feature_length=%i\n',size(features,1),size(features,2));

%%% Due to complexity, we randomly choose N features, if the #features > N.
N = option.codebook.maxsamples; % in previous settings, we set N = 200000 hdmb51, we wet 100000
rng default;
if size(features,1)>N
   kk = randperm(size(features,1));
   features = features(kk(1:N),:);
end

%%% data standardization
if option.features.standardization
    mu = mean(features,1);
    sigma = std(features,1);
else
    mu = zeros(1,size(features,2));
    sigma = ones(1,size(features,2));
end
features = (features-repmat(mu,size(features,1),1))./repmat(sigma+1e-6,size(features,1),1);

if option.codebook.visualize
    figure(2);imagesc(features);drawnow;pause;
end



if strcmp(option.codebook.type, 'Kmeans')

    %%% kmeans clustering with kmeans++ initialization and optimized algorithm.
    %%% here we use the fcl lib for fast clustering. Initialization is kmeans++, and we dont run several times. 
    opts.seed = 0;                  % change starting position of clustering
    opts.algorithm = 'kmeans_optimized';     % change the algorithm to 'kmeans_optimized'
    opts.init = 'kmeans++';           % use kmeans++ as initialization
    opts.no_cores = 7;              % number of cores to use. for scientific experiments always use 1! -1 means using all
    opts.max_iter = 100;             % stop after 100 iterations
    opts.tol = 1e-5;                % change the tolerance to converge quicker
    opts.silent = true;             % do not output anything while clustering
    opts.remove_empty = true;       % remove empty clusters from resulting cluster center matrix
    opts.additional_params.bv_annz = 0.125;
    fprintf('-- clustering (optimized kmeans)....\n');
    [ IDX, codebook,SUMD,running_info ] = fcl_kmeans(sparse(features'), NC, opts);

elseif strcmp(option.codebook.type, 'GMM')
    %%todo
    [codebook.means, codebook.covariance, codebook.priors] = ...
        vl_gmm(features', NC);
        
else
    fprintf('ERROR: select codebook type Kmeans or GMM!\n');
end
    
clear features stip_data


end
