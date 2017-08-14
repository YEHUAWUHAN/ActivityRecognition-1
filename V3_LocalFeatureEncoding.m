function des = V3_LocalFeatureEncoding(stip_data,eval_res,option)

N = length(stip_data);
% stip_data = temporal_reorder(stip_data);
des = {};

for i = 1:N    
    des{i}.video = stip_data{i}.video;
    des{i}.feature =...
        feature_encoding_once(stip_data{i}.feature,eval_res,option);
    if strcmp(option.fileIO.dataset_name, 'SenseEmotion3_Searching')
        des{i}.label = stip_data{i}.label;
    end
end

end




function des = temporal_reorder(src)
NN = length(src);
for ii = 1:NN
    [~,idx] = sort(src{ii}.features(:,7));
    src{ii}.features = src{ii}.features(idx,:);
end
des = src;
clear src;
end



function des_feature = feature_encoding_once(src,eval_res,option)

idx_range_feature = [];
feature_type_list = option.features.stip.feature_type_list;
for tt = 1:length(feature_type_list)
    type = feature_type_list{tt};
    idx_range_feature = [idx_range_feature feature_type_to_index(type)];
end

src_feature = src(:,idx_range_feature);
src_ts = src(:,1);
src_scales = src(:,7);
W = option.features.time_window;
S = option.features.stride;
N = max(src_ts); % #frames
des_feature = [];
vis = [];



if strcmp(option.features.type,'batch')

    des_feature = feature_encoding_snippet(src_feature,src_scales,eval_res,option);

else
    for ii = 1:S:N-W   
        %%% encode pose features
        if strcmp(option.features.type,'accumulated')
            p = 1;
        else
            p = ii;
        end

        if ii+S > N-W
            des_feature_one = feature_encoding_snippet(src_feature,eval_res,option);
            des_feature = [des_feature; des_feature_one];
            
        else
            %%% encode stip features inbetween
            
            interval = find( src_ts>=(p-1) & src_ts<=(ii-1+W) );
            des_feature_one = feature_encoding_snippet(src_feature(interval,:),eval_res,option);
                
            des_feature = [des_feature; des_feature_one];
       
        end

    end    
end

%%% notice that the FV is normalized.

end




function des_feature = feature_encoding_snippet(src_feature,scale_list,eval_res,option)
des_feature = [];
scales = option.features.stip.scales; % can be automatically determined


for ss = 1:length(scales)
    src_onescale = src_feature(abs(scale_list-scales(ss))<=10e-6, :);
    if isempty(src_onescale)
        des_onescale = zeros(1,2*option.codebook.NC_gmm*size(src_feature,2));
    else
    %%% read out dictionaries   
        gmm_means = eval_res.CodebookLearning{ss}.gmm_means;
        gmm_covar = eval_res.CodebookLearning{ss}.gmm_covar;
        gmm_prior = eval_res.CodebookLearning{ss}.gmm_prior;
        mu = eval_res.CodebookLearning{ss}.RawFeatureStandardization.mu;
        D = eval_res.CodebookLearning{ss}.RawFeatureStandardization.D;
        U = eval_res.CodebookLearning{ss}.RawFeatureStandardization.U;
        
        %%% whitening
        src_onescale = (src_onescale-repmat(mu,size(src_onescale,1),1))...
            *U*D;

        %%% fisher vector encoding
        des_temp = vl_fisher(src_onescale',gmm_means, gmm_covar, gmm_prior, 'Improved');
        des_onescale = des_temp';
    end
    des_feature = [des_feature des_onescale];
end

%%% normalization l2
des_feature = des_feature/norm(des_feature,2);

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
