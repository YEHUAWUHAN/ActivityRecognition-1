function des = V2_LocalFeatureEncoding(stip_data,eval_res,option)

N = length(stip_data);
stip_data = temporal_reorder(stip_data);
des = {};

for i = 1:N    
    des{i}.video = stip_data{i}.video;
    [des{i}.feature, des{i}.visible] =...
        feature_encoding_once(stip_data{i},eval_res,option);
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



function [des,vis] = feature_encoding_once(src,eval_res,option)

if option.features.stip.including_scale
    dd = 8;
else
    dd = 10;
end

W = option.features.time_window;
S = option.features.stride;
N = size(src.skeleton.HeadLeftArm,1); % #frames
des = [];
vis = [];
alpha = option.features.pose_stip_weight;

%%% read out dictionaries 
cb_stip = eval_res.CodebookLearning.codebook_stip;
cb_hra = eval_res.CodebookLearning.codebook_hra;
cb_hla = eval_res.CodebookLearning.codebook_hla;

mu_stip = eval_res.RawFeatureStandardization.mu_stip;
mu_hra = eval_res.RawFeatureStandardization.mu_hra;
mu_hla = eval_res.RawFeatureStandardization.mu_hla;

sigma_stip = eval_res.RawFeatureStandardization.sigma_stip;
sigma_hra = eval_res.RawFeatureStandardization.sigma_hra;
sigma_hla = eval_res.RawFeatureStandardization.sigma_hla;

if option.features.skeleton.fullbody
    cb_trl = eval_res.CodebookLearning.codebook_trl;
    cb_tll = eval_res.CodebookLearning.codebook_tll;
    mu_trl = eval_res.RawFeatureStandardization.mu_trl;
    mu_tll = eval_res.RawFeatureStandardization.mu_tll;
    sigma_trl = eval_res.RawFeatureStandardization.sigma_trl;
    sigma_tll = eval_res.RawFeatureStandardization.sigma_tll;
end


if strcmp(option.features.type,'batch')

    dd_hra=feature_encoding_snippet(src.skeleton.HeadRightArm,...
        cb_hra,mu_hra,sigma_hra,option);
    dd_hla=feature_encoding_snippet(src.skeleton.HeadLeftArm,...
        cb_hla,mu_hla,sigma_hla,option);
    if option.features.skeleton.fullbody
        dd_trl=feature_encoding_snippet(src.skeleton.TorsoRightLeg,...
            cb_trl,mu_trl,sigma_trl,option);
        dd_tll=feature_encoding_snippet(src.skeleton.TorsoLeftLeg,...
            cb_tll,mu_tll,sigma_tll,option);
    else
        dd_trl = [];
        dd_tll = [];
    end
    
    %%% encode stip features inbetween
    
    dd_s = feature_encoding_snippet(src.features(:,dd:end),...
        cb_stip,mu_stip,sigma_stip,option);
    des = [des;[alpha*dd_hra alpha*dd_hla alpha*dd_trl alpha*dd_tll (1-alpha)*dd_s]];
    vis = [vis;1];

else
    
    for ii = 1:S:N-W   
        %%% encode pose features
        if strcmp(option.features.type,'accumulated')
            p = 1;
        else
            p = ii;
        end
        
        if ii+S > N-W
            dd_hra=feature_encoding_snippet(src.skeleton.HeadRightArm,...
                cb_hra,mu_hra,sigma_hra,option);
            dd_hla=feature_encoding_snippet(src.skeleton.HeadLeftArm,...
                cb_hla,mu_hla,sigma_hla,option);
            if option.features.skeleton.fullbody
                dd_trl=feature_encoding_snippet(src.skeleton.TorsoRightLeg,...
                    cb_trl,mu_trl,sigma_trl,option);
                dd_tll=feature_encoding_snippet(src.skeleton.TorsoLeftLeg,...
                    cb_tll,mu_tll,sigma_tll,option);
            else
                dd_trl = [];
                dd_tll = [];
            end

            %%% encode stip features inbetween

            dd_s = feature_encoding_snippet(src.features(:,dd:end),...
                cb_stip,mu_stip,sigma_stip,option);
            des = [des;[alpha*dd_hra alpha*dd_hla alpha*dd_trl alpha*dd_tll (1-alpha)*dd_s]];
            vis = [vis;1];
        else

            dd_hra=feature_encoding_snippet(src.skeleton.HeadRightArm(p:ii+W,:),...
                cb_hra,mu_hra,sigma_hra,option);
            dd_hla=feature_encoding_snippet(src.skeleton.HeadLeftArm(p:ii+W,:),...
                cb_hla,mu_hla,sigma_hla,option);
            if option.features.skeleton.fullbody
                dd_trl=feature_encoding_snippet(src.skeleton.TorsoRightLeg(p:ii+W,:),...
                    cb_trl,mu_trl,sigma_trl,option);
                dd_tll=feature_encoding_snippet(src.skeleton.TorsoLeftLeg(p:ii+W,:),...
                    cb_tll,mu_tll,sigma_tll,option);
            else
                dd_trl = [];
                dd_tll = [];
            end

            %%% encode stip features inbetween
            ts = src.features(:,7);
            interval = find( ts>=(p-1) & ts<=(ii-1+W) );
            dd_s = feature_encoding_snippet(src.features(interval,dd:end),...
                cb_stip,mu_stip,sigma_stip,option);
            des = [des;[alpha*dd_hra alpha*dd_hla alpha*dd_trl alpha*dd_tll (1-alpha)*dd_s]];

            if option.features.skeleton.fullbody
                if sum(sum(src.skeleton.HeadRightArm(ii:ii+W,:)))==0 || sum(sum(src.skeleton.HeadLeftArm(ii:ii+W,:)))==0 ...
                    || sum(sum(src.skeleton.TorsoRightLeg(ii:ii+W,:)))==0 || sum(sum(src.skeleton.TorsoLeftLeg(ii:ii+W,:)))==0
                    vis = [vis; 0];
                else
                    vis = [vis; 1];
                end
            else
                if sum(sum(src.skeleton.HeadRightArm(ii:ii+W,:)))==0 || sum(sum(src.skeleton.HeadLeftArm(ii:ii+W,:)))==0 
                    vis = [vis; 0];
                else
                    vis = [vis; 1];
                end 
            end
        end
        
    end
end    
%%% l-1 normalization
if strcmp(option.codebook.type, 'Kmeans')
    des = des./repmat(sum(des,2),1,size(des,2));
end

end



function des = feature_encoding_snippet(src,codebook,mu,sigma,option)

if strcmp(option.codebook.type, 'Kmeans')
    codebook = (full(codebook))';
    method = option.codebook.encoding_method;

    %%% when no person in the scene, occurrence histogram is a zero vector
    if isempty(src)
        des = zeros(1,size(codebook,1));
        return;
    end
    %%% standardization
    src = (src-repmat(mu,size(src,1),1))./repmat(sigma,size(src,1),1);
    dist = pdist2(src,codebook); % the default distance is euclidean

    switch method
    case 'hard_voting'
      [val,idx] = min(dist,[],2);
      val_t = repmat(val,1,size(dist,2));
      des = sum(val_t==dist,1); 
      des = des./sum(des);
    case 'soft_voting'
      beta = -1;
      val = exp(-beta .* dist);
      des = val./repmat(sum(val,2),1,size(val,2));

    case 'VLAD'
      kdtree = vl_kdtreebuild(codebook') ;
      nn = vl_kdtreequery(kdtree, codebook', src');
      assignments = zeros(size(codebook,1),size(src,1));
      assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
      des = vl_vlad(src',codebook',assignments,'SquareRoot','NormalizeMass');
      des = des';
    otherwise
     disp('error: no other option.');

    end
    
    

    
elseif strcmp(option.codebook.type, 'GMM')
    %%% todo Fisher vector encoding
    des = vl_fisher(src',codebook.means, codebook.covariance, codebook.priors, 'Improved');
    des = des';
    
    
else
    fprintf('ERROR: select codebook type Kmeans or GMM!\n');
end



end


