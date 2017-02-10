function [hits,num_evals,features]=detect_poselets_MaxPooling(phog, svms, config, option)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% The code is modified by yan.zhang@uni-ulm.de at 2016.09.09, based on:
%%%----------------------------------------------------------------
%%% Given an RGB uint8 image returns the locations and scores of all
%%% poselets that were detected using the given svm classifiers.
%%%
%%% Copyright (C) 2009, Lubomir Bourdev and Jitendra Malik.
%%% This code is distributed with a non-commercial research license.
%%% Please see the license file license.txt included in the source directory.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


warning('off','MATLAB:intConvertOverflow');
warning('off','MATLAB:intConvertNonIntVal');

needs_features = nargout>2;


hits=hit_list;
all_hits = [];
num_evals = 0;
hits.n_poselets=0;
n_poselet_per_scale = 0;
for i = 1:length(svms)
    n_poselet_per_scale = n_poselet_per_scale+length(svms{i}.svm2poselet);
end

if strcmp(option.mode,'reduced')
    PatchSize = [1,3]; % patch size = [64,96] and [96,64] 
else
    PatchSize = 1:length(svms);
end

for aspect=PatchSize
    if needs_features
        features{aspect}=[];
    end
    for sc=1:length(phog.hog)
        [top_left,poselet1,score1,num_evals1,features1] = activate_poselets_one_scale(phog.hog{sc}.hog, phog.hog{sc}.samples_x, phog.hog{sc}.samples_y, svms{aspect}, needs_features, config);
        num_evals = num_evals + num_evals1;
        N = length(score1);
        hits.n_poselets = hits.n_poselets + length(svms{aspect}.svm2poselet);
        if N==0
            continue;
        end
        
        
        poselet1_all = poselet1+(sc-1)*n_poselet_per_scale; % re-encode the poselet index
        top_left = top_left+repmat(phog.hog{sc}.img_top_left,N,1);
        bounds = [top_left'; repmat(svms{aspect}.dims(2:-1:1)',1,N)]/phog.hog{sc}.scale;
        scales = repmat(phog.hog{sc}.scale,size(score1,1),1);
        hits=hits.append(hit_list(bounds,scales,score1,poselet1,poselet1_all,0));
        
        if needs_features
           features{aspect}(end+(1:N),:) = features1;
        end

        if config.DEBUG>1
           disp(sprintf('Scale %f hits: %d',phog.hog{sc}.scale,N));
        end
    end
end

N = hits.size;
if N>0
    hits.bounds = hits.bounds/phog.img_scale;
end

end




function [top_left,poselet_id,score,num_evals,features]=activate_poselets_one_scale(hog, samples_x, samples_y, svms, needs_features, config)
    [num_blocks, g_hog_blocks] = hog2features(hog,svms.dims,config); % returns in g_hog_blocks
    if prod(num_blocks)==0
       top_left=zeros(0,2);
       poselet_id=[];
       score=[];
       num_evals=[];
       features=[];
       return;
    end

    [qx,qy] = meshgrid(samples_x(1:num_blocks(1)),samples_y(1:num_blocks(2)));
    scores = g_hog_blocks*svms.svms(1:end-1,:)+repmat(svms.svms(end,:),prod(num_blocks),1);
    
    %---------------------------for recognition----------------------------
    
    % max-pooling 1
    %%% the following commented code implies each svm detector works
    %%% separately.
    [score1,q_loc1] = max(scores);
    idx = find(score1 > config.DETECT_SVM_THRESH); % remove negative responses
    score = (score1(idx))';
    q_loc = (q_loc1(idx))';   
    poselet_id = svms.svm2poselet(idx);

    % max-pooling 2
%     [n_loc,n_id1] = size(scores);
%     [score1, id1] = max(scores,[],2);
%     q_loc = 1:n_loc;
%     
%     %%% for each poselet, non-maximal suppression
%     xxx = [];
%     for ii = 1:n_id1
%         idxx = find(id1==ii);
%         score2 = score1(idxx);
%         [tmp,idxxx] = max(score2);
%         idxx(idxxx)=[];
%         xxx = [xxx;idxx];
%     end;
%     q_loc(xxx) = [];
%     score1(xxx) = [];
%     id1(xxx) = [];
%     score = score1;
%     poselet_id = svms.svm2poselet(id1);
%     
    
    
    
    
    top_left = [qx(q_loc) qy(q_loc)]+1;
    if size(top_left,2)>2   % this could happen if size(qx,1)==1 for a small image
        top_left = reshape(top_left,2,[])';
    elseif isempty(top_left)
        top_left = zeros(0,2);
    end

    num_evals = prod(num_blocks)*length(svms.svm2poselet);

    if needs_features
       features = g_hog_blocks(q_loc,:);
    else
       features=[];
    end
    %---------------------------for recognition-------------------------end

end







