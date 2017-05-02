function feature_des = V2_SkeletonFeatureExtraction(feature_src,option)
%%% convert stips in feature_src in feature_des
% option = V2_GetDefaultConfig('KTH');

for i = 1 : length(feature_src);
    vf = [option.fileIO.dataset_path,'/',option.fileIO.dataset_name,'_Pose_MPI/',feature_src{i}.video];
    fprintf('- 2D skeleton feature extraction: %s\n',vf);
    [HeadRightArm,HeadLeftArm,TorsoRightLeg,TorsoLeftLeg] =...
        SkeletonFeatureOneVideo(vf,option);
    
    feature_des{i} = feature_src{i};
    feature_src{i} = {};
    feature_des{i}.skeleton.HeadRightArm = HeadRightArm;
    feature_des{i}.skeleton.HeadLeftArm = HeadLeftArm;
    feature_des{i}.skeleton.TorsoRightLeg = TorsoRightLeg;
    feature_des{i}.skeleton.TorsoLeftLeg = TorsoLeftLeg;
end

   

end



function [HeadRightArm,HeadLeftArm,TorsoRightLeg,TorsoLeftLeg] =...
    SkeletonFeatureOneVideo(workingDir,option)

frames_obj = dir(workingDir);

HeadRightArm=[];
HeadLeftArm=[];
TorsoRightLeg=[];
TorsoLeftLeg=[];
for i = 3:length(frames_obj) % starting from non-dir files
    json_file = [workingDir,'/',frames_obj(i).name];
    skeleton = ReadJSONFile(json_file);
    %%% *Sometimes no person is in the scene and hence skeleton=nan
    if isnan(skeleton)
        HeadRightArm = [HeadRightArm; 0 0 0 0];
        HeadLeftArm = [HeadLeftArm; 0 0 0 0];
        TorsoRightLeg = [TorsoRightLeg; 0 0 0];
        TorsoLeftLeg = [TorsoLeftLeg; 0 0 0];

        continue;
    end
    
    %%% notice that pose estimation at some frames is incorrect, lots of
    %%% points and their detect confidences are all 0
    head_length=1e-6+sqrt((skeleton(1)-skeleton(4)).^2+(skeleton(2)-skeleton(5)).^2);
    HeadRightArm = [HeadRightArm;...
     sqrt((skeleton(1)-skeleton(4)).^2+(skeleton(2)-skeleton(5)).^2)/head_length,...
     sqrt((skeleton(1)-skeleton(7)).^2+(skeleton(2)-skeleton(8)).^2)/head_length,...
     sqrt((skeleton(1)-skeleton(10)).^2+(skeleton(2)-skeleton(11)).^2)/head_length,...
     sqrt((skeleton(1)-skeleton(13)).^2+(skeleton(2)-skeleton(14)).^2)/head_length];
    
    if sum(isnan(HeadRightArm(end,:)))>0
        fdsgdsafdaf
    end
    
    HeadLeftArm = [HeadLeftArm;...
     sqrt((skeleton(1)-skeleton(4)).^2+(skeleton(2)-skeleton(5)).^2)/head_length,...
     sqrt((skeleton(1)-skeleton(16)).^2+(skeleton(2)-skeleton(17)).^2)/head_length,...
     sqrt((skeleton(1)-skeleton(19)).^2+(skeleton(2)-skeleton(20)).^2)/head_length,...
     sqrt((skeleton(1)-skeleton(22)).^2+(skeleton(2)-skeleton(23)).^2)/head_length];
 
    if option.features.skeleton.fullbody
        TorsoRightLeg = [TorsoRightLeg;...
     sqrt((skeleton(43)-skeleton(25)).^2+(skeleton(44)-skeleton(26)).^2)/head_length,...
     sqrt((skeleton(43)-skeleton(28)).^2+(skeleton(44)-skeleton(29)).^2)/head_length,...
     sqrt((skeleton(43)-skeleton(31)).^2+(skeleton(44)-skeleton(32)).^2)/head_length];
     
        TorsoLeftLeg = [TorsoLeftLeg;...
     sqrt((skeleton(43)-skeleton(34)).^2+(skeleton(44)-skeleton(35)).^2)/head_length,...
     sqrt((skeleton(43)-skeleton(37)).^2+(skeleton(44)-skeleton(38)).^2)/head_length,...
     sqrt((skeleton(43)-skeleton(40)).^2+(skeleton(44)-skeleton(41)).^2)/head_length];
        
    end
    
end
        
end



function des = ReadJSONFile(filename)
aa = importdata(filename);
bb = aa{5}(11:end-1);
cc = strsplit(bb,',');
des = cellfun(@(x) str2double(x), cc);

% deal with 0-detection confidence, this means that the joint is occluded.
% then the joint is replaced by the chest coordinate
dec_conf = des(3:3:end);
idx_occluded = find(dec_conf==0);
for ii = 1:length(idx_occluded)
    des(idx_occluded(ii)*3-2) = des(end-2);
    des(idx_occluded(ii)*3-1) = des(end-1);
end

end









    