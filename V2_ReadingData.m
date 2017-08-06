function feature_des = V2_ReadingData(dataset,option)
%%% this function is used only once, when the feature extraction
%%% completes. 
if strcmp(dataset,'SenseEmotion3_Searching')
    
    %%% read data from file
    disp('- read data from file...');
    batches = 1:4;
    feature_data = {};
    parfor bb = batches
        video_list = importdata(sprintf([option.fileIO.dataset_path, ...
            '/VideoFileList_Searching_Left_batch%d.txt'], bb));
        
        for ii = 1:length(video_list)
            filename = ...
                sprintf([option.fileIO.dataset_path,'/DT_features/DT_%s.txt'],...
                video_list{ii});
            
            
        
        feature_data{bb} = ReadSTIPFile(filename,option);
        feature_data{bb} = V2_SkeletonFeatureExtraction(feature_data{bb});
        label_file = sprintf([option.fileIO.dataset_path,...
            '/VideoFileList_Searching_Left_batch%d_label.txt'],bb);
        labels = importdata(label_file);
        for i = 1:length(feature_data{bb})
            feature_data{bb}{i}.label = labels(i);
        end
    end
    
    %%% group data into training/testing, each time leave one group out
    disp('- group data into training/testing, each time leave one group out..');
    groups = 1:4;
    for gp = groups
        outfilename = sprintf('option.fileIO.dataset_name_leave_%i.mat',gp);
        groups_training = groups(groups~=gp);
        stip_data_S = {};
        stip_data_T = {};
        for ii = groups_training
            stip_data_S = [stip_data_S;feature_data{ii}];
        end
        stip_data_T = feature_data{gp};
        save(outfilename,'stip_data_S', 'stip_data_T','-v7.3');
    end
    
else
    disp('ERROR: other datasets are ready!');
end
delete(gcp('nocreate'));

end


    
    