function feature_data = V3_ReadingData(dataset,option)
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
        label_file = sprintf([option.fileIO.dataset_path,...
            '/VideoFileList_Searching_Left_batch%d_label.txt'],bb);
        labels = importdata(label_file);
        for ii = 1:length(video_list)
            fprintf('-- read file %s\n',video_list{ii});
            filename = ...
                sprintf([option.fileIO.dataset_path,'/DT_features/DT_%s.txt'],...
                video_list{ii});
            feature_data{bb}{ii}.video = video_list{ii};
            feature_data{bb}{ii}.feature = importdata(filename);
            feature_data{bb}{ii}.label = labels(ii);
        end
    end
%     %%% group data into training/testing, each time leave one group out
%     disp('- group data into training/testing, each time leave one group out..');
%     groups = 1:4;
%     for gp = groups
%         outfilename = sprintf([option.fileIO.dataset_name '_leave_batch%i.mat'],gp);
%         fprintf('-- saving %s\n',outfilename);
%         groups_training = groups(groups~=gp);
%         stip_data_S = {};
%         stip_data_T = {};
%         for ii = groups_training
%             stip_data_S = [stip_data_S feature_data{ii}];
%         end
%         stip_data_T = feature_data{gp};
%         save(outfilename,'stip_data_S', 'stip_data_T','-v7.3');
%     end
    
else
    disp('ERROR: other datasets are ready!');
end

delete(gcp('nocreate'));
end


    
    