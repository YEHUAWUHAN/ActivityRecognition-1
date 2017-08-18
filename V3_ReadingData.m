function feature_data = V3_ReadingData(option)
%%% this function is used only once, when the feature extraction
%%% completes. 

switch option.fileIO.dataset_name
    case 'SenseEmotion3_Searching'
    
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

    case 'HMDB51'
        %%% read data from file
        %%% generate file-to-label lookup table
        
        feature_data = {};
        dataset_path = option.fileIO.dataset_path; 
        dirinfo = dir(option.fileIO.dataset_path);
        dirinfo = dirinfo(3:end); % remove . and ..
        N = length(dirinfo);
        % remove non-video folders
        for i = 1:N
            if strcmp(dirinfo(i).name, 'testTrainMulti_7030_splits')
                dirinfo(i) = [];
                break;
            end
        end
        for i = 1:N-1
            if strcmp(dirinfo(i).name, 'DT_features')
                dirinfo(i) = [];
                break;
            end
        end

        % 
        for K = 1 : length(dirinfo)
            thisdir = dirinfo(K).name;
%             subdirinfo = dir(fullfile([option.fileIO.dataset_path,thisdir], '/*.avi'));
            subdirinfo = dir([dataset_path,'/',thisdir, '/*.avi']);
            label = K;
            for ii = 1 : length(subdirinfo)
                filename = subdirinfo(ii).name;             
                fprintf('-- read features: %s\n',filename);
                feature_filename = [dataset_path,'/DT_features/iDT_',filename,'.txt'];
                feature_data{ii}.video = filename; % with .avi without .txt
                feature_data{ii}.feature = importdata(feature_filename);
                feature_data{ii}.label = label;  
            end
        end
            
       
    
    case 'UCF101'
        feature_data = {};
        dirinfo = dir(option.fileIO.dataset_path);
        dirinfo = dirinfo(3:end); % remove . and ..
        N = length(dirinfo);
        % remove non-video folders
        for i = 1:N
            if strcmp(dirinfo(i).name, 'ucfTrainTestlist')
                dirinfo(i) = [];
                break;
            end
        end
        for i = 1:N-1
            if strcmp(dirinfo(i).name, 'DT_features')
                dirinfo(i) = [];
                break;
            end
        end
        
        for K = 1 : length(dirinfo)
            thisdir = dirinfo(K).name;
            subdirinfo = dir(fullfile([dataset_path,thisdir], '/*.avi'));
            label = K;
            for ii = 1 : length(subdirinfo)
                filename = subdirinfo(ii).name;             
                fprintf('-- read features: %f\n',filename);
                feature_filename = [dataset_path,'DT_features/iDT_',filename,'.txt'];
                feature_data{ii}.video = filename; % with .avi without .txt
                feature_data{ii}.feature = importdata(feature_filename);
                feature_data{ii}.label = label;  
            end
        end
        

    otherwise
        disp('ERROR: other datasets are ready!');
end

delete(gcp('nocreate'));
end


    
    