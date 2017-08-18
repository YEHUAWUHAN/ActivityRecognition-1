function feature_data = V4_ReadingData(option, idx_minibatch)
%%% This function is especially for large-scale leanring. 
%%% It is special for HMDB51 and UCF101

%%% two modes are supported:
%%% (1) 'global_random': randomly select balanced dataset from the training
%%% data for training GMM
%%% (2) 'mini-batch': after obtaining GMM, we read mini-batches, encoding
%%% and train the classifier sequentially.
dataset_path = option.fileIO.dataset_path;
switch option.fileIO.dataset_name

    case 'HMDB51'
        %%% read data from file
        %%% generate file-to-label lookup table
        
        feature_data = {};
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
        n = 1;
        split_list_path = [dataset_path,'/testTrainMulti_7030_splits'];
        
        
        if strcmp(option.fileIO.readingmode,'global_random')
            
            % read files from the split lists.
            for K = 1 : length(dirinfo)
                thisdir = dirinfo(K).name;
%                 subdirinfo = dir(fullfile([dataset_path,thisdir], '/*.avi'));
                split_info = importdata([split_list_path,'/',thisdir,'_test_split1.txt']);
                label = K;
                list_alltraining = split_info.textdata(split_info.data==1);
                batch_size = min(length(list_alltraining),option.fileIO.batch_size_per_class);
                list_batchtraining = list_alltraining(randperm(length(list_alltraining),batch_size));
                for ii = 1 : length(list_batchtraining)
                    filename = list_batchtraining{ii};             
                    fprintf('-- read features: %s\n',filename);
                    feature_filename = [dataset_path,'/DT_features/iDT_',filename,'.txt'];
                    feature_data{n}.video = filename; % with .avi without .txt
                    feature_data{n}.feature = importdata(feature_filename);
                    feature_data{n}.label = label;
                    n = n+1;
                end
            end
            
        elseif strcmp(option.fileIO.readingmode,'mini_batch')
            % In this setting, one mini-batch is balanced, where samples
            % from each class are randomly sampled.
            
            % 1. for each class, we shuffle the samples for training.
            for K = 1 : length(dirinfo)
                thisdir = dirinfo(K).name;
                split_info = importdata([split_list_path,'/',thisdir,'_test_split1.txt']);
                list_alltraining = split_info.textdata(split_info.data==1);
                list_alltraining = list_alltraining(randperm(length(list_alltraining))); 
            end
            
            % 2. for each class, we extract equal number of smaples to
            % compose the mini-batch.
            for K = 1 : length(dirinfo)
                thisdir = dirinfo(K).name;
%                 subdirinfo = dir(fullfile([dataset_path,thisdir], '/*.avi'));
                split_info = importdata([split_list_path,'/',thisdir,'_test_split1.txt']);
                label = K;
                batch_size = min(length(list_alltraining),option.fileIO.batch_size_per_class);
                if idx_minibatch > length(list_alltraining)
                    disp('[INFO] training data is used up.');
                    break;
                end
                for ii = idx_minibatch : min(idx_minibatch+batch_size, length(list_alltraining))
                    filename = list_alltraining{ii};             
                    fprintf('-- read features: %s\n',filename);
                    feature_filename = [dataset_path,'/DT_features/iDT_',filename,'.txt'];
                    feature_data{n}.video = filename; % with .avi without .txt
                    feature_data{n}.feature = importdata(feature_filename);
                    feature_data{n}.label = label;
                    n = n+1;
                end
            end
        else
            disp('Error: this reading mode is not supported!');
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


    
    