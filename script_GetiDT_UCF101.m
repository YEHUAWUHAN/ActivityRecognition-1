%%% script to generate a list for videos in a dataset.
clear;clc

%% generate video file list in the folder
dataset_path = '~/Videos/Dataset_UCF101/';
dirinfo = dir(dataset_path);
dirinfo = dirinfo(3:end); % remove . and ..
N = length(dirinfo);
% remove non-video folders
for i = 1:N
    if strcmp(dirinfo(i).name, 'ucfTrainTestlist')
        dirinfo(i) = [];
        break;
    end
end



%% extract improved dense trajectory and its features
for K = 1 : length(dirinfo)
  thisdir = dirinfo(K).name;
  subdirinfo = dir(fullfile([dataset_path,thisdir], '/*.avi'));
  for j = 1 : length(subdirinfo)
      filename = subdirinfo(j).name;
      filepath = subdirinfo(j).folder;
      command_bin = './improved_trajectory_release/release/DenseTrackStab ';
      command_input = [filepath,'/',filename];
      command_output = [dataset_path,'iDT_',filename,'.txt'];
      command = [command_bin command_input '>' command_output];
      fprintf('run command: %s\n',command);
      system(command);
  end
end

