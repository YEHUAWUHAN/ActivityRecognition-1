%%% script to generate a list for videos in a dataset.
clear;clc

%% generate video file list in the folder
dataset_path = '~/Videos/Dataset_HMDB51/';
dirinfo = dir(dataset_path);
dirinfo = dirinfo(3:end); % remove . and ..
N = length(dirinfo);
% remove non-video folders
for i = 1:N
    if strcmp(dirinfo(i).name, 'testTrainMulti_7030_splits')
        dirinfo(i) = [];
        break;
    end
end
N = N-1;

for i = 1:N
    if strcmp(dirinfo(i).name, 'DT_features')
        dirinfo(i) = [];
        break;
    end
end
N = N-1;

%% extract improved dense trajectory and its features
for K = 1 : length(dirinfo)
  thisdir = dirinfo(K).name;
  subdirinfo = dir(fullfile([dataset_path,thisdir], '/*.avi'));
  for j = 1 : length(subdirinfo)
      filename = subdirinfo(j).name;
      if sum(filename=='(' | filename==')' | filename=='&' | filename==';') >0 
          filepath = [dataset_path,thisdir];
          filename_input = strrep(filename,'(','\(');
          filename_input = strrep(filename_input,')','\)');
          filename_input = strrep(filename_input,'&','\&');
          filename_input = strrep(filename_input,';','";"');
      else 
          filename_input = filename;
      end
      command_bin = './improved_trajectory_release/release/DenseTrackStab ';
      command_input = [filepath,'/',filename_input];
      command_output = [dataset_path,'iDT_',filename,'.txt'];
      command = [command_bin command_input];
      fprintf('run command: %s\n',command);
      [status,cmdout] = system(command);
      fid = fopen(command_output,'wt');
      fprintf(fid,cmdout);
      fclose(fid);
      end
  end
end

