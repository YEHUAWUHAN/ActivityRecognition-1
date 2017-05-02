%% resemble continuous labeling for test videos - a demo
%%% Recursive Bayesian Estimation with piece-wise constant assumption
clear;close all;
clc;
res_bb = load('Result_experiment2/KTH_EvaluationResults_2017-04-12-11-21.mat');
option_bb = load('Result_experiment2/KTH_option_2017-04-12-11-21.mat');

res_aa = load('Result_experiment2/KTH_EvaluationResults_2017-04-12-11-25.mat');
option_aa = load('Result_experiment2/KTH_option_2017-04-12-11-25.mat');



test_labels = res_aa.eval_res.svm.test_labels;
Ytt = [];
B_track = {};
label_track = {};
act_list = {'boxing','handclapping','handwaving','jogging','running','walking'};



for jj = 33

fprintf('- labeling video %s\n', test_labels{jj}.video);
vv = VideoReader([option_aa.option.fileIO.dataset_path,'/',test_labels{jj}.video '.avi']);
nn_snippets = test_labels{jj}.n_snippets; % equals to #frames
nnsb = sum(cellfun( @(x) sum(x.n_snippets), test_labels(1:jj-1)) );
W = option_aa.option.features.time_window;
S = option_aa.option.features.stride;
vis = res_aa.eval_res.stip_T_encoded{jj}.visible;

labels_pa = res_aa.eval_res.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_pa_normalized = labels_pa./repmat(sum(labels_pa,2), 1,size(labels_pa,2));

labels_pb = res_bb.eval_res.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_pb_normalized = labels_pb./repmat(sum(labels_pb,2), 1,size(labels_pb,2));

idxf = 1;
idxc = 1;
idxs = 1;

% T = eye(6);
T = ones(6,6);
fig = figure; fig.Color = 'white';
fig.Position = [0 0 900 300];
ax1 = subplot(1,2,1, 'Parent', fig); title(ax1,'video stream');
ax2 = subplot(1,2,2, 'Parent', fig); title(ax2,'action label');

ylim(ax2,[0 1]);
h1 = [];
A = ones(size(labels_pa(1,:)));
b = A./sum(A);
A_track = [];
B_track{jj} = [];
labels_track{jj} = [];
Aint = A;
labelst = find(cellfun(@(x) strcmp(x,test_labels{jj}.label), act_list, 'UniformOutput', 1));
TTT = nn_snippets/2;
while hasFrame(vv)
    if idxs > size(labels_pa,1)
        break;
    end
    frame = readFrame(vv);
    if idxf == 1
        pause;
        h1 = imshow(frame,'Parent',ax1);drawnow;
    else
        set(h1,'CData',frame);
    end
    if mod(idxf,S)==0

        if vis(idxs)
        
            if idxs <= TTT
                b = labels_pb_normalized(idxs,:);
            else
                b = labels_pb_normalized(idxs,:);
            end
        end
        
        A = b.*(A*T);
        A = A./(sum(A));
        A_track = [A_track;A];
        Aint = Aint+A;
%         Aint = Aint./sum(Aint);
        B_track{jj} = [B_track{jj};Aint./sum(Aint)];

        if idxs>1
            plot(A_track,'o-','Parent',ax2,'LineWidth',2);
        else
            for ii = 1:6
                plot(A_track(ii),'o','Parent',ax2,'LineWidth',2);hold on;
            end
            hold off;
        end
        hl=legend('boxing','handclapping','handwaving','jogging','running','walking');
        rect = [0.25 0.25 0.25 0.1];
        set(hl, 'Position', rect);
%         plot(labels_p_normalized(1:idxc,:),'Parent',ax2,'LineWidth',2);legend('boxing','handclapping','handwaving','jogging','running','walking');
        
        drawnow;
        idxs = idxs+1;
    end
    idxf = idxf+1;
    pause(0.03);
end


end
    
    
