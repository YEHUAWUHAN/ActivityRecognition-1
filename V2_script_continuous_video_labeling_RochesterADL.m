    
%% resemble continuous labeling for test videos - a demo
%%% Recursive Bayesian Estimation with piece-wise constant assumption


clear all;
close all;
clc



resa=load('/home/yzhang/workspace/ActivityRecognition/Result_experiment2/RochesterADL_EvaluationResults_2017-04-12-16-33.mat');
optiona=load('/home/yzhang/workspace/ActivityRecognition/Result_experiment2/RochesterADL_option_2017-04-12-16-33.mat');

resb=load('/home/yzhang/workspace/ActivityRecognition/Result_experiment2/RochesterADL_EvaluationResults_2017-04-12-15-52.mat');
optionb=load('/home/yzhang/workspace/ActivityRecognition/Result_experiment2/RochesterADL_option_2017-04-12-15-52.mat');


ss = 3;
test_labels = resa.eval_res{ss}.svm.test_labels;
Ytt = [];
% for jj = 1:length(test_labels) 
colors = {[206,87,189],[112,208,87],[117,74,204],[200,205,90],[118,207,180],...
[201,75,48],[129,146,195],[190,135,66],[82,112,66],[81,55,54]};


for jj = 30

fprintf('- labeling video %s\n', test_labels{jj}.video);
vv = VideoReader([optiona.option.fileIO.dataset_path,'/',test_labels{jj}.video '.avi']);
nn_snippets = test_labels{jj}.n_snippets; % equals to #frames
nnsb = sum(cellfun( @(x) sum(x.n_snippets), test_labels(1:jj-1)) );
W = optiona.option.features.time_window;
S = optiona.option.features.stride;
labels_pa = resa.eval_res{ss}.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_pa_normalized = labels_pa./repmat(sum(labels_pa,2), 1,size(labels_pa,2));

labels_pb = resb.eval_res{ss}.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_pb_normalized = labels_pb./repmat(sum(labels_pb,2), 1,size(labels_pb,2));

vis = resa.eval_res{ss}.stip_T_encoded{jj}.visible;

idxf = 1; 
idxc = 1;
idxs = 1;

fig = figure; fig.Color = 'white';fig.Position = [0 0 1200 300];

% ax1 = axes;
% fig2 = figure(2);fig2.Color='white';
% ax2 = axes;

ax1 = subplot(1,2,1, 'Parent', fig); title(ax1,'video stream');
ax2 = subplot(1,2,2, 'Parent', fig); title(ax2,'action label');

ylim(ax2,[0 1]);
h1 = [];
A = ones(size(labels_pa(1,:)));
A = A./sum(A);
A_track = [];
B_track = [];
Aint = A;
T = ones(10,10);
TT = nn_snippets/2;

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
            if idxs <= TT
                b = labels_pa_normalized(idxs,:);
            else
                b = labels_pb_normalized(idxs,:);
            end
        end
        
        A = b.*(A*T);
        A = A./(sum(A));
        A_track = [A_track;A];
        Aint = Aint+A;
        Aint = Aint./sum(Aint);
        B_track = [B_track;Aint./sum(Aint)];

        for cc = 1:size(B_track,2)
            plot(A_track(:,cc),'o-','Color',colors{cc}/255,'Parent',ax2,'LineWidth',2);
            hold(ax2,'on');
        end

        hl=legend('answerPhone','chopBanana','dialPhones','drinkWater','eatBanana','eatSnack','lookupInPhonebook','peelBanana','useSilverware','writeOnWhiteBoard');
        rect = [0.02 0.25 0.1 0.1];
        set(hl, 'Position', rect)
        drawnow; 
        idxs = idxs+1; hold(ax2,'off');
    end
    idxf = idxf+1; pause(0.03);
end
set( get(ax2,'XLabel'), 'String', 'number of snippets','FontSize',10 );
set( get(ax2,'YLabel'), 'String', 'probabilities','FontSize',10 );
Ytt(jj) = find(Aint == max(Aint));

end




