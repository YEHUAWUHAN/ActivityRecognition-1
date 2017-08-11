clear;
clc;

%%% combining different observation p and transition p according to the
%%% data nature, we can achieve fast and accurate recursive recognition.

% close all;

TTT = 10; % number of snippets
%% accumulated training data

res_accumulated = load('RochesterADL_EvaluationResults_2017-04-12-16-33.mat');
option_accumulated = load('RochesterADL_option_2017-04-12-16-33.mat');

res_batch =load('RochesterADL_EvaluationResults_2017-04-12-15-52.mat');
option_batch =load('RochesterADL_option_2017-04-12-15-52.mat');

for ss = 1:5
%%% resemble continuous labeling for test videos - a demo
%%% Recursive Bayesian Estimation with piece-wise constant assumption

test_labels = res_accumulated.eval_res{ss}.svm.test_labels;
Ytt = [];
B_track = {};
act_list = {'answerPhone','chopBanana','dialPhone','drinkWater','eatBanana','eatSnack','lookupInPhonebook','peelBanana','useSilverware','writeOnWhiteboard'};
for jj = 1:length(test_labels)
    
fprintf('- labeling video %s\n', test_labels{jj}.video);
vv = VideoReader([option_batch.option.fileIO.dataset_path,'/',test_labels{jj}.video '.avi']);
nn_snippets = test_labels{jj}.n_snippets; % equals to #frames
nnsb = sum(cellfun( @(x) sum(x.n_snippets), test_labels(1:jj-1)) );
W = option_batch.option.features.time_window;
S = option_batch.option.features.stride;
labels_p_a = res_accumulated.eval_res{ss}.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_p_normalized_a = labels_p_a./repmat(sum(labels_p_a,2), 1,size(labels_p_a,2));

labels_p_b = res_batch.eval_res{ss}.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_p_normalized_b = labels_p_b./repmat(sum(labels_p_b,2), 1,size(labels_p_b,2));

idxf = 1;
idxc = 1;
idxs = 1;

% fig = gcf; fig.Color = 'white';
% ax1 = subplot(1,2,1, 'Parent', fig); title(ax1,'video stream');
% ax2 = subplot(1,2,2, 'Parent', fig); title(ax2,'action label');
% ylim(ax2,[0 1]);
h1 = [];
A = ones(size(labels_p_a(1,:)));
A = A./sum(A);
A_track = [];
B_track{jj} = [];
labels_track{ss,jj} = [];
Aint = A;
labelst = find(cellfun(@(x) strcmp(x,test_labels{jj}.label), act_list, 'UniformOutput', 1));
T = ones(10,10);
% TT = 0;
TTT = nn_snippets/2;
while hasFrame(vv)
    if idxc > size(labels_p_a,1)
        break;
    end

    if mod(idxf,S)==0

%         if idxc <= TTT
%             b = labels_p_normalized_a(idxc,:);
%         else
%             b = labels_p_normalized_b(idxc,:);
%         end
        b = max([labels_p_a(idxc,:);labels_p_b(idxc,:)]);
        b = b/sum(b);
%         
%         if idxc > 2000 
%             T = eye(10);
%         else
%             T = ones(10,10);
%         end
        idxc = idxc+1;
        
        A = b.*(A*T);
        A = A./(sum(A));
        A_track = [A_track;A];
        Aint = Aint+A;
%         Aint = Aint./sum(Aint);
        B_track{jj} = [B_track{jj};Aint./sum(Aint)];

%         plot(B_track,'o-','Parent',ax2,'LineWidth',2);legend('boxing','handclapping','handwaving','jogging','running','walking');
%         plot(labels_p_normalized(1:idxc,:),'Parent',ax2,'LineWidth',2);legend('boxing','handclapping','handwaving','jogging','running','walking');
        
%         drawnow;
        idxs = idxs+1;
    end
    idxf = idxf+1;
%     pause(0.0001);
end
Ytt(jj) = find(Aint == max(Aint));
[~,labels_track{ss,jj}] = max(A_track,[],2);
labels_track{ss,jj} = [labels_track{ss,jj} labelst*ones(size(labels_track{ss,jj}))];

end
end


%% plot the eval curves

%%% unify the snippet length by replicating the last snippets
aa = cell2mat(cellfun(@(x) size(x,1), labels_track,'UniformOutput',false));
max_length = max(aa(:));
for ss = 1:5
for jj = 1:size(labels_track,2)
    nn = size(labels_track{ss,jj},1);
    if nn == max_length
        continue;
    else
        pp = max_length-nn;
        com = repmat(labels_track{ss,jj}(end,:),pp,1);
        labels_track{ss,jj} = [labels_track{ss,jj};com];        
    end
end
end

%%% plot the curve
reg_res_curve = [];
reg_res_curve_individual = [];
for cc = 1:max_length

    Ytp = cell2mat(cellfun(@(x) x(cc,1), labels_track,'UniformOutput',false));
    Yt = cell2mat(cellfun(@(x) x(cc,2), labels_track,'UniformOutput',false));
    cm = zeros(10,10);
    for ii = 1:length(Yt)
        cm(Yt(ii),Ytp(ii)) = cm(Yt(ii),Ytp(ii))+1;
    end
    reg_res_curve_individual = [reg_res_curve_individual; (diag(cm)./sum(cm,2))'];
    reg_res_curve = [reg_res_curve; sum(Ytp(:)==Yt(:))/length(Yt(:))];
  

end
colors = {[206,87,189],[112,208,87],[117,74,204],[200,205,90],[118,207,180],...
[201,75,48],[129,146,195],[190,135,66],[82,112,66],[81,55,54]};

figure;set(gcf,'Color','white');
plot(1:max_length, reg_res_curve,'-','LineWidth',5);hold on;
grid on;
ylim([0 1]);
xlabel('Number of Snippets','FontSize',20);
ylabel('Recognition Rate','FontSize',20);
title('RochesterADL','FontSize',20);
save('radl_mix.mat','reg_res_curve');

