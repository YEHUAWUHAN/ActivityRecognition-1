clear;clc;
close all;


%% accumulated training

load('RochesterADL_EvaluationResults_2017-04-12-16-33.mat');
load('RochesterADL_option_2017-04-12-16-33.mat');

for ss = 1:5
%%% resemble continuous labeling for test videos - a demo
%%% Recursive Bayesian Estimation with piece-wise constant assumption

test_labels = eval_res{ss}.svm.test_labels;
Ytt = [];
B_track = {};
act_list = {'answerPhone','chopBanana','dialPhone','drinkWater','eatBanana','eatSnack','lookupInPhonebook','peelBanana','useSilverware','writeOnWhiteboard'};
for jj = 1:length(test_labels)
    
fprintf('- labeling video %s\n', test_labels{jj}.video);
vv = VideoReader([option.fileIO.dataset_path,'/',test_labels{jj}.video '.avi']);
nn_snippets = test_labels{jj}.n_snippets; % equals to #frames
nnsb = sum(cellfun( @(x) sum(x.n_snippets), test_labels(1:jj-1)) );
W = option.features.time_window;
S = option.features.stride;
labels_p = eval_res{ss}.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_p_normalized = labels_p./repmat(sum(labels_p,2), 1,size(labels_p,2));

idxf = 1;
idxc = 1;
idxs = 1;
vis = eval_res{ss}.stip_T_encoded{jj}.visible;

% fig = gcf; fig.Color = 'white';
% ax1 = subplot(1,2,1, 'Parent', fig); title(ax1,'video stream');
% ax2 = subplot(1,2,2, 'Parent', fig); title(ax2,'action label');
% ylim(ax2,[0 1]);
h1 = [];
A = ones(size(labels_p(1,:)));
A = A./sum(A);
A_track = [];
B_track{jj} = [];
labels_track{ss,jj} = [];
Aint = A;
labelst = find(cellfun(@(x) strcmp(x,test_labels{jj}.label), act_list, 'UniformOutput', 1));
% TT = 0.75*nn_snippets;
TT = 100000;
while hasFrame(vv)
    if idxc > size(labels_p,1)
        break;
    end
%     frame = readFrame(vv);
%     if idxf == 1
%         h1 = imshow(frame,'Parent',ax1);drawnow;
%     else
%         set(h1,'CData',frame);
%     end
    if mod(idxf,S)==0
        if idxs <=TT
            T = ones(10,10);
        else
            T = eye(10);
        end
        if vis(idxs)
            b = labels_p_normalized(idxc,:);
            idxc = idxc+1;
        else
            b = ones(size(labels_p(1,:)));
            b = b./(sum(b));
        end
        
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


%% batch training




load('RochesterADL_EvaluationResults_2017-04-12-15-52.mat');
load('RochesterADL_option_2017-04-12-15-52.mat');

for ss = 1:5
%%% resemble continuous labeling for test videos - a demo
%%% Recursive Bayesian Estimation with piece-wise constant assumption

test_labels = eval_res{ss}.svm.test_labels;
Ytt = [];
B_track = {};
act_list = {'answerPhone','chopBanana','dialPhone','drinkWater','eatBanana','eatSnack','lookupInPhonebook','peelBanana','useSilverware','writeOnWhiteboard'};
for jj = 1:length(test_labels)
    
fprintf('- labeling video %s\n', test_labels{jj}.video);
vv = VideoReader([option.fileIO.dataset_path,'/',test_labels{jj}.video '.avi']);
nn_snippets = test_labels{jj}.n_snippets; % equals to #frames
nnsb = sum(cellfun( @(x) sum(x.n_snippets), test_labels(1:jj-1)) );
W = option.features.time_window;
S = option.features.stride;
labels_p = eval_res{ss}.svm.prob_estimates(nnsb+1: nnsb+nn_snippets,:);
labels_p_normalized = labels_p./repmat(sum(labels_p,2), 1,size(labels_p,2));

idxf = 1;
idxc = 1;
idxs = 1;
vis = eval_res{ss}.stip_T_encoded{jj}.visible;

% fig = gcf; fig.Color = 'white';
% ax1 = subplot(1,2,1, 'Parent', fig); title(ax1,'video stream');
% ax2 = subplot(1,2,2, 'Parent', fig); title(ax2,'action label');
% ylim(ax2,[0 1]);
h1 = [];
A = ones(size(labels_p(1,:)));
A = A./sum(A);
A_track = [];
B_track{jj} = [];
labels_trackb{ss,jj} = [];
Aint = A;
labelst = find(cellfun(@(x) strcmp(x,test_labels{jj}.label), act_list, 'UniformOutput', 1));

% TT = 0.85*nn_snippets;
TT = 1000000;
while hasFrame(vv)
    if idxc > size(labels_p,1)
        break;
    end
%     frame = readFrame(vv);
%     if idxf == 1
%         h1 = imshow(frame,'Parent',ax1);drawnow;
%     else
%         set(h1,'CData',frame);
%     end
    if mod(idxf,S)==0
        if idxs <=TT
            T = ones(10,10);
        else
            T = eye(10);
        end
        if vis(idxs)
            b = labels_p_normalized(idxc,:);
            idxc = idxc+1;
        else
            b = ones(size(labels_p(1,:)));
            b = b./(sum(b));
        end
        
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
[~,labels_trackb{ss,jj}] = max(A_track,[],2);
labels_trackb{ss,jj} = [labels_trackb{ss,jj} labelst*ones(size(labels_trackb{ss,jj}))];

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
        
        com = repmat(labels_trackb{ss,jj}(end,:),pp,1);
        labels_trackb{ss,jj} = [labels_trackb{ss,jj};com];
    end
end
end

%%% plot the curve
reg_res_curve = [];
reg_res_curveb = [];
reg_res_curve_individual = [];
reg_res_curve_individualb = [];
for cc = 1:max_length

    Ytp = cell2mat(cellfun(@(x) x(cc,1), labels_track,'UniformOutput',false));
    Yt = cell2mat(cellfun(@(x) x(cc,2), labels_track,'UniformOutput',false));
    cm = zeros(10,10);
    for ii = 1:length(Yt)
        cm(Yt(ii),Ytp(ii)) = cm(Yt(ii),Ytp(ii))+1;
    end
    reg_res_curve_individual = [reg_res_curve_individual; (diag(cm)./sum(cm,2))'];
    reg_res_curve = [reg_res_curve; sum(Ytp(:)==Yt(:))/length(Yt(:))];
    
    Ytp = cell2mat(cellfun(@(x) x(cc,1), labels_trackb,'UniformOutput',false));
    Yt = cell2mat(cellfun(@(x) x(cc,2), labels_trackb,'UniformOutput',false));
    cm = zeros(10,10);
    for ii = 1:length(Yt)
        cm(Yt(ii),Ytp(ii)) = cm(Yt(ii),Ytp(ii))+1;
    end
    reg_res_curve_individualb = [reg_res_curve_individualb; (diag(cm)./sum(cm,2))'];
    reg_res_curveb = [reg_res_curveb; sum(Ytp(:)==Yt(:))/length(Yt(:))];
end

obj = load('radl_mix');
figure;set(gcf,'Color','white');
plot(1:max_length, reg_res_curve,'-','LineWidth',5);hold on;
plot(1:max_length, reg_res_curveb,'--','LineWidth',5);hold on;
plot(1:max_length, obj.reg_res_curve,':','LineWidth',5);

h_legend=legend('accumulated','batch','mixture');
set(h_legend,'FontSize',20);
grid on;
ylim([0 1]);
xlabel('Number of Snippets','FontSize',20);
ylabel('Recognition Rate','FontSize',20);
title('RochesterADL','FontSize',20);

% 
% figure(2);set(gcf,'Color','white');
% for cc = 1:size(reg_res_curve_individual,2)
%     plot(reg_res_curve_individual(:,cc),'-','Color',colors{cc}/255,'LineWidth',5);hold on;
% end
% legend(act_list)
% grid on;
% ylim([0 1]);
% xlabel('Number of Snippets');
% ylabel('Recognition Rate');
% title('Individual Performance - Accumulated');
% 
% 
% 
% figure(3);set(gcf,'Color','white');
% for cc = 1:size(reg_res_curve_individualb,2)
%     plot(reg_res_curve_individualb(:,cc),'-','Color',colors{cc}/255,'LineWidth',2);hold on;
% end
% legend(act_list)
% grid on;
% ylim([0 1]);
% xlabel('Number of Snippets');
% ylabel('Recognition Rate');
% title('Individual Performance - Batch');
% 



















