clear;clc;
load('RochesterADL_EvaluationResults_2017-04-05-11-42.mat');
load('RochesterADL_option_2017-04-05-11-42.mat');

rate = [];
for aa = 0:10
    rr = 0;
    dd = 0;
    for ss = 1:5
        rr = rr+sum(eval_res{ss}.svm.Yt{aa+1}==eval_res{ss}.svm.Ytp{aa+1});
        dd = dd+length(eval_res{ss}.svm.Yt{aa+1});
    end
    rate = [rate;rr/dd];
end

figure; set(gcf,'color','w');
plot(0:0.1:1,rate,'-s','LineWidth',5,'MarkerSize',10);
ylim([0.6 1]);
grid on;
xlabel('\alpha','FontSize',20);
ylabel('Recognition Rate','FontSize',20);
title('RochesterADL','FontSize',20);


    
%% confusion matrix
Ytp = [];
Yt = [];
for ss = 1:5
    Ytp = [Ytp;eval_res{ss}.svm.Ytp{3}];
    Yt = [Yt;eval_res{ss}.svm.Yt{3}];
end

cm = zeros(10,10);
for ii = 1:length(Yt)
    cm(Yt(ii),Ytp(ii)) = cm(Yt(ii),Ytp(ii))+1;
end
cm = cm./repmat(sum(cm,2),1,size(cm,2));
figure; set(gcf,'color','w');
imagesc(cm);colorbar;
set(gca,'XTickLabel',option.fileIO.act_list,'FontSize',20);
set(gca,'YTickLabel',option.fileIO.act_list,'FontSize',20);
set(gca,'XTickLabelRotation',45,'FontSize',10);
set(gca,'YTickLabelRotation',45,'FontSize',10);
title('Activity Recognition, \alpha=0.2','FontSize',20);
    