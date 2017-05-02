clear;
clc
load('KTH_EvaluationResults_2017-04-05-11-20.mat');
load('KTH_option_2017-04-05-11-20.mat');

rate = [];
for aa = 0:10
    rr = 0;
    dd = 0;
    rr = sum( eval_res.svm.Yt{aa+1}==eval_res.svm.Ytp{aa+1});
    dd = length(eval_res.svm.Yt{aa+1});

    rate = [rate; rr/dd];
end

figure; set(gcf,'color','w');
plot(0:0.1:1,rate,'-s','LineWidth',5,'MarkerSize',10);
ylim([0.6 1]);
grid on;
xlabel('\alpha','FontSize',20);
ylabel('Recognition Rate','FontSize',20);
title('KTH');


%% confusion matrix
Ytp = eval_res.svm.Ytp{3};
Yt = eval_res.svm.Yt{3};
cm = zeros(6,6);
for ii = 1:length(Yt)
    cm(Yt(ii),Ytp(ii)) = cm(Yt(ii),Ytp(ii))+1;
end
cm = cm./repmat(sum(cm,2),1,6);
figure; set(gcf,'color','w');
imagesc(cm);colorbar;
set(gca,'XTickLabel',option.fileIO.act_list,'FontSize',10);
set(gca,'YTickLabel',option.fileIO.act_list,'FontSize',10);
set(gca,'XTickLabelRotation',45);
set(gca,'YTickLabelRotation',45);
title('Action Recognition, \alpha=0.2','FontSize',20);