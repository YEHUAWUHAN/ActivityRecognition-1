function [Xs, Xt] = V3_pca_dr(Xs,Xt,target_dim, flag)

if strcmp(flag,'reduce')
    [U,Xs,~,~,~,Xs_mean] = pca(Xs,'NumComponents',target_dim, 'Centered','on');
    Xt = (Xt-repmat(Xs_mean,size(Xt,1),1))*U;
elseif strcmp(flag,'whitening')
    [U,Xs,lambda,~,~,Xs_mean] = pca(Xs,'NumComponents',target_dim, 'Centered','on');
    D = diag(lambda.^(-0.5));
    Xs = Xs*D;
    Xt = (Xt-repmat(Xs_mean,size(Xt,1),1))*U*D;
else
    fprintf('ERROR: V3_pca_dr(); incorrect flag\n');
end


% %%% mean and cov computation
% Xs_mean = mean(Xs);
% Xs_cov = cov(Xs);
% 
% %%% eigen-decomposition
% [U_all,D_all] = eig(Xs_cov);
% U = U_all(:,1:target_dim);
% D = D_all(1:target_dim,1:target_dim);
% 
% %%% whitening
% Xs = (Xs-repmat(Xs_mean,size(Xs,1),1))*U * D.^(-0.5);
% Xt = (Xt-repmat(Xs_mean,size(Xt,1),1))*U * D.^(-0.5);

end

