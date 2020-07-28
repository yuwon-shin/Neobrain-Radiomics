clear all
%%
load('T2_data.mat')
%%
x = X';

NF=20;  % 15 for T1 case
[idx, z] = rankfeatures(x(1:960,:),Y,'NumberOfIndices',NF);
X_new = x(idx,:);   % NF features(of all patients) remained       

tabulate(Y) % each 0&1class percentage

%% Box plot of significant features
for i=1:NF   
    subplot(1,NF,i)
    boxplot(x(idx(i),:),Y)
end
%% 8-folds
CN=8;
rng(5,'twister')         % For reproducibility
part = cvpartition(Y,'KFold',CN);  
Yall=[];
Yfitall=[];
Yscoreall=[];

for cn=1:CN
    istrain = training(part,cn); % Data for fitting
    istest = test(part,cn);      % Data for quality assessment
    tabulate(Y(istrain))
    %% Training
    N = sum(istrain);         % Number of observations in the training sample
    t = templateTree('MaxNumSplits',N); % control the depth of the tree
    tic
    rusTree = fitcensemble(X(istrain,idx(1:NF)),Y(istrain),'Method','RUSBoost', ...
        'NumLearningCycles',800,'Learners',t,'LearnRate',0.01,'nprint',100);
     %RUSBoost = Classification with Imbalacned Data

    %% Prediction
    [Yfit, Yscore] = predict(rusTree,X(istest,idx(1:NF)));
    confusionmat(Y(istest),Yfit) %test predict

    Yall = cat(1,Yall,Y(istest));  % Y test
    Yfitall=cat(1,Yfitall,Yfit);   % Y pridict

    Yscoreall=cat(1,Yscoreall,softmax(Yscore')');   % Y pridict class probability

end
%% Confusion Matrix and ROC
figure
confusionchart(Yall,Yfitall); %test 

figure
[XX,YY,TT,AUC] = perfcurve(Yall,Yscoreall(:,1),0);
plot(XX,YY,'b','LineWidth',2); 
xlabel('sensitivity')
ylabel('specificity')
title('ROC Curve');
title(sprintf('AUC=%.3f',AUC))
%%
% zz = sort(z,'descend');
% zz1 = zz(1:20,:);
% a = zeros(20,1);
% b = zz(21:960,:);
% zz2 = [a;b];
% p1 = bar(zz1);
% hold on;
% p2 = bar(zz2);
% set(p1,'FaceColor','black');
% set(p2,'FaceColor', [0.75 0.75 0.75]);
% title('Feature Ranking');
