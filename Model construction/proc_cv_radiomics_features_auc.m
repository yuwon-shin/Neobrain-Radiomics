clear all
%%
load('T1_T2_data.mat')

%%
x = X';
%%
NF=20;  
[idx1, z1] = rankfeatures(x(1:960,:),Y,'NumberOfIndices',NF/2);
[idx2, z2] = rankfeatures(x(961:end,:),Y,'NumberOfIndices',NF/2);
idx = cat(1,idx1,idx2+960);
X_new = x(idx,:);   % NF features(of all patients) remained       

tabulate(Y) % each 0&1class percentage

%% Box plot of significant features
for i=1:NF   
    subplot(1,NF,i)
    boxplot(x(idx(i),:),Y)
end
%% 8-folds
CN=8;
rng(5,'twister')         % For reproducibility, �õ尪=10,
part = cvpartition(Y,'KFold',CN);  %(n,'LeaveOut')
%part = cvpartition(length(Y),'LeaveOut');
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
[XX3,YY3,TT3,AUC3] = perfcurve(Yall,Yscoreall(:,1),0);
plot(XX3,YY3,'LineWidth',2); 
xlabel('sensitivity')
ylabel('specificity'); hold off;
title(sprintf('AUC=%.3f',AUC3))