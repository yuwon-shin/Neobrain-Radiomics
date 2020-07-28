clear all;
%%
M = csvread('pyradiomics_v4_T1_norm_results.csv',1);
Num = M(1:end,1);
X = M(1:end,3:end);
Y = M(1:end,2);
save T1_data.mat Num X Y
%%
M = csvread('pyradiomics_v4_T2_norm_results.csv',1);
Num = M(1:end,1);
X = M(1:end,3:end);
Y = M(1:end,2);
save T2_data.mat Num X Y
%%
M = csvread('pyradiomics_merge_T1_T2_results_norm.csv',1);
Num = M(1:end,1);
X = M(1:end,3:end);
Y = M(1:end,2);
save T1_T2_data.mat Num X Y