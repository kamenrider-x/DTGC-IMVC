clear all;
clc;
tic;
addpath('fun/','yaleA_3view/');
load("yaleA_3view_0.1_baladata.mat");

V = size(X,2);
N = size(X{1},2);
W = cell(V, 1);
for i=1:3
    X{i}=NormalizeFea(X{i},0);
end
for i = 1:V
    present_indices = find(inds(:, i) == 1); 
    mv = length(present_indices);
    present_matrix = zeros(mv, N);
    for j = 1:mv
        present_matrix(j, present_indices(j)) = 1;
    end
    W{i} = present_matrix;
    X{i} = X{i}(:, present_indices);
end

lambda1=0.01;
beta=0.1;
alpha=0.1;
test_num =1;

for test = 1:test_num
    [result1, Clus, affinity2] = DTGC(X, V, N, W, lambda1, alpha, beta, gt);
    result(test,:) = result1;
    
end
me = mean(result,1);
st = std(result,1);
result(test_num+1,:) = me;
result(test_num+2,:) = st;
record_time = toc;
mea_time = record_time/test_num;