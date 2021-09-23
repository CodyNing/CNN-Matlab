%% Network defintion
layers = get_lenet();

%% Loading data
fullset = true;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix

conf_ma = zeros(layers{10}.num);
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    
    for b = 1:100
        y_true = ytest(1, i + b - 1);
        [y_hat_prob, y_hat] = max(P(:, b));
        conf_ma(y_true, y_hat) = conf_ma(y_true, y_hat) + 1;
    end
end

names = {'0', '1', '2', '3', '4', '5' '6', '7', '8', '9'};
disp(array2table(conf_ma, 'RowNames', names, 'VariableNames', names));
