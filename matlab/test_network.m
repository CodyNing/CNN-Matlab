%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
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
        imgin1 = reshape(xtest(:, i + b - 1), [28, 28]);
        imshow(imgin1');
        title(sprintf('True: %d, Hat: %d', y_true - 1, y_hat - 1));
    end
end
disp(conf_ma);