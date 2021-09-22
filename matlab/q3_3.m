%% Network defintion
layers = get_lenet();

%% Loading data
x = zeros(28*28, 6);

bmps = dir('../q3.3/*.bmp');

batch_size = size(bmps, 1);

for i = 1:batch_size
    file = bmps(i);
    im = imread(sprintf('%s/%s', file.folder, file.name));
    im_gray = rgb2gray(im);
    im_norm = double(im_gray) / 255.0;
    x(:, i) = reshape(im_norm', 28*28, []);
end

% load the trained weights
load lenet.mat

layers{1}.batch_size = batch_size;

[output, P] = convnet_forward(params, layers, x);
[y_hat_prob, y_hat] = max(P);
disp(y_hat - 1);

for i=1:batch_size
    subplot(3,2,i);
    imshow(reshape(x(:,i), [28, 28])');
    title(sprintf('prediction: %d', y_hat(i) - 1));
end
