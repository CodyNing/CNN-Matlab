addpath '..\matlab';

%% load images
im_dir = dir('../images/*.*');
im_dir = im_dir(~ismember({im_dir.name},{'.','..'}));

file_count = size(im_dir, 1);

%% process images
x = zeros(28 * 28, 1);
x_i = 0;

for i = 1:file_count
    file = im_dir(i);
    im = imread(sprintf('%s/%s', file.folder, file.name));
    im_gray = rgb2gray(im);
    T = adaptthresh(im_gray, 0.95);
    BW = 1 - imbinarize(im_gray, T);
    CC = bwconncomp(BW);
    stats = regionprops(CC, 'BoundingBox');
    
    BB = cat(1, stats.BoundingBox);
    sub_hs = BB(:, 4);
    
    % remove sub image has height smaller than half of the mean height.
    sub_h_mean = sum(sub_hs) / CC.NumObjects;
    outlier = sub_hs < sub_h_mean / 2;
    
    valid_BB = BB(~outlier, :);
    
    for j = 1: length(valid_BB)
        sub_im = imcrop(BW, valid_BB(j, :));
        sub_h = valid_BB(j, 4);
        sub_w = valid_BB(j, 3);
        
        % pad image to a square first
        elen = max(sub_h, sub_w);
        sub_im_resize = padarray(sub_im, [floor((elen - sub_h)/2) floor((elen - sub_w)/2)], 0, 'post');
        sub_im_resize = padarray(sub_im_resize, [ceil((elen - sub_h)/2) ceil((elen - sub_w)/2)], 0, 'pre');
        
        % resize to 28 * 28
        sub_im_resize = imresize(sub_im_resize, [28, 28]);
        
        x_i = x_i + 1;
        x(:, x_i) = reshape(sub_im_resize', 28 * 28, []);
    end
end


%% feed into convnet
layers = get_lenet();
load lenet.mat

layers{1}.batch_size = x_i;

[output, P] = convnet_forward(params, layers, x);
[y_hat_prob, y_hat] = max(P);
disp(y_hat - 1);

for i=1:x_i
    subplot(15,5,i);
    imshow(reshape(x(:,i), [28, 28])');
    title(sprintf('prediction: %d', y_hat(i) - 1));
end

rmpath '..\matlab';