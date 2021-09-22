function [output] = conv_layer_forward(input, layer, param)
% Conv layer forward
% input: struct with input data
% layer: convolution layer struct
% param: weights for the convolution layer

% output: 

h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
pad = layer.pad;
stride = layer.stride;
num = layer.num;
% resolve output shape
h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;

assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')
input_n.height = h_in;
input_n.width = w_in;
input_n.channel = c;

%% Fill in the code
% Iterate over the each image in the batch, compute response,

data_out = zeros(h_out * w_out * num, batch_size);

% for each image in batch
for b = 1:batch_size
    % reshape current image to 3D
    im = reshape(input.data(:, b), h_in, w_in, c);
    im_cur = padarray(im, [pad, pad], 0, 'both');
    feature_map = zeros(h_out, w_out, num);
    
    % kernel = (k*k*c, n)' = (n, k*k*c)
    % window = (k, k, c) reshape (k*k*c, 1)
    % kernel * window + b = (n, 1)
    
    % slide kernal by stride
    for j = 1:h_out
        for i = 1:w_out

            f_i = (i - 1) * stride + 1;
            f_j = (j - 1) * stride + 1;
            window = reshape(im_cur(f_i:f_i + k - 1, f_j:f_j + k - 1, :), [k*k*c, 1] );
            
            %add bias
            feature_map(i, j, :) = param.w' * window + param.b';

        end 
    end
    
    % flatten result
    data_out(:, b) = reshape(feature_map, h_out * w_out * num, []);
    
end
% Fill in the output datastructure with data, and the shape. 
output.data = data_out;
output.height = h_out;
output.width = w_out;
output.channel = num;
output.batch_size = batch_size;

end

