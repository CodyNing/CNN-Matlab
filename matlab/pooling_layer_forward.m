function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros(h_out * w_out * c, batch_size);

    % for each image in batch
    for b = 1:batch_size
        % reshape current image to 3D
        im = reshape(input.data(:, b), h_in, w_in, c);
        im_cur = padarray(im, [pad, pad], 0, 'both');
        
        feature_map = zeros(h_out, w_out, c);

        % slide kernal by stride
        for j = 1:h_out
            for i = 1:w_out
                f_i = (i - 1) * stride + 1;
                f_j = (j - 1) * stride + 1;
                kernal = im_cur(f_i:f_i + k - 1, f_j:f_j + k - 1, :);
                feature_map(i, j, :) = max(kernal, [], [1, 2]);

            end 
        end

        % flatten result
        output.data(:, b) = reshape(feature_map, h_out * w_out * c, []);

    end

end

