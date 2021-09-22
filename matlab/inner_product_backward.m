function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
%(100 * 500)
param_grad.b = zeros(size(param.b));
%(800 * 100)
param_grad.w = zeros(size(param.w));


param_grad.b = sum(output.diff.');
% (800 * 100) * (100 * 500) = (800 * 500)
param_grad.w = input.data * output.diff.';
% (800 * 500) * (500 * 100) = (800 * 100)
input_od = param.w * output.diff;
end
