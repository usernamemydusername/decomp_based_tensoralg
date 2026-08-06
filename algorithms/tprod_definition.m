function C = tprod_definition(A, B)
    % Compute the T-product of two tensors A and B
    % Inputs:
    %   A - Tensor of size n1 x n2 x n3
    %   B - Tensor of size n2 x l x n3
    % Output:
    %   C - Tensor of size n1 x l x n3 (T-product of A and B)

    % Get dimensions
    [n1, n2, n3] = size(A);
    [~, l, ~] = size(B);

    bcirc_A = bcirc(A);

    unfold_B = unfold(B);
    product = bcirc_A * unfold_B;
    C = fold(product, n1, l, n3);
end

function mat = unfold(B)
    % Unfold tensor B into a matrix
    % Input:
    %   B - Tensor of size n2 x l x n3
    % Output:
    %   mat - Matrix of size (n2*n3) x l

    [n2, l, n3] = size(B);
    mat = zeros(n2 * n3, l);

    for i = 1:n3
        mat((i-1)*n2+1:i*n2, :) = B(:, :, i);
    end
end

function tensor = fold(mat, n1, l, n3)
    % Fold a matrix into a tensor
    % Input:
    %   mat - Matrix of size (n1*n3) x l
    %   n1, l, n3 - Dimensions of the resulting tensor
    % Output:
    %   tensor - Tensor of size n1 x l x n3

    tensor = zeros(n1, l, n3);
    for i = 1:n3
        tensor(:, :, i) = mat((i-1)*n1+1:i*n1, :);
    end
end