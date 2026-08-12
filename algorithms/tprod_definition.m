function C = tprod_definition(A, B)
%TPROD_DEFINITION  Definition-based (block-circulant) T-product of two
% third-order tensors: C = A * B in the T-product sense.
%
% Algorithm: forms the full block-circulant matrix of A (size
% (n1*n3) x (n2*n3)) and the block-column unfolding of B (size
% (n2*n3) x l), multiplies them with a single dense matrix product, and
% folds the result back into a tensor. This is the textbook/baseline
% T-product implementation -- O(n1*n2*n3*l) dense FLOPs and O((n1*n3)*(n2*n3))
% memory for the explicit block-circulant matrix -- used as the accuracy
% and timing baseline against tprod_ttd.m / tprod_htd.m.
%
% Inputs:
%   A   [n1 x n2 x n3] tensor
%   B   [n2 x l  x n3] tensor
%
% Outputs:
%   C   [n1 x l x n3] tensor, the T-product of A and B
%
% Requires:
%   - tproduct toolbox: bcirc
%
% Notes:
%   - Not a good fit for large n1*n3 (the dense block-circulant matrix
%     grows quadratically); TTD-/HTD-based T-product avoid ever forming
%     it when A, B are (approximately) low-rank.

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