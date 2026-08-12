function [G1C, G2C, G3C] = tprod_ttd(G1_A, G2_A, G3_A, G1_B, G2_B, G3_B, n1, n2, l, n3, tt_eps, tt_rmax)
%TPROD_TTD  TTD-based T-product of two third-order tensors, computed
% directly from their TT (tensor-train) cores, without forming either
% input tensor or the block-circulant matrix densely.
%
% Algorithm: the transform-mode (mode 3) cores of A and B are FFT'd and
% multiplied pointwise per frequency (T-product is elementwise matrix
% multiplication in the transform domain); the mode-1/mode-2 cores are
% combined via a bond-dimension contraction (M(a1,b1,a2) = sum_p
% G2_A(a1,p,a2) G1_B(1,p,b1)) so the product tensor's TT ranks are
% exactly r1C = r1A*r1B and r2C = r2A*r2B. The construction loops are
% vectorized over the (small, n3-bounded) output ranks r2A,r2B rather
% than looped naively over all four input ranks, which is what keeps
% this fast even once r1A,r1B grow large with the true rank of the data.
%
% Inputs:
%   G1_A,G2_A,G3_A   TT cores of A: [1 x n1 x r1A], [r1A x n2 x r2A],
%                    [r2A x n3 x 1]
%   G1_B,G2_B,G3_B   TT cores of B: [1 x n2 x r1B], [r1B x l x r2B],
%                    [r2B x n3 x 1]
%   n1,n2,l,n3       ambient dimensions (A is n1 x n2 x n3, B is
%                    n2 x l x n3, output C is n1 x l x n3)
%   tt_eps,tt_rmax   (optional, defaults 1e-10, 50) accepted for API
%                    compatibility; not currently used to round the
%                    output cores
%
% Outputs:
%   G1C,G2C,G3C   TT cores of C = A*B (T-product): [1 x n1 x r1C],
%                 [r1C x l x r2C], [r2C x n3 x 1], with r1C = r1A*r1B,
%                 r2C = r2A*r2B
%
% Notes:
%   - Requires the TT-Toolbox convention for TT-core layout on the input
%     side (see extract_tt_cores_dim3 in the calling driver for how to
%     obtain G1_A/G2_A/G3_A from a tt_tensor object).
%   - Output ranks are NOT rounded/truncated -- for inputs with already
%     large ranks, r1C/r2C can grow multiplicatively; round the result
%     externally (e.g. via tt_tensor + round()) if a compressed output is
%     needed for further use.

if nargin < 11 || isempty(tt_eps),  tt_eps  = 1e-10; end
if nargin < 12 || isempty(tt_rmax), tt_rmax = 50;   end

r1A = size(G1_A, 3);  r2A = size(G2_A, 3);
r1B = size(G1_B, 3);  r2B = size(G2_B, 3);

G3A_fft = reshape(fft(G3_A, [], 2), r2A, n3);
G3B_fft = reshape(fft(G3_B, [], 2), r2B, n3);

% Core 1: G1C(1,:,(a1-1)*r1B+b1) = G1_A(1,:,a1) for all b1 -- each "page"
% of G1_A repeated r1B times consecutively (was: double loop over a1,b1).
G1C = repelem(G1_A, 1, 1, r1B);

% M(a1,b1,a2) = sum_p G2_A(a1,p,a2) * G1_B(1,p,b1) -- identical to the
% original (r2A is small, this part was never the bottleneck).
M = zeros(r1A, r1B, r2A);
G1B_mat = squeeze(G1_B(1,:,:));  % (n2 x r1B)
for a2 = 1:r2A
    GA = squeeze(G2_A(:,:,a2));  % (r1A x n2)
    M(:,:,a2) = GA * G1B_mat;
end

% Core 2: was a quadruple loop; now looped only over (b1,b2) -- both
% small -- with the (a1, j, a2) block computed via one broadcast per
% (b1,b2) pair instead of one scalar+vector-add per (a1,b1,a2,b2).
G2C = zeros(r1A*r1B, l, r2A*r2B);
for b1 = 1:r1B
    Mb1 = reshape(M(:,b1,:), r1A, r2A);       % (r1A x r2A)
    for b2 = 1:r2B
        Gb1b2 = reshape(G2_B(b1,:,b2), 1, l); % (1 x l)
        block = reshape(Mb1, r1A, 1, r2A) .* reshape(Gb1b2, 1, l, 1); % (r1A x l x r2A)
        G2C(b1:r1B:end, :, b2:r2B:end) = block;
    end
end

% Core 3 in Fourier domain (r2A,r2B small; unchanged).
G3C_fft = zeros(r2A*r2B, n3, 1);
for a2 = 1:r2A
    for b2 = 1:r2B
        idx = (a2-1)*r2B + b2;
        G3C_fft(idx,:,1) = (G3A_fft(a2,:) .* G3B_fft(b2,:));
    end
end
G3C = ifft(G3C_fft, [], 2);
end
