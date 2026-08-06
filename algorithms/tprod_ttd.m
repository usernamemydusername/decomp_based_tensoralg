function [G1C, G2C, G3C] = tprod_ttd(G1_A, G2_A, G3_A, G1_B, G2_B, G3_B, n1, n2, l, n3, tt_eps, tt_rmax)
%T_PRODUCT_TTD_FAST_2_V2 Numerically IDENTICAL to t_product_ttd_fast_2.m
% (same math, same output ranks r1C=r1A*r1B, r2C=r2A*r2B); only the core
% construction loops are rewritten as vectorized broadcasts instead of
% explicit MATLAB-level nested loops with squeeze() calls inside.
%
% Motivation: t_product_ttd_fast_2's G2C loop iterates
% a1=1:r1A, b1=1:r1B, a2=1:r2A, b2=1:r2B -- O(r1A*r1B*r2A*r2B) MATLAB
% scalar-loop iterations, each touching a length-l vector. r2A,r2B are
% small (bounded by n3), but r1A,r1B grow with the true rank of the
% inputs; for a near-full-rank sparse tensor at large n this becomes
% many millions of iterations and was observed to take >30 minutes at
% just n1=1024 (before even reaching any rank-rounding step). Since
% r1B,r2B are small, looping only over (b1,b2) -- not (a1,b1,a2,b2) -- and
% vectorizing the a1,a2,j dimensions via broadcasting reduces the loop
% trip count from r1A*r1B*r2A*r2B to r1B*r2B while touching the exact
% same total number of elements.
%
% See t_product_ttd_fast_2.m for the full derivation/comments; this file
% only changes HOW the same cores are computed.

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
