function [U_rep, S_rep, V_rep] = tsvd_ttd_dim3(G1, G2, G3, n1, n2, n3, tol)
%TSVD_TTD_DIM3  TTD-based third-order T-SVD, computed directly from the
% TT (tensor-train) cores of a tensor T in R^{n1 x n2 x n3}, without ever
% forming T or its per-frequency (mode-3 Fourier) frontal slices densely.
%
% Algorithm: mode 1 is compressed to its TT bond rank r1 via a QR of the
% first TT core; mode 2 is compressed to a rank-q orthogonal basis
% (q = min(n2, r1*r2)) via a QR of the mode-2 unfolding of the second TT
% core (exact, not an approximation: an isometry Q2 satisfies Q2.'*Q2=I,
% so a matrix M and its Q2-reduced form M*Q2 share singular values/left
% singular vectors); the transform mode (mode 3) is handled via a
% length-n3 FFT of the third TT core. Each of the n3 resulting frequency
% slices then only needs a compact SVD of a small r1-by-q matrix (instead
% of an r1-by-n2 matrix), giving the per-frequency T-SVD factors, which
% are returned in factored TT-core form (never reconstructed to a dense
% n1 x s x n3 tensor here).
%
% Inputs:
%   G1        [1  x n1 x r1] first TT core of T
%   G2        [r1 x n2 x r2] second TT core of T
%   G3        [r2 x n3 x 1 ] third (transform-mode) TT core of T
%   n1,n2,n3  ambient tensor dimensions
%   tol       (optional, default 1e-12) relative singular-value cutoff
%             used per frequency slice to decide the local rank
%
% Outputs: U_rep, S_rep, V_rep -- each a struct describing the
% corresponding T-SVD factor tensor (U, S, or V, size [n_mode x s x n3]
% once reconstructed) in TT-core form:
%   .format          'TTD'
%   .cores           {P1,P2,P3} (U_rep/S_rep/V_rep share the same P3;
%                    mode_sizes below distinguishes each tensor's own
%                    leading two core sizes)
%   .mode_sizes      [n1,s,n3] / [s,s,n3] / [n2,s,n3] for U/S/V respectively
%   .frequency_ranks [n3 x 1] local T-SVD rank kept at each frequency
%   .max_svd_rank    max(frequency_ranks) -- the rank s at which the
%                    cores are padded/zero-extended; also the largest
%                    order a caller can truncate a reduced model to
%
% Requires:
%   - qr, fft, svd (base MATLAB only -- no external toolbox)
%
% Notes:
%   - s = max_svd_rank is the TENSOR'S OWN numerical rank at tolerance
%     tol; a caller requesting a target order r should pass
%     r = min(r_requested, max_svd_rank) before reconstructing/truncating,
%     since asking for more than max_svd_rank adds no information.
%   - To get dense U/S/V, truncate cores{2} to the desired rank r, apply
%     an inverse FFT along mode 3, then left-multiply by the mode-1/
%     mode-2 leaf bases in cores{1} -- see tera_reduce.m for a worked
%     example of this reconstruction.

if nargin < 7 || isempty(tol)
    tol = 1e-12;
end

r1 = size(G1,3);
r2_ = size(G2,3);

assert(size(G1,1)==1 && size(G1,2)==n1, 'G1 must have size [1,n1,r1].');
assert(size(G2,1)==r1 && size(G2,2)==n2, 'G2 size mismatch.');
assert(size(G3,2)==n3 && size(G3,3)==1, 'G3 must have size [r2,n3,1].');

% FFT on the tube core (mode-2 of G3): [r2 x n3]
G3_mat = reshape(G3, [r2_, n3]);
G3_fft = fft(G3_mat, [], 2);

% QR of the matricized first core (compresses mode 1 -> r1).
G1_mat = reshape(G1, [n1, r1]);
[Qmat, Rmat] = qr(G1_mat, 0);

% ---- NEW: QR of the mode-2 unfolding of G2 (compresses mode 2 -> q) ----
% G2 is [r1,n2,r2]; permute to [n2,r1,r2] and flatten (a1,a2) into columns.
G2_mat2 = reshape(permute(G2,[2 1 3]), n2, r1*r2_);
[Q2, Rmat2] = qr(G2_mat2, 0);       % Q2: n2 x q, Rmat2: q x (r1*r2)
q = size(Q2,2);
% G2_reduced(a1,k,a2) = Rmat2(k, a1+r1*(a2-1)); recover via reshape/permute.
G2_reduced = permute(reshape(Rmat2.', r1, r2_, q), [1 3 2]);   % [r1,q,r2]
G2_flat = reshape(G2_reduced, r1*q, r2_);                      % was r1*n2 x r2 in _729

Ublk = cell(n3,1); Sblk = cell(n3,1); Vblk = cell(n3,1);
local_rank = zeros(n3,1);
smax = 0;

for beta = 1:n3
    g = G3_fft(:,beta);
    Abeta = reshape(G2_flat*g, r1, q);
    Mbeta = Rmat*Abeta;
    [Ub,Sb,Vb] = svd(Mbeta,'econ');

    sigma = diag(Sb);
    if isempty(sigma)
        sk = 0;
    else
        scale = max(sigma);
        sk = sum(sigma > tol*max(1,scale));
    end

    local_rank(beta) = sk;
    smax = max(smax, sk);
    Ublk{beta} = Ub(:,1:sk);
    Sblk{beta} = Sb(1:sk,1:sk);
    Vblk{beta} = Vb(:,1:sk);   % q x sk (NOT n2 x sk)
end

s = max(1, smax);
P2 = complex(zeros(r1, s, n3));
Q2c = complex(zeros(s, s, n3));
R2 = complex(zeros(q, s, n3));

for beta = 1:n3
    sk = local_rank(beta);
    if sk > 0
        P2(:,1:sk,beta) = Ublk{beta};
        Q2c(1:sk,1:sk,beta) = Sblk{beta};
        R2(:,1:sk,beta) = Vblk{beta};
    end
end

P3 = build_inverse_dft_tt_core_dim3(n3);
P1 = reshape(Qmat, 1, n1, r1);
Q1 = reshape(eye(s), 1, s, s);
R1 = reshape(Q2, 1, n2, q);   % genuine mode-2 basis, replaces the n2-identity core

U_rep = struct('format','TTD','cores',{{P1,P2,P3}}, ...
    'mode_sizes',[n1,s,n3], 'frequency_ranks',local_rank, 'max_svd_rank',s);
S_rep = struct('format','TTD','cores',{{Q1,Q2c,P3}}, ...
    'mode_sizes',[s,s,n3], 'frequency_ranks',local_rank, 'max_svd_rank',s);
V_rep = struct('format','TTD','cores',{{R1,R2,P3}}, ...
    'mode_sizes',[n2,s,n3], 'frequency_ranks',local_rank, 'max_svd_rank',s);
end

function P3 = build_inverse_dft_tt_core_dim3(n3)
% Single-mode inverse-DFT TT core: P3(beta,j3,1) = Finv_n3(j3,beta).
Finv3 = ifft(eye(n3));
P3 = complex(zeros(n3, n3, 1));
P3(:,:,1) = Finv3.';
end
