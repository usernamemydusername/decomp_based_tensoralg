function [U_rep, S_rep, V_rep] = tsvd_ttd_dim3(G1, G2, G3, n1, n2, n3, tol)
%COMPUTE_SVD_AK_TTD_DIM3_730 Core-form TTD-based third-order T-SVD, with
% mode-2 also compressed onto an orthogonal basis (in addition to mode-1,
% which _729 already compresses via QR of G1). Third-order analogue of
% tsvd_ttd_dim4.m; since there is only ONE transform mode
% (mode 3), no combined frequency index is needed.
%
% Motivation / correctness: identical argument as in
% tsvd_ttd_dim4.m. In _729, the per-frequency reduced matrix
% Mbeta = Rmat*Abeta is r1-by-n2: mode 1 is compressed to rank r1 via QR
% of G1, but mode 2 is left at its full physical size n2 (R1 is an
% implicit n2 x n2 identity core), so each of the n3 per-frequency SVDs
% costs O(r1^2 n2). Since Abeta is, for every frequency, a linear
% combination of the SAME r2 fixed (r1 x n2) slices of G2, its mode-2
% columns live inside a shared subspace of dimension <= r1*r2 (TT
% network-cut bound: cutting both edges touching node 2 bounds its rank
% against the rest by the product of the two edge ranks r1*r2), regardless
% of frequency. We extract that shared subspace ONCE via a QR of the
% mode-2 unfolding of G2, then every per-frequency SVD operates on an
% r1-by-q matrix (q = min(n2, r1*r2)) instead of r1-by-n2.
%
% Exactness: write Mbeta = Abeta_reduced * Q2.', where Q2 (n2 x q) has
% orthonormal columns from the QR above. Since Q2.'*Q2 = I_q, Mbeta and
% Abeta_reduced share the same singular values and left singular vectors;
% if Abeta_reduced = Uhat*Shat*What', then Mbeta = Uhat*Shat*(Q2*What)' is
% a valid compact SVD of Mbeta. So V_beta = Q2*What_beta reproduces
% exactly the same T-SVD factors as _729, just computed more cheaply and
% left in factored (Q2, What) form.
%
% Output schema unchanged from _729 (TT cores, no full tensors formed):
%   U_rep.cores = {P1,P2,P3}
%   S_rep.cores = {Q1,Q2,P3}
%   V_rep.cores = {R1,R2,P3}
% The only structural change vs. _729: V_rep's R1 leaf is now the genuine
% mode-2 basis Q2 (n2 x q), replacing the n2 x n2 implicit identity core,
% and R2 is q x s x n3 instead of n2 x s x n3 -- mirroring exactly how
% U_rep already stores Q1 (n1 x r1) instead of an n1-sized identity.

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
