function [U_rep,S_rep,V_rep] = tsvd_htd_dim3(U1,U2,U3,B12,Broot,tol)
%TSVD_HTD_DIM3  HTD/Tucker-based third-order T-SVD, computed directly
% from the HTD (hierarchical Tucker) factors of a tensor T in
% R^{n1 x n2 x n3}, without forming T or its frontal slices densely.
%
% Algorithm: third-order HTD reduces to a plain Tucker decomposition
% (leaf factors U1,U2 on modes 1-2, small core B12/Broot). The transform
% mode (mode 3) is handled by FFT-ing U3 and contracting it against the
% small core to get a size-r1-by-r2 "reduced slice" G_beta per frequency
% beta = 1..n3; each G_beta then only needs a compact SVD of that small
% matrix. The T-SVD factors are returned as leaf + frequency-domain-core
% pairs (Tucker-style core-form), never reconstructed to a dense tensor
% here.
%
% Inputs:
%   U1      [n1 x r1] mode-1 leaf factor (orthonormal columns)
%   U2      [n2 x r2] mode-2 leaf factor (orthonormal columns)
%   U3      [n3 x r3] mode-3 (transform-mode) leaf factor
%   B12     [r1 x r2 x r12] mode-{1,2} transfer core
%   Broot   [r12 x r3] root transfer matrix
%   tol     (optional, default 1e-12) relative singular-value cutoff used
%           per frequency slice to decide the local rank
%
% Outputs: U_rep, S_rep, V_rep -- each a struct describing the
% corresponding T-SVD factor tensor in Tucker-style core-form:
%   .format          'HTD_TD'
%   .leaf            U1 (for U_rep), [] (for S_rep, mode-1 is trivially
%                    I_s), or U2 (for V_rep)
%   .core            frequency-domain core, size [r1 x s x n3] /
%                    [s x s x n3] / [r2 x s x n3] for U/S/V respectively
%   .mode_sizes      [n1,s,n3] / [s,s,n3] / [n2,s,n3]
%   .frequency_ranks [n3 x 1] local T-SVD rank kept at each frequency
%   .max_svd_rank    max(frequency_ranks)
%
% Requires:
%   - fft, svd (base MATLAB only -- no external toolbox)
%
% Notes:
%   - s = max_svd_rank is the tensor's own numerical rank at tolerance
%     tol; a caller requesting target order r should pass
%     r = min(r_requested, max_svd_rank).
%   - To get dense U/S/V: truncate .core to rank r, inverse-FFT along the
%     frequency dimension, then left-multiply by .leaf (no-op for S).

if nargin < 6 || isempty(tol)
    tol = 1e-12;
end

[n1,r1] = size(U1); %#ok<ASGLU>
[n2,r2] = size(U2); %#ok<ASGLU>
[n3,r3] = size(U3);
r12 = size(B12,3);
assert(all(size(Broot)==[r12,r3]), 'Broot must be (r12 x r3).');

% Small core G(r1,r2,r3): G(:,:,p) = sum_a B12(:,:,a)*Broot(a,p)
G = zeros(r1,r2,r3);
for p = 1:r3
    for a = 1:r12
        G(:,:,p) = G(:,:,p) + B12(:,:,a) * Broot(a,p);
    end
end

C3 = fft(U3, [], 1);   % [n3 x r3], frequency weights

Ublk = cell(n3,1); Sblk = cell(n3,1); Vblk = cell(n3,1);
local_rank = zeros(n3,1);
smax = 0;

for beta = 1:n3
    Gbeta = zeros(r1,r2);
    for p = 1:r3
        Gbeta = Gbeta + G(:,:,p) * C3(beta,p);
    end

    [Ub,Sb,Vb] = svd(Gbeta,'econ');
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
    Vblk{beta} = Vb(:,1:sk);
end

s = max(1, smax);
Ptilde = complex(zeros(r1, s, n3));
Qtilde = complex(zeros(s, s, n3));
Rtilde = complex(zeros(r2, s, n3));

for beta = 1:n3
    sk = local_rank(beta);
    if sk > 0
        Ptilde(:,1:sk,beta) = Ublk{beta};
        Qtilde(1:sk,1:sk,beta) = Sblk{beta};
        Rtilde(:,1:sk,beta) = Vblk{beta};
    end
end

U_rep = struct('format','HTD_TD','leaf',U1, 'core',Ptilde, ...
    'mode_sizes',[n1,s,n3], 'frequency_ranks',local_rank, 'max_svd_rank',s);
S_rep = struct('format','HTD_TD','leaf',[], 'core',Qtilde, ...
    'mode_sizes',[s,s,n3], 'frequency_ranks',local_rank, 'max_svd_rank',s);
V_rep = struct('format','HTD_TD','leaf',U2, 'core',Rtilde, ...
    'mode_sizes',[n2,s,n3], 'frequency_ranks',local_rank, 'max_svd_rank',s);
end
