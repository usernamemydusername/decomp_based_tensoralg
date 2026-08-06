function [U_rep,S_rep,V_rep] = tsvd_htd_dim3(U1,U2,U3,B12,Broot,tol)
%COMPUTE_HTD_TSVD_DIM3_729 Core-form HTD/TD-based third-order T-SVD.
% Implements Proposition (HTD/TD-based T-SVD) in mainCC_260729.tex
% (prop:htd_tsvd) directly: since third-order HTD reduces to standard
% Tucker format, the T-SVD factor tensors are represented as a small
% frequency-domain core plus mode-1 (and, for V, mode-1) factor matrices --
% NOT as a full n1 x s x n3 (etc.) tensor.
%
% Input: U1 [n1 x r1], U2 [n2 x r2], U3 [n3 x r3], B12 [r1 x r2 x r12],
%        Broot [r12 x r3] (same HTD/TD convention as compute_htd_tsvd_dim3.m).
%
% Output schema (Tucker-style core-form; NOT reconstructed):
%   U_rep.leaf = U1 (n1 x r1);           U_rep.core = Ptilde (r1 x s x n3, frequency-domain)
%   S_rep.leaf = [] (mode-1 is I_s, trivial); S_rep.core = Qtilde (s x s x n3, frequency-domain)
%   V_rep.leaf = U2 (n2 x r2);           V_rep.core = Rtilde (r2 x s x n3, frequency-domain)
% Reconstruction (validation/consumer-only): apply ifft along mode 3 to the
% core, then a mode-1 tensor-matrix product with the leaf (trivial/no-op
% for S). This is NOT done here.

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
