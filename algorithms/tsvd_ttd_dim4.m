function [U_rep, S_rep, V_rep] = tsvd_ttd_dim4(G1,G2,G3,G4,n1,n2,n3,n4,tol)
%COMPUTE_SVD_AK_TTD_DIM4_730 Core-form TTD-based fourth-order T-SVD, with
% mode-2 also compressed onto an orthogonal basis (in addition to mode-1,
% which _729 already compresses via QR of G1).
%
% Motivation: in _729, the per-frequency reduced matrix M_beta = R1*A_beta
% is r1-by-n2 -- mode 1 is compressed to rank r1, but mode 2 is left at
% its full physical size n2, so every one of the Nf = n3*n4 per-frequency
% SVDs costs O(r1^2 n2). Since A_beta = sum_{a2} G2(:,:,a2) g_beta(a2) is,
% for every frequency, a linear combination of the SAME r2 fixed
% (r1 x n2) slices of G2, its "mode-2 columns" live inside a shared
% subspace of dimension <= r1*r2 (a standard TT network-cut bound: cutting
% both edges touching node 2 bounds node 2's rank against the rest by the
% product of the two edge ranks r1*r2), regardless of frequency. We
% extract that shared subspace ONCE via a QR of the mode-2 unfolding of
% G2, then every per-frequency SVD operates on an r1-by-q matrix
% (q = min(n2, r1*r2), independent of n2) instead of r1-by-n2.
%
% Correctness (exact, not an approximation): write M_beta = A_beta_reduced
% * Q2.', where Q2 (n2 x q) has orthonormal columns from the QR above.
% Since Q2.'*Q2 = I_q (i.e. Q2.' has orthonormal ROWS), M_beta*M_beta' =
% A_beta_reduced*A_beta_reduced', so M_beta and A_beta_reduced share the
% same singular values and left singular vectors; if A_beta_reduced =
% Uhat*Shat*What', then M_beta = Uhat*Shat*(Q2*What)' is a valid compact
% SVD of M_beta (Q2*What has orthonormal columns because Q2 and What do).
% So V_beta = Q2*What_beta reproduces exactly the same T-SVD factors as
% _729, just computed more cheaply and left in factored (Q2, What) form.
%
% Output schema unchanged from _729 (TT cores, no full tensors formed):
%   U_rep.cores = {P1,P2,P3,P4}
%   S_rep.cores = {Q1,Q2c,P3,P4}
%   V_rep.cores = {R1,R2,P3,P4}
% The only structural change vs. _729: V_rep's R1 leaf is now the genuine
% mode-2 basis Q2 (n2 x q), replacing the n2 x n2 implicit identity core,
% and R2 is q x s x Nf instead of n2 x s x Nf -- mirroring exactly how
% U_rep already stores Q1 (n1 x r1) instead of an n1-sized identity.

if nargin < 9 || isempty(tol)
    tol = 1e-12;
end

validateattributes(G1, {'numeric'}, {'3d','nonempty'});
validateattributes(G2, {'numeric'}, {'3d','nonempty'});
validateattributes(G3, {'numeric'}, {'3d','nonempty'});
validateattributes(G4, {'numeric'}, {'3d','nonempty'});

r1 = size(G1,3);
r2 = size(G2,3);
r3 = size(G3,3);

assert(size(G1,1)==1 && size(G1,2)==n1, 'G1 must have size [1,n1,r1].');
assert(size(G2,1)==r1 && size(G2,2)==n2 && size(G2,3)==r2, 'G2 size mismatch.');
assert(size(G3,1)==r2 && size(G3,2)==n3 && size(G3,3)==r3, 'G3 size mismatch.');
assert(size(G4,1)==r3 && size(G4,2)==n4 && size(G4,3)==1, 'G4 must have size [r3,n4,1].');

% Contract the transform-mode input cores, then apply the joint FFT.
G3mat = reshape(permute(G3,[2 1 3]), n3*r2, r3);
G4mat = reshape(G4, r3, n4);
tmp = reshape(G3mat*G4mat, n3, r2, n4);
tmp = permute(tmp,[2 1 3]);                         % [r2,n3,n4]
tmp_fft = fft(fft(tmp,[],2),[],3);                  % indexed by (beta3,beta4)

% QR factorization of the first TT core (compresses mode 1 -> r1).
G1mat = reshape(G1,n1,r1);
[Qmat,Rmat] = qr(G1mat,0);

% ---- NEW: QR of the mode-2 unfolding of G2 (compresses mode 2 -> q) ----
% G2 is [r1,n2,r2]; permute to [n2,r1,r2] and flatten (a1,a2) into columns.
G2_mat2 = reshape(permute(G2,[2 1 3]), n2, r1*r2);
[Q2, Rmat2] = qr(G2_mat2, 0);       % Q2: n2 x q, Rmat2: q x (r1*r2)
q = size(Q2,2);
% G2_reduced(a1,k,a2) = Rmat2(k, a1+r1*(a2-1)); recover via reshape/permute.
G2_reduced = permute(reshape(Rmat2.', r1, r2, q), [1 3 2]);   % [r1,q,r2]
G2flat = reshape(G2_reduced, r1*q, r2);                       % was r1*n2 x r2 in _729

% Reduced SVD at every frequency tuple (now r1 x q instead of r1 x n2).
Ublk = cell(n3,n4);
Sblk = cell(n3,n4);
Vblk = cell(n3,n4);
local_rank = zeros(n3,n4);
smax = 0;

for beta4 = 1:n4
    for beta3 = 1:n3
        g = tmp_fft(:,beta3,beta4);
        Abeta = reshape(G2flat*g,r1,q);
        Mbeta = Rmat*Abeta;
        [Ub,Sb,Vb] = svd(Mbeta,'econ');

        sigma = diag(Sb);
        if isempty(sigma)
            sk = 0;
        else
            scale = max(sigma);
            sk = sum(sigma > tol*max(1,scale));
        end

        local_rank(beta3,beta4) = sk;
        smax = max(smax,sk);
        Ublk{beta3,beta4} = Ub(:,1:sk);
        Sblk{beta3,beta4} = Sb(1:sk,1:sk);
        Vblk{beta3,beta4} = Vb(:,1:sk);   % q x sk (NOT n2 x sk)
    end
end

% Use one zero singular channel only for the completely zero tensor.
s = max(1,smax);
Nf = n3*n4;
P2 = complex(zeros(r1,s,Nf));
Q2c = complex(zeros(s,s,Nf));
R2 = complex(zeros(q,s,Nf));

for beta4 = 1:n4
    rho4 = beta4;
    for beta3 = 1:n3
        rho3 = beta3 + n3*(rho4-1);
        sk = local_rank(beta3,beta4);
        if sk > 0
            P2(:,1:sk,rho3) = Ublk{beta3,beta4};
            Q2c(1:sk,1:sk,rho3) = Sblk{beta3,beta4};
            R2(:,1:sk,rho3) = Vblk{beta3,beta4};
        end
    end
end

[P3,P4] = build_inverse_dft_tt_cores(n3,n4);
P1 = reshape(Qmat,1,n1,r1);
Q1 = reshape(eye(s),1,s,s);
R1 = reshape(Q2,1,n2,q);   % genuine mode-2 basis, replaces the n2-identity core

U_rep = struct('format','TTD','cores',{{P1,P2,P3,P4}}, ...
    'mode_sizes',[n1,s,n3,n4], 'tt_ranks',[1,r1,Nf,n4,1], ...
    'frequency_ranks',local_rank, 'max_svd_rank',s);
S_rep = struct('format','TTD','cores',{{Q1,Q2c,P3,P4}}, ...
    'mode_sizes',[s,s,n3,n4], 'tt_ranks',[1,s,Nf,n4,1], ...
    'frequency_ranks',local_rank, 'max_svd_rank',s);
V_rep = struct('format','TTD','cores',{{R1,R2,P3,P4}}, ...
    'mode_sizes',[n2,s,n3,n4], 'tt_ranks',[1,q,Nf,n4,1], ...
    'frequency_ranks',local_rank, 'max_svd_rank',s);
end

function [core3,core4] = build_inverse_dft_tt_cores(n3,n4)
Finv3 = ifft(eye(n3));
Finv4 = ifft(eye(n4));

core3 = complex(zeros(n3*n4,n3,n4));
for beta4 = 1:n4
    for beta3 = 1:n3
        rho3 = beta3 + n3*(beta4-1);
        core3(rho3,:,beta4) = Finv3(:,beta3).';
    end
end

core4 = complex(zeros(n4,n4,1));
for beta4 = 1:n4
    core4(beta4,:,1) = Finv4(:,beta4).';
end
end
