function [U_rep,S_rep,V_rep] = tsvd_htd_dim4(U1,U2,U3,U4,B12,B34,Broot,tol)
%COMPUTE_HTD_TSVD_DIM4_729_V2 Core-form HTD-based fourth-order T-SVD.
% Numerically IDENTICAL to compute_htd_tsvd_dim4_729.m (same output
% schema, same math); only the two construction steps that were
% previously explicit nested for-loops are rewritten as matrix
% multiplications (matricize + BLAS), so the cost scales with a single
% dgemm chain instead of a MATLAB-level loop over r12*r34 or r3*r4
% iterations.
%
% Motivation: once the driver's rank selection became tolerance-adaptive
% (no longer artificially capped), the achieved ranks r12/r34/r3/r4 grow
% with the true rank of the data. compute_htd_tsvd_dim4_729's original
% G-construction loop (over a=1:r12,b=1:r34) and per-frequency Gbeta loop
% (over a3=1:r3,a4=1:r4, repeated n3*n4 times) then dominate runtime --
% observed to make HTD-based T-SVD slower than the definition-based
% method at n1=256 in the sparse case even after the rank-cap fix. Both
% loops are pure tensor contractions and are mathematically identical to
% a reshape + matrix-multiply chain; only the mechanism to compute them
% changes.
%
% Each output has fields:
%   leaves    : {L1,L2,L3,L4}
%   transfers : struct with nodes B12, B34, Broot

if nargin < 8 || isempty(tol)
    tol = 1e-12;
end

[n1,r1] = size(U1);
[n2,r2] = size(U2);
[n3,r3] = size(U3);
[n4,r4] = size(U4);
r12 = size(B12,3);
r34 = size(B34,3);

assert(size(B12,1)==r1 && size(B12,2)==r2, 'B12 size mismatch.');
assert(size(B34,1)==r3 && size(B34,2)==r4, 'B34 size mismatch.');
assert(size(Broot,1)==r12 && size(Broot,2)==r34, 'Broot size mismatch.');

% Contract the input HTD transfer tensors to the reduced four-way core.
% (was: double loop over a=1:r12, b=1:r34, accumulating
%  G = G + Broot(a,b,1) .* (B12(:,:,a) .* B34(:,:,b)) )
Broot2 = reshape(Broot, r12, r34);
B12mat = reshape(B12, r1*r2, r12);
B34mat = reshape(B34, r3*r4, r34);
G = reshape(B12mat * Broot2 * B34mat.', r1, r2, r3, r4);

U3hat = fft(U3,[],1);
U4hat = fft(U4,[],1);

% Per-frequency reduced core Gf(:,:,beta3,beta4), for all (beta3,beta4)
% at once via matricized contraction against U3hat/U4hat.
% (was: for beta4, for beta3, for a3, for a4 -- O(n3*n4*r3*r4) MATLAB loop)
tmp = reshape(G, r1*r2, r3, r4);
tmp = permute(tmp, [1 3 2]);            % (r1*r2, r4, r3)
tmp = reshape(tmp, r1*r2*r4, r3);
tmp = tmp * U3hat.';                    % (r1*r2*r4, n3)
tmp = reshape(tmp, r1*r2, r4, n3);
tmp = permute(tmp, [1 3 2]);            % (r1*r2, n3, r4)
tmp = reshape(tmp, r1*r2*n3, r4);
tmp = tmp * U4hat.';                    % (r1*r2*n3, n4)
Gf = reshape(tmp, r1, r2, n3, n4);      % Gf(:,:,beta3,beta4)

Ublk = cell(n3,n4);
Sblk = cell(n3,n4);
Vblk = cell(n3,n4);
local_rank = zeros(n3,n4);
smax = 0;

for beta4 = 1:n4
    for beta3 = 1:n3
        Gbeta = Gf(:,:,beta3,beta4);

        [Ub,Sb,Vb] = svd(Gbeta,'econ');
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
        Vblk{beta3,beta4} = Vb(:,1:sk);
    end
end

s = max(1,smax);
Nf = n3*n4;
Finv3 = ifft(eye(n3));
Finv4 = ifft(eye(n4));

% Root transfer tensors. The first root index is the paired index at {1,2};
% the second root index is rho_f = beta3+n3*(beta4-1).
BrootU = complex(zeros(r1*s,Nf,1));
BrootS = complex(zeros(s*s,Nf,1));
BrootV = complex(zeros(r2*s,Nf,1));

% (was: quadruple-nested loop over beta3,beta4,xi,{alpha1|alpha2|zeta},
%  assigning one scalar at a time -- O(n3*n4*sk*(r1+r2+sk)) MATLAB-level
%  scalar loop, which dominated runtime once sk grew with the true rank.)
% phiU = alpha1+(xi-1)*r1 is exactly the column-major linear index of the
% (r1 x sk) block Ublk{...}, since r1 (the leaf width) does not depend on
% frequency -- so the whole block can be written in one vectorized
% assignment. Same for BrootV. BrootS's stride uses the GLOBAL s (not the
% local sk), so its sk x sk block is first embedded into an s x s zero
% matrix before flattening.
for beta4 = 1:n4
    for beta3 = 1:n3
        rhoF = beta3 + n3*(beta4-1);
        sk = local_rank(beta3,beta4);
        if sk > 0
            BrootU(1:r1*sk, rhoF, 1) = reshape(Ublk{beta3,beta4}, r1*sk, 1);
            BrootV(1:r2*sk, rhoF, 1) = reshape(Vblk{beta3,beta4}, r2*sk, 1);
            Sfull = complex(zeros(s,s));
            Sfull(1:sk,1:sk) = Sblk{beta3,beta4};
            BrootS(:, rhoF, 1) = reshape(Sfull, s*s, 1);
        end
    end
end

% The nonzero pattern of B12 and B34 is deterministic, so store it as an
% exact implicit selector rather than a dense mostly-zero tensor.
B12U = pairing_transfer(r1,s,r1);
B12S = pairing_transfer(s,s,s);
B12V = pairing_transfer(r2,s,r2);
B34shared = pairing_transfer(n3,n4,n3);

U_rep = struct('format','HTD','tree','{1,2}|{3,4}', ...
    'leaves',{{U1,eye(s),Finv3,Finv4}}, ...
    'transfers',struct('B12',B12U,'B34',B34shared,'Broot',BrootU), ...
    'mode_sizes',[n1,s,n3,n4], 'frequency_ranks',local_rank, ...
    'max_svd_rank',s);

S_rep = struct('format','HTD','tree','{1,2}|{3,4}', ...
    'leaves',{{eye(s),eye(s),Finv3,Finv4}}, ...
    'transfers',struct('B12',B12S,'B34',B34shared,'Broot',BrootS), ...
    'mode_sizes',[s,s,n3,n4], 'frequency_ranks',local_rank, ...
    'max_svd_rank',s);

V_rep = struct('format','HTD','tree','{1,2}|{3,4}', ...
    'leaves',{{U2,eye(s),Finv3,Finv4}}, ...
    'transfers',struct('B12',B12V,'B34',B34shared,'Broot',BrootV), ...
    'mode_sizes',[n2,s,n3,n4], 'frequency_ranks',local_rank, ...
    'max_svd_rank',s);
end

function T = pairing_transfer(left_size,right_size,left_multiplier)
% Nonzeros satisfy T(a,b,a+left_multiplier*(b-1)) = 1.
T = struct('type','pairing_transfer', ...
    'left_size',left_size, 'right_size',right_size, ...
    'output_size',left_size*right_size, ...
    'left_multiplier',left_multiplier, ...
    'size',[left_size,right_size,left_size*right_size]);
end
