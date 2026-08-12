function [U_rep, S_rep, V_rep] = tsvd_ttd_dim4_ref(G1,G2,G3,G4,n1,n2,n3,n4,tol)
%TSVD_TTD_DIM4_REF  TTD-based fourth-order T-SVD -- reference/validation
% implementation, numerically equivalent to tsvd_ttd_dim4.m but WITHOUT
% that file's mode-2 compression (mode 2 is left at its full physical
% size n2 in the per-frequency reduced matrix, i.e. each of the
% Nf = n3*n4 per-frequency SVDs operates on an r1-by-n2 matrix instead of
% the smaller r1-by-q matrix tsvd_ttd_dim4.m uses).
%
% Role in this repo: NOT used to produce any reported timing/accuracy
% result. tsvd_dim4_timing.m calls this function once, at a small
% validation size (n1=n2=16), purely to confirm that tsvd_ttd_dim4.m's
% reconstructed tensor agrees with this simpler/slower implementation
% before using tsvd_ttd_dim4.m for the actual (much larger) timed sweep.
% Kept in this repo only so that self-check is reproducible.
%
% Inputs:
%   G1              [1  x n1 x r1] first TT core of T
%   G2              [r1 x n2 x r2] second TT core of T
%   G3              [r2 x n3 x r3] third (transform-mode) TT core of T
%   G4              [r3 x n4 x 1 ] fourth (transform-mode) TT core of T
%   n1,n2,n3,n4     ambient tensor dimensions
%   tol             (optional, default 1e-12) relative singular-value
%                   cutoff used per frequency pair to decide local rank
%
% Outputs: U_rep, S_rep, V_rep, same schema as tsvd_ttd_dim4.m (TT-core
% form; V_rep's mode-2 leaf here is an implicit n2 x n2 identity core
% rather than the compressed q x n2 basis tsvd_ttd_dim4.m returns -- the
% two reconstruct to the same dense tensor).
%
% Notes:
%   - Use tsvd_ttd_dim4.m for anything performance-sensitive; this file
%     exists purely as an independent cross-check.
%   - Requires no external toolbox beyond base MATLAB (qr, fft, svd).

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

% QR factorization of the first TT core.
G1mat = reshape(G1,n1,r1);
[Qmat,Rmat] = qr(G1mat,0);

% Reduced SVD at every frequency tuple.
G2flat = reshape(G2,r1*n2,r2);
Ublk = cell(n3,n4);
Sblk = cell(n3,n4);
Vblk = cell(n3,n4);
local_rank = zeros(n3,n4);
smax = 0;

for beta4 = 1:n4
    for beta3 = 1:n3
        g = tmp_fft(:,beta3,beta4);
        Abeta = reshape(G2flat*g,r1,n2);
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
        Vblk{beta3,beta4} = Vb(:,1:sk);
    end
end

% Use one zero singular channel only for the completely zero tensor.
s = max(1,smax);
Nf = n3*n4;
P2 = complex(zeros(r1,s,Nf));
Q2 = complex(zeros(s,s,Nf));
R2 = complex(zeros(n2,s,Nf));

for beta4 = 1:n4
    rho4 = beta4;
    for beta3 = 1:n3
        rho3 = beta3 + n3*(rho4-1);
        sk = local_rank(beta3,beta4);
        if sk > 0
            P2(:,1:sk,rho3) = Ublk{beta3,beta4};
            Q2(1:sk,1:sk,rho3) = Sblk{beta3,beta4};
            R2(:,1:sk,rho3) = Vblk{beta3,beta4};
        end
    end
end

[P3,P4] = build_inverse_dft_tt_cores(n3,n4);
P1 = reshape(Qmat,1,n1,r1);
Q1 = reshape(eye(s),1,s,s);
R1 = implicit_identity_tt_core(n2);

U_rep = struct('format','TTD','cores',{{P1,P2,P3,P4}}, ...
    'mode_sizes',[n1,s,n3,n4], 'tt_ranks',[1,r1,Nf,n4,1], ...
    'frequency_ranks',local_rank, 'max_svd_rank',s);
S_rep = struct('format','TTD','cores',{{Q1,Q2,P3,P4}}, ...
    'mode_sizes',[s,s,n3,n4], 'tt_ranks',[1,s,Nf,n4,1], ...
    'frequency_ranks',local_rank, 'max_svd_rank',s);
V_rep = struct('format','TTD','cores',{{R1,R2,P3,P4}}, ...
    'mode_sizes',[n2,s,n3,n4], 'tt_ranks',[1,n2,Nf,n4,1], ...
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

function core = implicit_identity_tt_core(n)
core = struct('type','identity_tt_core', ...
    'physical_size',n, 'rank_size',n, 'size',[1,n,n]);
end
