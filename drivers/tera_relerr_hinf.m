%% ---------------------------------------------------------------
%  T-ERA relative H-infinity error table (3 cases x 3 methods = 9 numbers):
%  Definition-based vs TTD-based vs HTD-based Hankel T-SVD backends.
%
%  Successor to Test_TERA_3_804_hinf.m, replacing its comparison
%  methodology entirely for the reason below.
%
%  Why _804_hinf's Sparse-case numbers are not meaningful:
%  _804_hinf built each method's reduced (A,B,C) via T_ERA_fast_730 and
%  compared them through resolvent/H-infinity evaluation. B_red/C_red are
%  extracted from H's (1:l,1:m,:) corner block -- correct ERA theory for a
%  REAL system (that corner literally equals the first Markov parameter
%  C*B), but Sparse/LowTT/LowHT's H here is a synthetic tensor fed
%  directly as a Hankel-matrix surrogate (see gen_H_case_prebuilt below),
%  never derived from any real (A,B,C). For the Sparse case specifically
%  (target_nnz=30 spread over nRows*nCols*s ~ 9e8 entries at H=10000),
%  that l x m x s corner is essentially always empty (~0.3% chance of any
%  hit per trial, verified empirically), so B_red/C_red there are built
%  from pure SVD round-off noise and the resulting relative error
%  saturates at an uninformative 1.0.
%
%  This driver instead measures T-SVD reconstruction fidelity directly
%  against the FULL tensor H, sidestepping the corner-block issue AND the
%  need to match state-space order/basis across methods (which is what
%  made a naive A_red/B_red/C_red Frobenius comparison crash in
%  Test_TERA_3_804.m, dimension mismatch between the 't' baseline's fixed
%  order r=min(nRows,nCols)-k and ttd/htd's own numerical rank):
%    For each method m, reconstruct Hhat_m(:,:,beta) = Uhat*Shat*Vhat' in
%    the Fourier (mode-3) domain at that method's own truncation order,
%    for beta=1..s, and compute
%        relerr_m = max_beta ||Ahat(:,:,beta) - Hhat_m(:,:,beta)||_2
%                   / max_beta ||Ahat(:,:,beta)||_2
%    (spectral/operator norm -- a genuine H-infinity distance, sup over
%    frequency of the largest singular value of the error). No ERA
%    (A,B,C) construction, no corner extraction, no separate high-order
%    reference model is needed (each method is compared directly against
%    the exact input tensor H) -- this also removes one of the two
%    full tsvd(H) calls _804_hinf needed per (case,trial), so this driver
%    is cheaper as well as more meaningful.
%
%  For the 't' (Definition-based) method, tsvd(H) is still a full economy
%  T-SVD, but the reconstruction is truncated to that method's own
%  numerically effective rank (detected from the decay of its singular
%  values) rather than the requested fixed order k -- mathematically
%  identical (singular values past the effective rank contribute ~0 to
%  the reconstruction) but avoids forming an unnecessarily high-rank
%  (~2000) dense reconstruction when the tensor's true rank is ~30.
%
%  Sparse case generation (target_nnz=30, independent of tensor size) and
%  opts/n/s/H/k_frac/ntrials are unchanged from Test_TERA_3_804_hinf.m.
%  ---------------------------------------------------------------

clear; clc;

cd(fileparts(mfilename('fullpath')));
addpath(genpath(pwd));

n = 100;
s = 9;
H = 10000;

k_frac = 0.8;
case_names = {'Sparse', 'LowTT', 'LowHT'};
numCases = numel(case_names);
ntrials = 3;

opts = struct();
opts.svd_tol = 1e-12;
opts.tt_eps  = 1e-10;
opts.tt_rmax = 1e5;
opts.ht_r1   = 1e5;
opts.ht_r2   = 1e5;
opts.ht_r3   = min(9,s);
opts.ht_r12  = 1e5;

mem_budget_gb = 100;

method_names = {'Definition-based','TTD-based','HTD-based'};
method_tags  = {'t','ttd','htd'};
numMethods = numel(method_tags);

[l, L] = pick_lL(H);
m = l; T = L;
nRows = l*(L+1); nCols = m*(T+1);
k = min(round(k_frac * min(nRows,nCols)), min(nRows,nCols)-1);
r_normal = min(nRows,nCols) - k;

RelErr = nan(numCases, numMethods, ntrials);
RelErr_TTDvsDef = nan(numCases, ntrials);
RelErr_HTDvsDef = nan(numCases, ntrials);

for trial = 1:ntrials
    for c = 1:numCases
        fprintf('\n=== Case: %s | trial %d/%d ===\n', case_names{c}, trial, ntrials);
        Hpack = gen_H_case_prebuilt(c, nRows, nCols, s, opts);
        Hten = Hpack.full.H;

        Ahat = fft(Hten, [], 3);
        max_full = 0;
        for beta = 1:s
            max_full = max(max_full, svds(Ahat(:,:,beta), 1));
        end

        r1_tt = size(Hpack.ttd.tt_cores.G1, 3);
        mem_gb_ttd = s * r1_tt^2 * 16 / 1e9;
        r1_ht = size(Hpack.htd.ht_factors.U1, 2);
        mem_gb_htd = s * r1_ht^2 * 16 / 1e9;

        recon = struct();

        for mIdx = 1:numMethods
            tag = method_tags{mIdx};
            switch tag
                case 't'
                    [U,S,V] = tsvd(Hten);
                    r_req = min(k, size(U,2));
                    Shat_full = fft(S(1:r_req,1:r_req,:), [], 3);
                    r_eff = 1;
                    for beta = 1:s
                        sv = abs(diag(Shat_full(:,:,beta)));
                        r_eff = max(r_eff, sum(sv > opts.svd_tol * max(sv)));
                    end
                    r_eff = min(r_eff, r_req);
                    U1 = U(:,1:r_eff,:); S1 = S(1:r_eff,1:r_eff,:); V1 = V(:,1:r_eff,:);
                    r_used = r_eff;

                case 'ttd'
                    if mem_gb_ttd > mem_budget_gb
                        fprintf('  %-18s : SKIPPED -- estimated memory %.1f GB > budget %.1f GB (TT rank r1=%d)\n', ...
                            method_names{mIdx}, mem_gb_ttd, mem_budget_gb, r1_tt);
                        RelErr(c, mIdx, trial) = NaN;
                        continue;
                    end
                    G1 = Hpack.ttd.tt_cores.G1; G2 = Hpack.ttd.tt_cores.G2; G3 = Hpack.ttd.tt_cores.G3;
                    [U_rep, S_rep, V_rep] = tsvd_ttd_dim3(G1, G2, G3, nRows, nCols, s, opts.svd_tol);
                    r_used = min(k, U_rep.max_svd_rank);
                    [U1, S1, V1] = reconstruct_truncated_ttd_dim3_730_local(U_rep, S_rep, V_rep, r_used);

                case 'htd'
                    if mem_gb_htd > mem_budget_gb
                        fprintf('  %-18s : SKIPPED -- estimated memory %.1f GB > budget %.1f GB (HT rank r1=%d)\n', ...
                            method_names{mIdx}, mem_gb_htd, mem_budget_gb, r1_ht);
                        RelErr(c, mIdx, trial) = NaN;
                        continue;
                    end
                    U1h = Hpack.htd.ht_factors.U1; U2h = Hpack.htd.ht_factors.U2;
                    U3h = Hpack.htd.ht_factors.U3; B12h = Hpack.htd.ht_factors.B12;
                    Brooth = Hpack.htd.ht_factors.Broot;
                    [U_rep, S_rep, V_rep] = tsvd_htd_dim3(U1h, U2h, U3h, B12h, Brooth, opts.svd_tol);
                    r_used = min(k, U_rep.max_svd_rank);
                    [U1, S1, V1] = reconstruct_truncated_htd_dim3_729_local(U_rep, S_rep, V_rep, r_used);
            end

            Uhat = fft(U1, [], 3); Shat = fft(S1, [], 3); Vhat = fft(V1, [], 3);
            max_err = 0;
            for beta = 1:s
                Tb = Uhat(:,:,beta) * Shat(:,:,beta) * Vhat(:,:,beta)';
                max_err = max(max_err, svds(Ahat(:,:,beta) - Tb, 1));
            end
            relerr = max_err / max(max_full, eps);
            RelErr(c, mIdx, trial) = relerr;
            fprintf('  %-18s : relative H-infinity (full-tensor) error = %.15e (order r=%d)\n', ...
                method_names{mIdx}, relerr, r_used);

            recon.(tag) = struct('Uhat', Uhat, 'Shat', Shat, 'Vhat', Vhat);
        end

        if isfield(recon, 't') && isfield(recon, 'ttd')
            [max_err, max_def] = full_pair_diff(recon.t, recon.ttd, s);
            RelErr_TTDvsDef(c, trial) = max_err / max(max_def, eps);
            fprintf('  %-18s : relative H-infinity error vs Definition-based = %.15e\n', 'TTD-based', RelErr_TTDvsDef(c, trial));
        end
        if isfield(recon, 't') && isfield(recon, 'htd')
            [max_err, max_def] = full_pair_diff(recon.t, recon.htd, s);
            RelErr_HTDvsDef(c, trial) = max_err / max(max_def, eps);
            fprintf('  %-18s : relative H-infinity error vs Definition-based = %.15e\n', 'HTD-based', RelErr_HTDvsDef(c, trial));
        end
    end
end

RelErr_mean = mean(RelErr, 3, 'omitnan');
RelErr_std  = std(RelErr, 0, 3, 'omitnan');

RelErr_TTDvsDef_mean = mean(RelErr_TTDvsDef, 2, 'omitnan');
RelErr_TTDvsDef_std  = std(RelErr_TTDvsDef, 0, 2, 'omitnan');
RelErr_HTDvsDef_mean = mean(RelErr_HTDvsDef, 2, 'omitnan');
RelErr_HTDvsDef_std  = std(RelErr_HTDvsDef, 0, 2, 'omitnan');

fprintf('\n\n=== T-ERA relative H-infinity error table, full-tensor reconstruction vs exact H (mean over %d trials) ===\n', ntrials);
fprintf('%-10s', 'Case');
for mIdx = 1:numMethods
    fprintf('%38s', method_names{mIdx});
end
fprintf('\n');
for c = 1:numCases
    fprintf('%-10s', case_names{c});
    for mIdx = 1:numMethods
        fprintf('%38s', sprintf('%.15e (+-%.3e)', RelErr_mean(c,mIdx), RelErr_std(c,mIdx)));
    end
    fprintf('\n');
end

fprintf('\n=== T-ERA relative H-infinity error vs Definition-based, SAME order k, full-tensor (mean over %d trials) ===\n', ntrials);
fprintf('%-10s%38s%38s\n', 'Case', 'TTD_vs_Def', 'HTD_vs_Def');
for c = 1:numCases
    fprintf('%-10s', case_names{c});
    fprintf('%38s', sprintf('%.15e (+-%.3e)', RelErr_TTDvsDef_mean(c), RelErr_TTDvsDef_std(c)));
    fprintf('%38s', sprintf('%.15e (+-%.3e)', RelErr_HTDvsDef_mean(c), RelErr_HTDvsDef_std(c)));
    fprintf('\n');
end

if ~exist('results','dir'); mkdir('results'); end
jobid = getenv('SLURM_JOB_ID');
if isempty(jobid); jobid = datestr(now,'yyyymmdd_HHMMSS'); end

RowNames = case_names(:);
Tbl = array2table(RelErr_mean, 'VariableNames', strrep(method_names,'-','_'), 'RowNames', RowNames);
writetable(Tbl, fullfile('results', sprintf('tera_hinf_full_relerr_805_%s.csv', jobid)), 'WriteRowNames', true);

TblDef = table(RelErr_TTDvsDef_mean, RelErr_HTDvsDef_mean, ...
    'VariableNames', {'TTD_vs_Def','HTD_vs_Def'}, 'RowNames', RowNames);
writetable(TblDef, fullfile('results', sprintf('tera_hinf_full_relerr_vs_def_805_%s.csv', jobid)), 'WriteRowNames', true);

save(fullfile('results', sprintf('tera_hinf_full_relerr_805_%s.mat', jobid)), ...
     'case_names','method_names','RelErr','RelErr_mean','RelErr_std', ...
     'RelErr_TTDvsDef','RelErr_HTDvsDef','RelErr_TTDvsDef_mean','RelErr_TTDvsDef_std', ...
     'RelErr_HTDvsDef_mean','RelErr_HTDvsDef_std', ...
     'ntrials','H','n','s','k_frac');

disp(Tbl);
disp(TblDef);

%% ======================= Local Functions =======================

function [max_err, max_def] = full_pair_diff(recA, recB, s)
max_err = 0; max_def = 0;
for beta = 1:s
    Ta = recA.Uhat(:,:,beta) * recA.Shat(:,:,beta) * recA.Vhat(:,:,beta)';
    Tb = recB.Uhat(:,:,beta) * recB.Shat(:,:,beta) * recB.Vhat(:,:,beta)';
    max_err = max(max_err, svds(Ta - Tb, 1));
    max_def = max(max_def, svds(Ta, 1));
end
end

function [U1, S1, V1] = reconstruct_truncated_ttd_dim3_730_local(U_rep, S_rep, V_rep, r)
P1 = U_rep.cores{1};   % [1, n1, r1]
P2 = U_rep.cores{2};   % [r1, s, n3]
Q2 = S_rep.cores{2};   % [s, s, n3]
R1 = V_rep.cores{1};   % [1, n2, q]
R2 = V_rep.cores{2};   % [q, s, n3]

n1 = size(P1,2);
n2 = size(R1,2);
n3 = size(P2,3);

P2t = P2(:, 1:r, :);
Q2t = Q2(1:r, 1:r, :);
R2t = R2(:, 1:r, :);

Qmat = reshape(P1, n1, size(P1,3));
Q2mat = reshape(R1, n2, size(R1,3));

Uphys = ifft(P2t, [], 3);
Sphys = ifft(Q2t, [], 3);
Vphys = ifft(R2t, [], 3);

r1 = size(Qmat,2);
q = size(Q2mat,2);
U1 = real(reshape(Qmat * reshape(Uphys, r1, r*n3), n1, r, n3));
S1 = real(Sphys);
V1 = real(reshape(Q2mat * reshape(Vphys, q, r*n3), n2, r, n3));
end

function [U1, S1, V1] = reconstruct_truncated_htd_dim3_729_local(U_rep, S_rep, V_rep, r)
U1leaf = U_rep.leaf;
V1leaf = V_rep.leaf;

Pcore = U_rep.core(:, 1:r, :);
Qcore = S_rep.core(1:r, 1:r, :);
Rcore = V_rep.core(:, 1:r, :);

n1 = size(U1leaf,1); r1_ht = size(U1leaf,2);
n2 = size(V1leaf,1); r2_ht = size(V1leaf,2);
n3 = size(Pcore,3);

Uphys = ifft(Pcore, [], 3);
Sphys = ifft(Qcore, [], 3);
Vphys = ifft(Rcore, [], 3);

U1 = real(reshape(U1leaf * reshape(Uphys, r1_ht, r*n3), n1, r, n3));
S1 = real(Sphys);
V1 = real(reshape(V1leaf * reshape(Vphys, r2_ht, r*n3), n2, r, n3));
end

function [l, L] = pick_lL(H)
divs = find(mod(H, 1:floor(sqrt(H))) == 0);
cands = unique([divs, H./divs]);
target = sqrt(H);
[~, idx] = min(abs(cands - target));
l = cands(idx);
L = H/l - 1;
L = round(L);
if l*(L+1) ~= H
    error('Failed factorization: H=%d, got l=%d, L=%d', H, l, L);
end
end

function Hpack = gen_H_case_prebuilt(case_id, nRows, nCols, s, opts)
Hpack = struct();
TT_H_gen = [];
U1_gen = []; U2_gen = []; U3_gen = [];
B12_gen = []; Broot_gen = [];

switch case_id
    case 1   % Sparse random -- FIXED nnz (~30), independent of N
        target_nnz = 30;
        N = nRows*nCols*s;
        density = min(1, target_nnz/N);
        nnz_min = max(1, round(density*N));
        idx = randperm(N, nnz_min);
        H = zeros(N,1); H(idx) = randn(nnz_min,1);
        H = reshape(H, [nRows,nCols,s]);

    case 2   % Low TT-rank
        TT_H_gen = tt_rand([nRows,nCols,s], 3, [1 3 3 1]);
        H = reshape(full(TT_H_gen), [nRows,nCols,s]);

    case 3   % Low HT-rank
        r1 = 2; r2 = 2; r3 = 2; r12 = 2;
        U1_gen = orth(randn(nRows, r1));
        U2_gen = orth(randn(nCols, r2));
        U3_gen = orth(randn(s, r3));
        B12_gen = randn(r1, r2, r12);
        Broot_gen = randn(r12, r3);
        H = htd3_to_full(U1_gen, U2_gen, U3_gen, B12_gen, Broot_gen);

    otherwise
        error('Unknown case_id');
end

Hpack.full = struct('type','full', 'H', H, 'Hh', H);

if case_id == 2 && ~isempty(TT_H_gen)
    TT_H = TT_H_gen;
else
    TT_H = tt_tensor(H);
    try
        TT_H = round(TT_H, opts.tt_eps, opts.tt_rmax);
    catch
    end
end
fprintf('TT ranks used: ');
disp(TT_H.r');

[G1,G2,G3] = extract_tt_cores_dim3(TT_H, nRows, nCols, s);
Hpack.ttd = struct('type','ttd', 'tt_cores', struct('G1',G1,'G2',G2,'G3',G3), 'Hh', H);

if case_id == 3 && ~isempty(U1_gen)
    U1 = U1_gen; U2 = U2_gen; U3 = U3_gen; B12 = B12_gen; Broot = Broot_gen;
else
    ht_r1  = min(opts.ht_r1, nRows);
    ht_r2  = min(opts.ht_r2, nCols);
    ht_r3  = min(opts.ht_r3, s);
    ht_r12 = min(opts.ht_r12, ht_r1*ht_r2);
    [U1,U2,U3,B12,Broot] = numeric_to_ht_factors_dim3(H, ht_r1, ht_r2, ht_r3, ht_r12, opts.tt_eps);
end
fprintf('HT factor sizes used: U1=%s, U2=%s, U3=%s, B12=%s, Broot=%s\n', ...
    mat2str(size(U1)), mat2str(size(U2)), mat2str(size(U3)), ...
    mat2str(size(B12)), mat2str(size(Broot)));

Hpack.htd = struct('type','htd', 'ht_factors', struct('U1',U1,'U2',U2,'U3',U3,'B12',B12,'Broot',Broot), 'Hh', H);
end

function [G1,G2,G3] = extract_tt_cores_dim3(TT, n1, n2, n3)
r = TT.r;
r1 = r(2); r2 = r(3);
core = TT.core; ps = TT.ps;
G1 = reshape(core(ps(1):ps(2)-1), [1, n1, r1]);
G2 = reshape(core(ps(2):ps(3)-1), [r1, n2, r2]);
G3 = reshape(core(ps(3):ps(4)-1), [r2, n3, 1]);
end

function [U1,U2,U3,B12,Broot] = numeric_to_ht_factors_dim3(H, r1, r2, r3, r12, tol)
if nargin < 6 || isempty(tol)
    tol = 1e-10;
end
[n1,n2,n3_full] = size(H);
r1  = min(r1, n1); r2  = min(r2, n2); r3  = min(r3, n3_full);

[U1,S1] = svd(reshape(H, n1, []), 'econ');
sv1 = diag(S1); r1 = min([r1, max(1, sum(sv1 > tol*sv1(1))), size(U1,2)]);
U1 = U1(:,1:r1);

[U2,S2] = svd(reshape(permute(H,[2 1 3]), n2, []), 'econ');
sv2 = diag(S2); r2 = min([r2, max(1, sum(sv2 > tol*sv2(1))), size(U2,2)]);
U2 = U2(:,1:r2);

[U3,S3] = svd(reshape(permute(H,[3 1 2]), n3_full, []), 'econ');
sv3 = diag(S3); r3 = min([r3, max(1, sum(sv3 > tol*sv3(1))), size(U3,2)]);
U3 = U3(:,1:r3);

G = ttm3(H, U1', U2', U3');
Gmat = reshape(G, r1*r2, r3);
[U,S,V] = svd(Gmat, 'econ');
sv12 = diag(S);
r12 = min([r12, max(1, sum(sv12 > tol*sv12(1))), size(U,2), size(V,2)]);
U = U(:,1:r12); S = S(1:r12,1:r12); V = V(:,1:r12);
B12 = reshape(U, [r1, r2, r12]);
Broot = S * V.';
end

function G = ttm3(H, M1, M2, M3)
[n1,n2,n3] = size(H);
r1 = size(M1,1); r2 = size(M2,1); r3 = size(M3,1);
X = reshape(H, n1, []);
X = M1 * X;
X = reshape(X, r1, n2, n3);
X = permute(X, [2 1 3]);
X = reshape(X, n2, []);
X = M2 * X;
X = reshape(X, r2, r1, n3);
X = permute(X, [2 1 3]);
X = permute(X, [3 1 2]);
X = reshape(X, n3, []);
X = M3 * X;
X = reshape(X, r3, r1, r2);
G = permute(X, [2 3 1]);
end

function A = htd3_to_full(U1,U2,U3,B12,Broot)
[n1,~] = size(U1);
[n2,~] = size(U2);
[n3, r3] = size(U3);
r12 = size(B12,3);

G = zeros(size(B12,1), size(B12,2), r3);
for a = 1:r3
    for q = 1:r12
        G(:,:,a) = G(:,:,a) + Broot(q,a) * B12(:,:,q);
    end
end

A = zeros(n1,n2,n3);
for a = 1:r3
    S = U1 * G(:,:,a) * U2.';
    for t = 1:n3
        A(:,:,t) = A(:,:,t) + U3(t,a) * S;
    end
end
end
