%% ---------------------------------------------------------------
%  T-ERA relative H-infinity error table (3 cases x 3 methods = 9 numbers):
%  Definition-based vs TTD-based vs HTD-based Hankel T-SVD backends.
%
%  Successor to Test_TERA_3_729_10_hinf.m, updated to match the nnz-cap fix
%  and tera_reduce backend used by Test_TERA_3_803.m / Test_TERA_3_804.m.
%  This is the accuracy/H-infinity companion to that timing driver: _803/_804
%  only timed the three methods (and _804's naive Frobenius-norm A_red/B_red/
%  C_red comparison crashed on a dimension mismatch -- the 't' baseline
%  truncates to the FIXED order r=min(nRows,nCols)-k=2000, while 'ttd'/'htd'
%  truncate to the tensor's own numerical rank, e.g. ~30/3/2 in the three
%  cases -- so the reduced models have different state dimension and cannot
%  be subtracted elementwise). This file instead reuses the pre-existing,
%  basis/dimension-independent H-infinity relative-error methodology from
%  Test_TERA_3_729_10_hinf.m: each method's reduced model is compared to a
%  common higher-order reference via per-frequency resolvent (transfer
%  function) evaluation, so comparisons remain valid across differing state
%  orders.
%
%  Changes relative to Test_TERA_3_729_10_hinf.m:
%    (1) Sparse case (case 1) Hankel generation: nonzero COUNT is now capped
%        at a fixed target_nnz (~30), independent of N=nRows*nCols*s -- same
%        fix as Test_TERA_3_803.m/_804.m -- instead of the old fixed density
%        (1e-4) that let nnz, and the achieved TT/HT rank, grow with problem
%        size (H is fixed at 10000 here, so this mainly keeps the fix
%        consistent across all T-ERA drivers rather than changing behavior
%        at this one fixed H).
%    (2) All T_ERA_fast_729 calls -> tera_reduce (mirrors _803/_804;
%        adds mode-2 compression on the 'ttd' branch, see that function's
%        header). The 'htd' branch is unchanged internally.
%  All other logic (reference-model construction at r_ref=1.3*r_normal,
%  memory guard, H-infinity computation, opts, n=100/s=9/H=10000/k_frac=0.8,
%  ntrials=3) is carried forward unchanged from Test_TERA_3_729_10_hinf.m.
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

w_grid = linspace(0, pi, 60);

opts = struct();
opts.svd_tol = 1e-12;
opts.tt_eps  = 1e-10;
opts.tt_rmax = 1e5;    % was 30 -- effectively uncapped, tolerance governs
opts.ht_r1   = 1e5;
opts.ht_r2   = 1e5;
opts.ht_r3   = min(9,s);
opts.ht_r12  = 1e5;

mem_budget_gb = 100;   % same budget as Test_TERA_3_729_10.m

method_names = {'Definition-based','TTD-based','HTD-based'};
method_tags  = {'t','ttd','htd'};
numMethods = numel(method_tags);

RelErr = nan(numCases, numMethods, ntrials);
% Extra comparison: TTD-based / HTD-based vs Definition-based, both AT THE
% SAME target order k (unlike RelErr above, which compares every method
% against a separate higher-order reference model). This gives a second,
% tighter view of TTD/HTD fidelity that doesn't go through the reference.
RelErr_TTDvsDef = nan(numCases, ntrials);
RelErr_HTDvsDef = nan(numCases, ntrials);

[l, L] = pick_lL(H);
m = l; T = L;
nRows = l*(L+1); nCols = m*(T+1);
k = min(round(k_frac * min(nRows,nCols)), min(nRows,nCols)-1);
r_normal = min(nRows,nCols) - k;
r_ref = min(round(1.3*r_normal), min(nRows,nCols)-1);
k_ref = min(nRows,nCols) - r_ref;

for trial = 1:ntrials
    [A, B, C] = gen_ABC_base(n, m, l, s);
    A = stabilize_tensor_A(A, 0.95);

    for c = 1:numCases
        fprintf('\n=== Case: %s | trial %d/%d ===\n', case_names{c}, trial, ntrials);
        Hpack = gen_H_case_prebuilt(c, nRows, nCols, s, opts);

        [A_ref, B_ref, C_ref, ~] = tera_reduce(A, B, C, k_ref, T, L, 't', opts, Hpack.full);
        fprintf('  %-18s : (reference, order r=%d)\n', 'High-precision ref', size(A_ref,1));

        % Memory guard, only meaningfully large for the Sparse case
        % (LowTT/LowHT ranks stay tiny regardless), matching the same
        % reasoning already used in Test_TERA_3_729_10.m.
        r1_tt = size(Hpack.ttd.tt_cores.G1, 3);
        mem_gb_ttd = s * r1_tt^2 * 16 / 1e9;
        r1_ht = size(Hpack.htd.ht_factors.U1, 2);
        mem_gb_htd = s * r1_ht^2 * 16 / 1e9;

        A_def=[]; B_def=[]; C_def=[];
        A_ttd=[]; B_ttd=[]; C_ttd=[];
        A_htd=[]; B_htd=[]; C_htd=[];

        for mIdx = 1:numMethods
            tag = method_tags{mIdx};
            switch tag
                case 't'
                    Hin = Hpack.full;
                case 'ttd'
                    if mem_gb_ttd > mem_budget_gb
                        fprintf('  %-18s : SKIPPED -- estimated memory %.1f GB > budget %.1f GB (TT rank r1=%d)\n', ...
                            method_names{mIdx}, mem_gb_ttd, mem_budget_gb, r1_tt);
                        RelErr(c, mIdx, trial) = NaN;
                        continue;
                    end
                    Hin = Hpack.ttd;
                case 'htd'
                    if mem_gb_htd > mem_budget_gb
                        fprintf('  %-18s : SKIPPED -- estimated memory %.1f GB > budget %.1f GB (HT rank r1=%d)\n', ...
                            method_names{mIdx}, mem_gb_htd, mem_budget_gb, r1_ht);
                        RelErr(c, mIdx, trial) = NaN;
                        continue;
                    end
                    Hin = Hpack.htd;
            end
            [A_red, B_red, C_red, ~] = tera_reduce(A, B, C, k, T, L, tag, opts, Hin);
            relerr = relative_hinf_error(A_ref, B_ref, C_ref, A_red, B_red, C_red, w_grid);
            RelErr(c, mIdx, trial) = relerr;
            fprintf('  %-18s : relative H-infinity error vs reference = %.15e (order r=%d)\n', ...
                method_names{mIdx}, relerr, size(A_red,1));

            switch tag
                case 't';   A_def=A_red; B_def=B_red; C_def=C_red;
                case 'ttd'; A_ttd=A_red; B_ttd=B_red; C_ttd=C_red;
                case 'htd'; A_htd=A_red; B_htd=B_red; C_htd=C_red;
            end
        end

        % TTD-based / HTD-based vs Definition-based, same order k (skipped
        % methods -- e.g. memory-guard NaN -- leave the comparison as NaN
        % too, since there is no reduced model to compare).
        if ~isempty(A_ttd)
            relerr_ttd_def = relative_hinf_error(A_def, B_def, C_def, A_ttd, B_ttd, C_ttd, w_grid);
            RelErr_TTDvsDef(c, trial) = relerr_ttd_def;
            fprintf('  %-18s : relative H-infinity error vs Definition-based = %.15e\n', 'TTD-based', relerr_ttd_def);
        end
        if ~isempty(A_htd)
            relerr_htd_def = relative_hinf_error(A_def, B_def, C_def, A_htd, B_htd, C_htd, w_grid);
            RelErr_HTDvsDef(c, trial) = relerr_htd_def;
            fprintf('  %-18s : relative H-infinity error vs Definition-based = %.15e\n', 'HTD-based', relerr_htd_def);
        end
    end
end

RelErr_mean = mean(RelErr, 3, 'omitnan');
RelErr_std  = std(RelErr, 0, 3, 'omitnan');

RelErr_TTDvsDef_mean = mean(RelErr_TTDvsDef, 2, 'omitnan');
RelErr_TTDvsDef_std  = std(RelErr_TTDvsDef, 0, 2, 'omitnan');
RelErr_HTDvsDef_mean = mean(RelErr_HTDvsDef, 2, 'omitnan');
RelErr_HTDvsDef_std  = std(RelErr_HTDvsDef, 0, 2, 'omitnan');

fprintf('\n\n=== T-ERA relative H-infinity error table vs high-precision reference (mean over %d trials) ===\n', ntrials);
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

fprintf('\n=== T-ERA relative H-infinity error vs Definition-based, SAME order k (mean over %d trials) ===\n', ntrials);
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
writetable(Tbl, fullfile('results', sprintf('tera_hinf_relerr_804_%s.csv', jobid)), 'WriteRowNames', true);

TblDef = table(RelErr_TTDvsDef_mean, RelErr_HTDvsDef_mean, ...
    'VariableNames', {'TTD_vs_Def','HTD_vs_Def'}, 'RowNames', RowNames);
writetable(TblDef, fullfile('results', sprintf('tera_hinf_relerr_vs_def_804_%s.csv', jobid)), 'WriteRowNames', true);

save(fullfile('results', sprintf('tera_hinf_relerr_804_%s.mat', jobid)), ...
     'case_names','method_names','RelErr','RelErr_mean','RelErr_std', ...
     'RelErr_TTDvsDef','RelErr_HTDvsDef','RelErr_TTDvsDef_mean','RelErr_TTDvsDef_std', ...
     'RelErr_HTDvsDef_mean','RelErr_HTDvsDef_std', ...
     'ntrials','w_grid','H','n','s','k_frac');

disp(Tbl);
disp(TblDef);

%% ======================= Local Functions =======================

function relerr = relative_hinf_error(A, B, C, A_red, B_red, C_red, w_grid)
s = size(A,3);
n1 = size(A,1); n2 = size(B,2); l1 = size(C,1);
r1 = size(A_red,1); m2 = size(B_red,2); l2 = size(C_red,1);
assert(n2==m2 && l1==l2, 'I/O dimension mismatch between reference and reduced systems.');

Ahat = fft(A,[],3);     Bhat = fft(B,[],3);     Chat = fft(C,[],3);
Aredhat = fft(A_red,[],3); Bredhat = fft(B_red,[],3); Credhat = fft(C_red,[],3);

In1 = eye(n1); Ir1 = eye(r1);
max_err = 0; max_full = 0;
radius = 1.01;

for beta = 1:s
    Ab = Ahat(:,:,beta); Bb = Bhat(:,:,beta); Cb = Chat(:,:,beta);
    Arb = Aredhat(:,:,beta); Brb = Bredhat(:,:,beta); Crb = Credhat(:,:,beta);

    for zi = 1:numel(w_grid)
        z = radius*exp(1i*w_grid(zi));
        Gfull = Cb  * ((z*In1 - Ab)  \ Bb);
        Gred  = Crb * ((z*Ir1 - Arb) \ Brb);
        sv_err  = svd(Gfull - Gred);
        sv_full = svd(Gfull);
        max_err  = max(max_err,  sv_err(1));
        max_full = max(max_full, sv_full(1));
    end
end

relerr = max_err / max(max_full, eps);
end

function [A,B,C] = gen_ABC_base(n, m, l, s)
density = 1e-3;
A = sprandn(n*n*s, 1, density);
A = reshape(full(A), [n,n,s]) / 200;
B = randn(n,m,s);
C = randn(l,n,s);
end

function A = stabilize_tensor_A(A, target_rho)
Ahat = fft(A, [], 3);
s = size(A,3);
for j = 1:s
    Aj = Ahat(:,:,j);
    rho = max(abs(eig(Aj)));
    if rho > 0
        Ahat(:,:,j) = (target_rho / max(target_rho, rho)) * Aj;
    end
end
A = real(ifft(Ahat, [], 3));
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
    case 1   % Sparse random -- _804: FIXED nnz (~30), independent of N
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
