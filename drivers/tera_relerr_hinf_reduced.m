%% ---------------------------------------------------------------
%  TERA_RELERR_HINF_REDUCED
%  Builds the T-ERA reduced-order model (A_red,B_red,C_red) with each of
%  the Definition-based, TTD-based, and HTD-based T-SVD backends, then
%  compares the resulting transfer functions using the exact
%  H-infinity norm (relative to the Definition-based model), across the
%  Sparse / Low-TT-rank / Low-HT-rank test cases.
%
%  Required files:
%    - tera_reduce.m           (reduced-system construction; internally
%                                uses tsvd_ttd_dim3.m / tsvd_htd_dim3.m
%                                for the 'ttd'/'htd' backends)
%    - bcirc.m                 (block-circulant unfolding, used to get
%                                the exact H-infinity norm via hinfnorm()
%                                instead of a frequency-grid sweep)
%    - Control System Toolbox  (ss, hinfnorm)
%    - TT-Toolbox              (tt_tensor, round, tt_rand)
%
%  Input (hardcoded problem parameters, no external args):
%    - n=100, s=9, H=10000 (nRows=nCols=10000)
%    - case_names = {'Sparse','LowTT','LowHT'}
%    - ntrials=3
%
%  Output (written to results/, filenames suffixed by SLURM_JOB_ID or a
%  timestamp):
%    - tera_relerr_hinf_reduced_<id>.csv/.mat -- one row per case with
%      r_cap (achieved reduced order), TTD-vs-Definition and HTD-vs-
%      Definition relative H-infinity error (mean/std over trials), and
%      the Definition-based diag-check sanity value.
%  ---------------------------------------------------------------

clear; clc;

cd(fileparts(mfilename('fullpath')));
addpath(genpath(pwd));

n = 100;
s = 9;
H = 10000;

case_names = {'Sparse', 'LowTT', 'LowHT'};
numCases = numel(case_names);
ntrials = 3;

svd_rank_tol = 1e-10;
D_lo = 0.2; D_hi = 0.8;

opts = struct();
opts.svd_tol = 1e-12;
opts.tt_eps  = 1e-10;
opts.tt_rmax = 1e5;
opts.ht_r1   = 1e5;
opts.ht_r2   = 1e5;
opts.ht_r3   = min(9,s);
opts.ht_r12  = 1e5;

method_names = {'Definition-based','TTD-based','HTD-based'};
method_tags  = {'t','ttd','htd'};
numMethods = numel(method_tags);

RelErr_hinf = nan(numCases, numMethods, ntrials);   % col 1 unused (Def vs itself)
RelErr_TTDvsDef_hinf = nan(numCases, ntrials);
RelErr_HTDvsDef_hinf = nan(numCases, ntrials);
Achieved_rcap = nan(numCases, ntrials);
DiagCheck_Def = nan(numCases, ntrials);   % ||A_red_def - diag(D)||/||diag(D)||, sanity check

[l, L] = pick_lL(H);
m = l; T = L;
nRows = l*(L+1); nCols = m*(T+1);

for trial = 1:ntrials
    for c = 1:numCases
        fprintf('\n=== Case: %s | trial %d/%d ===\n', case_names{c}, trial, ntrials);
        Hpack = gen_H_case_prebuilt(c, nRows, nCols, l, m, s, opts);
        Hmat = Hpack.full.H;

        % ---- full/exact per-slice SVD of H, achieved-rank cap ----
        Hhat = fft(Hmat, [], 3);
        rank_beta = zeros(s,1);
        Ucell = cell(s,1); Scell = cell(s,1); Vcell = cell(s,1);
        for b = 1:s
            [U,S,V] = svd(Hhat(:,:,b));
            sv = diag(S);
            rank_beta(b) = max(1, sum(sv > svd_rank_tol*sv(1)));
            Ucell{b}=U; Scell{b}=S; Vcell{b}=V;
        end
        r_cap = min(rank_beta);
        Achieved_rcap(c,trial) = r_cap;
        fprintf('  achieved rank per beta: min=%d max=%d -> r_cap=%d\n', min(rank_beta), max(rank_beta), r_cap);

        % ---- prescribed stable diagonal D, beta-independent ----
        Dvals = linspace(D_lo, D_hi, r_cap);

        % ---- Hh_hat_beta = U S^{1/2} D S^{1/2} V',  from FULL SVD ----
        Hh_hat = zeros(nRows, nCols, s);
        for b = 1:s
            U = Ucell{b}; S = Scell{b}; V = Vcell{b};
            r_full = min(size(S,1), size(S,2));
            sv = diag(S);
            Shalf = zeros(r_full,1); Shalf(1:r_cap) = sqrt(sv(1:r_cap));
            Dfull = zeros(r_full,1); Dfull(1:r_cap) = Dvals(:);
            middle = diag(Shalf .* Dfull .* Shalf);
            Hh_hat(:,:,b) = U(:,1:r_full) * middle * V(:,1:r_full)';
        end
        Hh_new = real(ifft(Hh_hat, [], 3));

        k_target = min(nRows,nCols) - r_cap;

        A_red_all = cell(numMethods,1); B_red_all = cell(numMethods,1); C_red_all = cell(numMethods,1);
        A_dummy = zeros(1,1,s); B_dummy = zeros(1,m,s); C_dummy = zeros(l,1,s);

        for mIdx = 1:numMethods
            tag = method_tags{mIdx};
            switch tag
                case 't';   Hin = struct('type','full','H',Hmat,'Hh',Hh_new);
                case 'ttd'; Hin = struct('type','ttd','tt_cores',Hpack.ttd.tt_cores,'Hh',Hh_new);
                case 'htd'; Hin = struct('type','htd','ht_factors',Hpack.htd.ht_factors,'Hh',Hh_new);
            end
            [A_red,B_red,C_red,~] = tera_reduce(A_dummy,B_dummy,C_dummy,k_target,T,L,tag,opts,Hin);
            A_red_all{mIdx}=A_red; B_red_all{mIdx}=B_red; C_red_all{mIdx}=C_red;
            fprintf('  %-18s : order r=%d\n', method_names{mIdx}, size(A_red,1));
        end

        % ---- sanity check: Definition-based A_red should match diag(D) ----
        Adef_hat = fft(A_red_all{1},[],3);
        dchk = 0;
        for b = 1:s
            Ab = Adef_hat(:,:,b);
            dchk = max(dchk, norm(Ab - diag(Dvals(1:size(Ab,1))), 'fro') / max(norm(diag(Dvals(1:size(Ab,1))),'fro'), eps));
        end
        DiagCheck_Def(c,trial) = dchk;
        fprintf('  Definition-based ||A_red-diag(D)||/||diag(D)|| (max over beta) = %.3e\n', dchk);

        % ---- H-infinity relative error: TTD vs Def, HTD vs Def ----
        sys_def = ss(bcirc(A_red_all{1}), bcirc(B_red_all{1}), bcirc(C_red_all{1}), 0, -1);
        hn_def = hinfnorm(sys_def);
        for mIdx = 2:numMethods
            sys_m = ss(bcirc(A_red_all{mIdx}), bcirc(B_red_all{mIdx}), bcirc(C_red_all{mIdx}), 0, -1);
            relerr = hinfnorm(sys_def - sys_m) / max(hn_def, eps);
            RelErr_hinf(c,mIdx,trial) = relerr;
            fprintf('  relerr(%s vs Definition-based) hinf = %.6e\n', method_names{mIdx}, relerr);
        end
        RelErr_TTDvsDef_hinf(c,trial) = RelErr_hinf(c,2,trial);
        RelErr_HTDvsDef_hinf(c,trial) = RelErr_hinf(c,3,trial);
    end
end

RelErr_TTDvsDef_mean = mean(RelErr_TTDvsDef_hinf, 2, 'omitnan');
RelErr_TTDvsDef_std  = std(RelErr_TTDvsDef_hinf, 0, 2, 'omitnan');
RelErr_HTDvsDef_mean = mean(RelErr_HTDvsDef_hinf, 2, 'omitnan');
RelErr_HTDvsDef_std  = std(RelErr_HTDvsDef_hinf, 0, 2, 'omitnan');
Achieved_rcap_mean   = mean(Achieved_rcap, 2, 'omitnan');
DiagCheck_Def_max    = max(DiagCheck_Def, [], 2);

fprintf('\n\n=== T-ERA reduced-system relative H-infinity error, TTD/HTD vs Definition-based (mean over %d trials) ===\n', ntrials);
fprintf('%-10s%18s%18s%18s%22s\n', 'Case', 'r_cap(avg)', 'TTD_vs_Def', 'HTD_vs_Def', 'Def A_red diag-check');
for c = 1:numCases
    fprintf('%-10s%18.1f%18s%18s%22s\n', case_names{c}, Achieved_rcap_mean(c), ...
        sprintf('%.3e(+-%.1e)', RelErr_TTDvsDef_mean(c), RelErr_TTDvsDef_std(c)), ...
        sprintf('%.3e(+-%.1e)', RelErr_HTDvsDef_mean(c), RelErr_HTDvsDef_std(c)), ...
        sprintf('%.3e', DiagCheck_Def_max(c)));
end

if ~exist('results','dir'); mkdir('results'); end
jobid = getenv('SLURM_JOB_ID');
if isempty(jobid); jobid = datestr(now,'yyyymmdd_HHMMSS'); end

RowNames = case_names(:);
TblDef = table(Achieved_rcap_mean, RelErr_TTDvsDef_mean, RelErr_TTDvsDef_std, RelErr_HTDvsDef_mean, RelErr_HTDvsDef_std, DiagCheck_Def_max, ...
    'VariableNames', {'r_cap_avg','TTD_vs_Def_mean','TTD_vs_Def_std','HTD_vs_Def_mean','HTD_vs_Def_std','Def_diagcheck_max'}, 'RowNames', RowNames);
writetable(TblDef, fullfile('results', sprintf('tera_relerr_hinf_reduced_%s.csv', jobid)), 'WriteRowNames', true);

save(fullfile('results', sprintf('tera_relerr_hinf_reduced_%s.mat', jobid)), ...
     'case_names','method_names', ...
     'RelErr_hinf','RelErr_TTDvsDef_hinf','RelErr_HTDvsDef_hinf', ...
     'RelErr_TTDvsDef_mean','RelErr_TTDvsDef_std','RelErr_HTDvsDef_mean','RelErr_HTDvsDef_std', ...
     'Achieved_rcap','DiagCheck_Def', ...
     'ntrials','H','n','s','D_lo','D_hi','svd_rank_tol');

disp(TblDef);

%% ======================= Local Functions =======================

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

function Hpack = gen_H_case_prebuilt(case_id, nRows, nCols, l, m, s, opts)
Hpack = struct();
TT_H_gen = [];
U1_gen = []; U2_gen = []; U3_gen = [];
B12_gen = []; Broot_gen = [];

switch case_id
    case 1   % Sparse random -- nnz placed inside the (1:l,1:m,:) corner
        target_nnz = 30;
        N_corner = l*m*s;
        nnz_min = min(target_nnz, N_corner);
        idx_corner = randperm(N_corner, nnz_min);
        Hcorner = zeros(l,m,s);
        Hcorner(idx_corner) = randn(nnz_min,1);
        H = zeros(nRows,nCols,s);
        H(1:l,1:m,:) = Hcorner;

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
