%% TPROD_TIMING  3rd-order T-product timing/accuracy comparison:
% definition-based vs TTD-based vs HTD-based, across three tensor-
% generation regimes and a range of problem sizes.
%
% What it does: for each of three cases -- (1) Random Sparse (fixed
% nnz=30, independent of tensor size), (2) Low TT-rank, (3) Low HT-rank
% -- and each size n=n1=n2=l in 2.^(2:13) (transform-mode dimension
% n3=6 fixed), generates a random A,B pair in that regime and times:
%   - tprod_definition.m (definition-based, block-circulant)
%   - tprod_ttd.m (TTD-based, computed from TT cores)
%   - tprod_htd.m (HTD-based, computed from HT factors)
% and (Sparse case) records the relative error of the TTD-/HTD-based
% product against the definition-based result. Saves timing/accuracy
% tables and a 3-panel log-log timing figure.
%
% Needs on the path: tprod_definition.m, tprod_ttd.m, tprod_htd.m, plus
% the TT-Toolbox (tt_tensor/tt_rand/round) and htucker toolboxes for
% tensor generation.
%
% Outputs (to results/, filenames suffixed by SLURM_JOB_ID or a
% timestamp): tprod_dim3_data_803_<id>.mat, tprod_dim3_panel_803_<id>.fig.
%
% Notes:
%   - Definition-based T-product is skipped (NaN) above a fixed size
%     cutoff (n_full_max_def) or on OOM/error (caught per case); TTD-
%     based T-product in the Sparse case is additionally skipped under a
%     dynamic memory-budget estimate (mem_budget_gb) once its achieved
%     rank makes the core tensor too large.
%   - Low TT-rank / Low HT-rank cases are constructed to have a small,
%     size-independent true rank, so none of the above guards ever bind
%     there.

clc; clear; close all;

%% -------------------- (1) Shared settings --------------------
rng(1);

n3 = 6;                 % transform-mode dimension (paper: n3 = 6)
powers = 2:13;          % _803: sweep now starts at n=4 (was 3:13, n=8)
sizes = 2.^powers;      % paper: sizes = 2.^powers
num_trials = 5;         % paper: num_trials = 5

n_array = sizes;        % n1 = n2 = l = sizes(i) for each size below
r = n3;
n_trial = num_trials;

n_full_max_def   = 2.^13;  % run definition-based only when n <= this
n_full_max_build = Inf;

% sparse case (case 1) nonzero count -- _803: FIXED target nnz (independent
% of n), rather than a fixed density that let nnz (and hence achieved
% rank) grow with n. density is derived per-n as target_nnz/N so the
% generation call still reads like a density-based sprandn.
target_nnz_sparse = 30;

% TT compression: accuracy governed by tt_eps only (rmax effectively uncapped)
tt_eps  = 1e-10;
tt_rmax = 1e5;

% HTucker: rmax effectively uncapped -> numeric_to_ht_factors_dim3 is
% tolerance-adaptive (see below), same tol as tt_eps.
ht_rmax = 1e5;
ht_rel_eps = 1e-10;

% Memory guard for TTD-based T-product in the Sparse case: skip (NaN) if
% the estimated size of the core tensor G2C = (r1A*r1B) x l x (r2A*r2B)
% (complex double, 16 bytes/element) would exceed this budget. Set well
% below the SLURM job's --mem allocation to leave headroom for MATLAB
% overhead and other live arrays.
mem_budget_gb = 150;

L = numel(n_array);

t_def_sparse = nan(n_trial, L);
t_ttd_sparse = nan(n_trial, L);
t_htd_sparse = nan(n_trial, L);
err_ttd_sparse = nan(n_trial, L);
err_htd_sparse = nan(n_trial, L);

t_def_tt = nan(n_trial, L);
t_ttd_tt = nan(n_trial, L);
t_htd_tt = nan(n_trial, L);

t_def_ht = nan(n_trial, L);
t_ttd_ht = nan(n_trial, L);
t_htd_ht = nan(n_trial, L);

has_tt = exist('tt_tensor','file')==2;
has_ht = exist('htensor','class')==8 || exist('htensor','file')==2;

%% ============================================================
% CASE 1: Sparse full tensors (full -> TT/HT approximations), accuracy-fixed
% ============================================================
for trial = 1:n_trial
    fprintf('[Case1 Sparse] Trial %d/%d\n', trial, n_trial);

    for i = 1:L
        n = n_array(i);  m = n;

        if n > n_full_max_build
            continue;
        end

        % ---- Generate sparse A_full, B_full (fixed nnz, guard against all-zero) ----
        N_A = n*n*r;
        density_A = min(1, target_nnz_sparse / N_A);
        nnz_A = max(1, round(density_A * N_A));
        idxA = randperm(N_A, nnz_A);
        A_vec = zeros(N_A,1);  A_vec(idxA) = randn(nnz_A,1);
        A_full = reshape(A_vec, [n,n,r]);

        N_B = n*m*r;
        density_B = min(1, target_nnz_sparse / N_B);
        nnz_B = max(1, round(density_B * N_B));
        idxB = randperm(N_B, nnz_B);
        B_vec = zeros(N_B,1);  B_vec(idxB) = randn(nnz_B,1);
        B_full = reshape(B_vec, [n,m,r]);

        A_full = A_full / max(1, norm(A_full(:)));
        B_full = B_full / max(1, norm(B_full(:)));

        C_def = [];

        % ---- (a) Definition-based (threshold; OOM/error -> NaN, continue) ----
        if n <= n_full_max_def
            try
                tic;
                C_def = tprod_definition(A_full, B_full);
                t_def_sparse(trial,i) = toc;
            catch ME
                fprintf('[Case1 Sparse] tprod_definition FAILED at n=%d: %s\n', n, ME.message);
                t_def_sparse(trial,i) = NaN;
                C_def = [];
            end
        end

        % ---- (b) TTD-based (with memory guard) ----
        G1C = []; G2C = []; G3C = [];
        if has_tt
            TT_A = tt_tensor(A_full);
            TT_B = tt_tensor(B_full);
            try
                TT_A = round(TT_A, tt_eps, tt_rmax);
                TT_B = round(TT_B, tt_eps, tt_rmax);
            catch
            end
            [G1A,G2A,G3A] = extract_tt_cores_dim3(TT_A, n, n, r);
            [G1B,G2B,G3B] = extract_tt_cores_dim3(TT_B, n, m, r);

            r1_est = TT_A.r(2) * TT_B.r(2);
            r2_est = TT_A.r(3) * TT_B.r(3);
            mem_gb_needed = r1_est * m * r2_est * 16 / 1e9;   % G2C, complex double

            if mem_gb_needed > mem_budget_gb
                fprintf('[Case1 Sparse] TTD-based SKIPPED at n=%d: estimated core memory %.1f GB > budget %.1f GB (ranks %d x %d, %d x %d) -- expected once rs dominates per mainCC_260729.tex O((r+s+rs)n log n) complexity.\n', ...
                    n, mem_gb_needed, mem_budget_gb, TT_A.r(2), TT_A.r(3), TT_B.r(2), TT_B.r(3));
                t_ttd_sparse(trial,i) = NaN;
            else
                tic;
                [G1C, G2C, G3C] = tprod_ttd(G1A,G2A,G3A, G1B,G2B,G3B, n, n, m, r);
                t_ttd_sparse(trial,i) = toc;
            end
        end

        % ---- (c) HTD-based (tolerance-adaptive HT factors) ----
        U1C=[]; U2C=[]; U3C=[]; B12C=[]; BrootC=[];
        if has_ht
            r12A = min([ht_rmax, n*n, r]);
            r1A  = min([ht_rmax, n]);
            r2A  = min([ht_rmax, n]);

            r12B = min([ht_rmax, n*m, r]);
            r1B  = min([ht_rmax, n]);
            r2B  = min([ht_rmax, m]);

            [U1A,U2A,U3A,B12A,BrootA] = numeric_to_ht_factors_dim3(A_full, r1A,r2A,r12A, ht_rel_eps);
            [U1B,U2B,U3B,B12B,BrootB] = numeric_to_ht_factors_dim3(B_full, r1B,r2B,r12B, ht_rel_eps);

            tic;
            [U1C,U2C,U3C,B12C,BrootC] = tprod_htd(U1A,U2A,U3A,B12A,BrootA, U1B,U2B,U3B,B12B,BrootB);
            t_htd_sparse(trial,i) = toc;
        end

        % ---- Accuracy check against definition-based (outside timing) ----
        if ~isempty(C_def)
            if ~isempty(G1C)
                C_ttd = tt3_to_full(G1C,G2C,G3C);
                err_ttd_sparse(trial,i) = norm(C_ttd(:)-C_def(:))/max(1,norm(C_def(:)));
            end
            if ~isempty(U1C)
                C_htd = htd3_to_full(U1C,U2C,U3C,B12C,BrootC);
                err_htd_sparse(trial,i) = norm(C_htd(:)-C_def(:))/max(1,norm(C_def(:)));
            end
        end

        fprintf('trial=%d | i=%d | n=%d | def=%.6f s | htd=%.6f s | ttd=%.6f s | errTTD=%.3e | errHTD=%.3e\n', ...
            trial, i, n, ...
            t_def_sparse(trial,i), ...
            t_htd_sparse(trial,i), ...
            t_ttd_sparse(trial,i), ...
            err_ttd_sparse(trial,i), ...
            err_htd_sparse(trial,i));
    end
end
fprintf('\n');

%% ==================================
%% CASE 2: Low TT-rank tensors (unchanged behavior; true rank stays small)
%% ==================================
tt_rank = [1, 2, 2, 1];

for trial = 1:n_trial
    fprintf('[Case2 Low TT-rank] Trial %d/%d\n', trial, n_trial);

    for i = 1:L
        n = n_array(i);  m = n;

        if has_tt
            TT_A = tt_rand([n,n,r], 3, tt_rank);
            TT_B = tt_rand([n,m,r], 3, tt_rank);
            [G1A,G2A,G3A] = extract_tt_cores_dim3(TT_A, n, n, r);
            [G1B,G2B,G3B] = extract_tt_cores_dim3(TT_B, n, m, r);

            tic;
            [G1C, G2C, G3C] = tprod_ttd(G1A,G2A,G3A, G1B,G2B,G3B, n, n, m, r);
            t_ttd_tt(trial,i) = toc;
        end

        if n <= n_full_max_build
            if has_tt
                A_full = reshape(full(TT_A), [n,n,r]);
                B_full = reshape(full(TT_B), [n,m,r]);
            else
                continue;
            end

            if n <= n_full_max_def
                try
                    tic;
                    C_def = tprod_definition(A_full, B_full);
                    t_def_tt(trial,i) = toc;
                catch ME
                    fprintf('[Case2 Low TT-rank] tprod_definition FAILED at n=%d: %s\n', n, ME.message);
                    t_def_tt(trial,i) = NaN;
                    C_def = [];
                end
            end

            if has_ht
                r12A = min([ht_rmax, n*n, r]);
                r1A  = min([ht_rmax, n]);
                r2A  = min([ht_rmax, n]);

                r12B = min([ht_rmax, n*m, r]);
                r1B  = min([ht_rmax, n]);
                r2B  = min([ht_rmax, m]);

                [U1A,U2A,U3A,B12A,BrootA] = numeric_to_ht_factors_dim3(A_full, r1A,r2A,r12A, ht_rel_eps);
                [U1B,U2B,U3B,B12B,BrootB] = numeric_to_ht_factors_dim3(B_full, r1B,r2B,r12B, ht_rel_eps);

                tic;
                [U1C,U2C,U3C,B12C,BrootC] = tprod_htd(U1A,U2A,U3A,B12A,BrootA, U1B,U2B,U3B,B12B,BrootB);
                t_htd_tt(trial,i) = toc;

                fprintf('trial=%d | i=%d | def=%.6f s | htd=%.6f s | ttd=%.6f s\n', ...
                    trial, i, ...
                    t_def_tt(trial,i), ...
                    t_htd_tt(trial,i), ...
                    t_ttd_tt(trial,i));
            end
        end
    end
end
fprintf('\n');

%% ==================================
%% CASE 3: Low HTD-rank tensors (direct HT cores; unchanged behavior)
%% ==================================
for trial = 1:n_trial
    fprintf('[Case3 Low HT-rank] Trial %d/%d\n', trial, n_trial);

    for i = 1:L
        n = n_array(i);  m = n;

        r12A = min([8, n*n, r]);    r1A = min([ceil(sqrt(r12A)), n]); r2A = min([ceil(sqrt(r12A)), n]); r3A = r12A;
        r12B = min([8, n*m, r]);    r1B = min([ceil(sqrt(r12B)), n]); r2B = min([ceil(sqrt(r12B)), m]); r3B = r12B;

        U1A = orth(randn(n, r1A));  U2A = orth(randn(n, r2A));  U3A = orth(randn(r, r3A));
        U1B = orth(randn(n, r1B));  U2B = orth(randn(m, r2B));  U3B = orth(randn(r, r3B));

        B12A = randn(r1A, r2A, r12A);
        BrootA = randn(r12A, r3A);

        B12B = randn(r1B, r2B, r12B);
        BrootB = randn(r12B, r3B);

        tic;
        [U1C,U2C,U3C,B12C,BrootC] = tprod_htd(U1A,U2A,U3A,B12A,BrootA, U1B,U2B,U3B,B12B,BrootB);
        t_htd_ht(trial,i) = toc;

        if n <= n_full_max_build
            A_full = htd3_to_full(U1A,U2A,U3A,B12A,BrootA);
            B_full = htd3_to_full(U1B,U2B,U3B,B12B,BrootB);

            if n <= n_full_max_def
                try
                    tic;
                    C_def = tprod_definition(A_full, B_full);
                    t_def_ht(trial,i) = toc;
                catch ME
                    fprintf('[Case3 Low HT-rank] tprod_definition FAILED at n=%d: %s\n', n, ME.message);
                    t_def_ht(trial,i) = NaN;
                    C_def = [];
                end
            end

            if has_tt
                TT_A = tt_tensor(A_full);
                TT_B = tt_tensor(B_full);
                try
                    TT_A = round(TT_A, tt_eps, tt_rmax);
                    TT_B = round(TT_B, tt_eps, tt_rmax);
                catch
                end
                [G1A,G2A,G3A] = extract_tt_cores_dim3(TT_A, n, n, r);
                [G1B,G2B,G3B] = extract_tt_cores_dim3(TT_B, n, m, r);

                tic;
                [G1C, G2C, G3C] = tprod_ttd(G1A,G2A,G3A, G1B,G2B,G3B, n, n, m, r);
                t_ttd_ht(trial,i) = toc;

                fprintf('trial=%d | i=%d | def=%.6f s | htd=%.6f s | ttd=%.6f s\n', ...
                    trial, i, ...
                    t_def_ht(trial,i), ...
                    t_htd_ht(trial,i), ...
                    t_ttd_ht(trial,i));
            end
        end
    end
end
fprintf('\n');

%% ------------------ averages (omit NaNs) ------------------
t_def_sparse_avg = mean(t_def_sparse,1,'omitnan');
t_ttd_sparse_avg = mean(t_ttd_sparse,1,'omitnan');
t_htd_sparse_avg = mean(t_htd_sparse,1,'omitnan');
err_ttd_sparse_max = max(err_ttd_sparse,[],1,'omitnan');
err_htd_sparse_max = max(err_htd_sparse,[],1,'omitnan');

t_def_tt_avg = mean(t_def_tt,1,'omitnan');
t_ttd_tt_avg = mean(t_ttd_tt,1,'omitnan');
t_htd_tt_avg = mean(t_htd_tt,1,'omitnan');

t_def_ht_avg = mean(t_def_ht,1,'omitnan');
t_ttd_ht_avg = mean(t_ttd_ht,1,'omitnan');
t_htd_ht_avg = mean(t_htd_ht,1,'omitnan');

fprintf('\nMax relative reconstruction error (Sparse case) across trials, per size:\n');
for i = 1:L
    fprintf('  n=%6d : errTTD=%.3e  errHTD=%.3e\n', n_array(i), err_ttd_sparse_max(i), err_htd_sparse_max(i));
end

%% ------------------ Save Results ------------------
outdir = fullfile(pwd, 'results');
if ~exist(outdir,'dir'); mkdir(outdir); end

jobid = getenv('SLURM_JOB_ID');
if isempty(jobid)
    jobid = datestr(now,'yyyymmdd_HHMMSS');
end

save(fullfile(outdir, sprintf('tprod_dim3_data_803_%s.mat', jobid)), ...
     'n_array', ...
     't_def_sparse_avg','t_htd_sparse_avg','t_ttd_sparse_avg', ...
     'err_ttd_sparse_max','err_htd_sparse_max', ...
     't_def_tt_avg','t_htd_tt_avg','t_ttd_tt_avg', ...
     't_def_ht_avg','t_htd_ht_avg','t_ttd_ht_avg', ...
     '-v7.3');

%% ------------------ plots ------------------
fig = figure('Position', [80, 120, 1320, 420]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');
ylabel(tl, 'Time (s)', 'FontSize', 16);

ax1 = nexttile(tl, 1);
plot_one_case_ax(ax1, n_array, t_def_sparse_avg, t_htd_sparse_avg, t_ttd_sparse_avg, ...
    sprintf('Case 1: Random Sparse (r=%d)', r), false);

ax2 = nexttile(tl, 2);
plot_one_case_ax(ax2, n_array, t_def_tt_avg, t_htd_tt_avg, t_ttd_tt_avg, ...
    sprintf('Case 2: Low TT-Rank (r=%d)', r), true);

ax3 = nexttile(tl, 3);
plot_one_case_ax(ax3, n_array, t_def_ht_avg, t_htd_ht_avg, t_ttd_ht_avg, ...
    sprintf('Case 3: Low HT-Rank (r=%d)', r), false);

yl = [min([ax1.YLim(1), ax2.YLim(1), ax3.YLim(1)]), max([ax1.YLim(2), ax2.YLim(2), ax3.YLim(2)])];
ax1.YLim = yl; ax2.YLim = yl; ax3.YLim = yl;
ax2.YTickLabel = [];
ax3.YTickLabel = [];

savefig(fig, fullfile(outdir, sprintf('tprod_dim3_panel_803_%s.fig', jobid)));

fprintf('\nSaved results and figure to folder: %s\n', outdir);

%% ============================================================
%% Local helper functions
%% ============================================================

function [G1,G2,G3] = extract_tt_cores_dim3(TT_A, n1, n2, n3)
cr = TT_A.core; ps = TT_A.ps;
r0 = TT_A.r(1); r1 = TT_A.r(2); r2 = TT_A.r(3); r3 = TT_A.r(4);
G1 = reshape(cr(ps(1):ps(2)-1), [r0, n1, r1]);
G2 = reshape(cr(ps(2):ps(3)-1), [r1, n2, r2]);
G3 = reshape(cr(ps(3):ps(4)-1), [r2, n3, r3]);
end

function A = tt3_to_full(G1,G2,G3)
[~,n1,r1] = size(G1);
[~,n2,r2] = size(G2);
[~,n3,~]  = size(G3);
M1 = reshape(G1,n1,r1);
T12 = reshape(M1*reshape(G2,r1,n2*r2), n1,n2,r2);
T12r = reshape(T12,n1*n2,r2);
A = real(reshape(T12r*reshape(G3,r2,n3), n1,n2,n3));
end

function A = htd3_to_full(U1,U2,U3,B12,Broot)
% Reconstruct full tensor from 3D HT factors with tree {1,2}--{3}
[n1,~] = size(U1);
[n2,~] = size(U2);
[r, r3] = size(U3);
r12 = size(B12,3);

r1 = size(B12,1); r2 = size(B12,2);
G = zeros(r1,r2,r3);
for a = 1:r3
    for s = 1:r12
        G(:,:,a) = G(:,:,a) + Broot(s,a) * B12(:,:,s);
    end
end

A = zeros(n1,n2,r);
for a = 1:r3
    S = U1 * G(:,:,a) * U2.';
    for t = 1:r
        A(:,:,t) = A(:,:,t) + U3(t,a) * S;
    end
end
end

function [U1,U2,U3,B12,Broot] = numeric_to_ht_factors_dim3(A, r1, r2, r12, tol)
% Tolerance-adaptive HT-like factorization for 3D tensor A (n1 x n2 x n3),
% tree {1,2}-{3}. r1,r2,r12 here are CEILINGS; the actual rank kept at
% each split is chosen via a relative-singular-value threshold (tol),
% mirroring the TT side's tolerance-driven round(..., tt_eps, tt_rmax).
if nargin < 5 || isempty(tol)
    tol = 1e-10;
end
[n1,n2,n3] = size(A);

A_mat = reshape(A, n1*n2, n3);
[U,S,V] = svd(A_mat, 'econ');
sv = diag(S);
r12 = min([r12, max(1, sum(sv > tol*sv(1))), size(S,1)]);
U12 = U(:,1:r12);
U3  = V(:,1:r12);
Broot = S(1:r12,1:r12);

T12 = reshape(U12, [n1, n2, r12]);

X1 = reshape(T12, n1, n2*r12);
[U1,S1,~] = svd(X1, 'econ');
sv1 = diag(S1);
r1 = min([r1, max(1, sum(sv1 > tol*sv1(1))), size(U1,2)]);
U1 = U1(:, 1:r1);

X2 = reshape(permute(T12,[2 1 3]), n2, n1*r12);
[U2,S2,~] = svd(X2, 'econ');
sv2 = diag(S2);
r2 = min([r2, max(1, sum(sv2 > tol*sv2(1))), size(U2,2)]);
U2 = U2(:, 1:r2);

r1 = size(U1,2); r2 = size(U2,2);
B12 = zeros(r1, r2, r12);
for k = 1:r12
    B12(:,:,k) = U1' * T12(:,:,k) * U2;
end
end

function plot_one_case_ax(ax, n_array, t_def_avg, t_htd_avg, t_ttd_avg, ttl, show_xlabel)

c_def = [0.0, 0.45, 0.70];
c_htd = [0.0, 0.60, 0.50];
c_ttd = [0.90, 0.60, 0.00];

FS_tick   = 14;
FS_label  = 16;
FS_title  = 18;
FS_legend = 13;
LW        = 2.2;
MS        = 7;

axes(ax); %#ok<LAXES>
hold(ax, 'on');

semilogy(ax, log2(n_array), t_def_avg, '-o', 'Color', c_def, 'LineWidth', LW, 'MarkerSize', MS);
semilogy(ax, log2(n_array), t_htd_avg, '-d', 'Color', c_htd, 'LineWidth', LW, 'MarkerSize', MS);
semilogy(ax, log2(n_array), t_ttd_avg, '-x', 'Color', c_ttd, 'LineWidth', LW, 'MarkerSize', MS+1);

xlim(ax, [log2(n_array(1)), log2(n_array(end))]);
xt = log2(n_array);
xticks(ax, xt);
xticklabels(ax, arrayfun(@(e) sprintf('2^{%d}', e), xt, 'UniformOutput', false));

set(ax, 'YScale', 'log');
set(ax, 'FontSize', FS_tick);

title(ax, ttl, 'FontSize', FS_title);

if show_xlabel
    xlabel(ax, 'Dimension n_1 = n_2', 'Interpreter', 'tex', 'FontSize', FS_label);
else
    ax.XLabel.String = "";
end

lgd = legend(ax, {'Definition-based','HTD-based T-product','TTD-based T-product'}, ...
    'Location', 'northwest', 'FontSize', FS_legend, 'Box', 'off');
lgd.Units = 'normalized';

grid(ax, 'off');
hold(ax, 'off');
ax.XTickMode = 'manual';
ax.XTickLabelMode = 'manual';
end
