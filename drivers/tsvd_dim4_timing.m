%% TSVD_DIM4_TIMING  4th-order T-SVD timing/accuracy comparison:
% definition-based vs TTD-based vs HTD-based, across three tensor-
% generation regimes and a range of problem sizes.
%
% What it does:
%   1. Runs a one-time, small-scale (n1=n2=16) self-consistency check:
%      confirms tsvd_ttd_dim4.m's reconstruction agrees with the
%      independent tsvd_ttd_dim4_ref.m implementation, and that both the
%      TTD- and HTD-based reconstructions agree with a directly-generated
%      tensor to <=1e-6 relative error, before running any timed sweep.
%   2. Compute computational time for each of three cases (1) Random Sparse (fixed nnz=30,
%      independent of tensor size), (2) Low TT-rank, (3) Low HT-rank of size n1=n2 in 2.^(2:13) (n3=n4=4 fixed):
%        - definition_tsvd_dim4 (local function): full per-frequency
%          dense SVD after a 2-D FFT along modes 3,4
%        - tsvd_ttd_dim4.m (TTD-based, computed from TT cores)
%        - tsvd_htd_dim4.m (HTD-based, computed from HT factors)
%      and (for the Sparse case, and wherever else feasible) records the
%      reconstruction relative error of the TTD-/HTD-based T-SVD against
%      the exact input tensor.
%   3. Saves timing/accuracy tables and a 3-panel log-log timing figure.
%
% Required files: tsvd_ttd_dim4.m, tsvd_ttd_dim4_ref.m (validation
% only), tsvd_htd_dim4.m, plus the TT-Toolbox (tt_tensor/tt_rand/round)
% and htucker (htenrandn/orthog) toolboxes for tensor generation.
%
% Outputs (to results/): tsvd_dim4_803_<id>.mat (timing/accuracy arrays) and
% tsvd_dim4_panel_803_<id>.fig (3-panel figure, one panel per case).
%
% Notes:
%   - Definition-based and TTD-based (Sparse case only) are skipped
%     (recorded as NaN) above fixed size cutoffs / a dynamic memory-
%     budget estimate, to avoid OOM crashing the whole sweep; see
%     n_full_max_def / n_full_max_ttd_sparse / mem_budget_gb.
%   - n3=n4=4 are fixed across the whole sweep (only n1=n2 grows), so
%     definition-based cost here grows like the SVD of an n1 x n2 matrix,
%     NOT like n1^4 -- that quartic-in-n regime only appears if n3,n4 are
%     also scaled up to n (see the paper's complexity remarks, which
%     assume n1=n2=n3=n).

clc; clear; close all;

%% ------------------ Shared experiment settings ------------------
powers = 2:13;               % _803: sweep now starts at n1=n2=4 (was 3:13)
sizes = 2.^powers;
num_trials = 5;

n_array = sizes;
n_full_max_def = 2.^13;      % _803: attempt across full sweep; try/catch below records OOM as NaN
n_full_max_ttd_sparse = 2.^13;  % _803: raised from 2^12 -- root cause (rank blowup) fixed by nnz cap above; dynamic memory guard remains as the safety net
n3 = 4;
n4 = 4;
n_trial = num_trials;

tol_svd = 1e-12;

tt_eps   = 1e-10;
tt_rmax  = 1e5;      % ceiling only; round() is tolerance-driven (tt_eps)
ht_rel_eps = 1e-10;  % HTD side: same tolerance, tolerance-driven (see numeric_to_ht_factors_dim4)
ht_rmax  = 1e5;      % ceiling only, rarely binds once tolerance-driven

target_nnz_sparse = 30;  % _803: FIXED nonzero count (was density=1e-4 growing with n)

% Memory guard for the Sparse case: skip (NaN) if the estimated dominant
% intermediate array would exceed this budget. Set well below the SLURM
% job's --mem allocation to leave headroom for MATLAB overhead.
mem_budget_gb = 150;

%% ------------------ Small validation (correctness), OUTSIDE any timing ------------------
fprintf('\n=== Validation: core-form (_729 vs _730) reconstruction residuals (n1=n2=16) ===\n');
n1v = 16; n2v = 16;
val_tol = tol_svd;
val_errs = [];

val_tt_rank = [1, 2, 2, 2, 1];
TT_Av = tt_rand([n1v, n2v, n3, n4], 4, val_tt_rank);
Av_full_tt = reshape(full(TT_Av), [n1v, n2v, n3, n4]);
[Gv1, Gv2, Gv3, Gv4] = extract_tt_cores_dim4(TT_Av, n1v, n2v, n3, n4);

[U_rep_tt, S_rep_tt, V_rep_tt] = tsvd_ttd_dim4(Gv1,Gv2,Gv3,Gv4,n1v,n2v,n3,n4,val_tol);
Urec_tt = tt_contract4_729(U_rep_tt.cores);
Srec_tt = tt_contract4_729(S_rep_tt.cores);
Vrec_tt = tt_contract4_729(V_rep_tt.cores);

[Arec_ttd_tt, imrel_ttd_tt] = reconstruct_tensor_from_svd_729(Urec_tt, Srec_tt, Vrec_tt);
errA_ttd_tt = norm(Arec_ttd_tt(:)-Av_full_tt(:))/max(1,norm(Av_full_tt(:)));
fprintf('[Validation][TTD-730 reconstruct T][low TT-rank] relA=%.3e  imag_rel=%.3e\n', errA_ttd_tt, imrel_ttd_tt);
val_errs = [val_errs, errA_ttd_tt]; %#ok<AGROW>

% Cross-check against _729 on the same input (must agree on the RECONSTRUCTED
% tensor -- individual U/S/V columns may differ by sign/phase, that's fine).
[U_rep_729, S_rep_729, V_rep_729] = tsvd_ttd_dim4_ref(Gv1,Gv2,Gv3,Gv4,n1v,n2v,n3,n4,val_tol);
Urec_729 = tt_contract4_729(U_rep_729.cores);
Srec_729 = tt_contract4_729(S_rep_729.cores);
Vrec_729 = tt_contract4_729(V_rep_729.cores);
[Arec_729, ~] = reconstruct_tensor_from_svd_729(Urec_729, Srec_729, Vrec_729);
err_729_vs_730 = norm(Arec_729(:)-Arec_ttd_tt(:))/max(1,norm(Arec_729(:)));
fprintf('[Validation][_729 vs _730 agreement][low TT-rank] relA=%.3e\n', err_729_vs_730);
val_errs = [val_errs, err_729_vs_730]; %#ok<AGROW>

r12v = min([ht_rmax, n1v*n2v, n3*n4]);
r1v = min(ht_rmax, n1v); r2v = min(ht_rmax, n2v);
r3v = min(ht_rmax, n3);  r4v = min(ht_rmax, n4);
[U1v,U2v,U3v,U4v,B12v,B34v,Brootv] = numeric_to_ht_factors_dim4(Av_full_tt, r1v,r2v,r3v,r4v,r12v, ht_rel_eps);
Av_full_ht = htd4_to_full_729(U1v,U2v,U3v,U4v,B12v,B34v,Brootv);

[U_rep_ht, S_rep_ht, V_rep_ht] = tsvd_htd_dim4(U1v,U2v,U3v,U4v,B12v,B34v,Brootv,val_tol);
Urec_ht = htd_reconstruct_729(U_rep_ht);
Srec_ht = htd_reconstruct_729(S_rep_ht);
Vrec_ht = htd_reconstruct_729(V_rep_ht);

[Arec_htd, imrel_htd] = reconstruct_tensor_from_svd_729(Urec_ht, Srec_ht, Vrec_ht);
errA_htd = norm(Arec_htd(:)-Av_full_ht(:))/max(1,norm(Av_full_ht(:)));
fprintf('[Validation][HTD-729 reconstruct T][HT approx] relA=%.3e  imag_rel=%.3e\n', errA_htd, imrel_htd);
val_errs = [val_errs, errA_htd]; %#ok<AGROW>

[Udef, Sdef, Vdef] = definition_tsvd_dim4(Av_full_tt, val_tol);
[Arec_def, imrel_def] = reconstruct_tensor_from_svd_729(Udef, Sdef, Vdef);
errA_def = norm(Arec_def(:)-Av_full_tt(:))/max(1,norm(Av_full_tt(:)));
fprintf('[Validation][definition-based reconstruct T][low TT-rank] relA=%.3e  imag_rel=%.3e\n', errA_def, imrel_def);
val_errs = [val_errs, errA_def]; %#ok<AGROW>

if max(val_errs) > 1e-6
    warning('compare_tsvd_dim4_sparse_tt_ht_v4_730:validationTolerance', ...
        'Reconstruction relative error exceeded 1e-6 (max=%.3e).', max(val_errs));
else
    fprintf('All reconstruction relative errors <= 1e-6 (max=%.3e).\n', max(val_errs));
end
fprintf('=== Validation complete ===\n\n');
clear TT_Av Av_full_tt Gv1 Gv2 Gv3 Gv4 U_rep_tt S_rep_tt V_rep_tt Urec_tt Srec_tt Vrec_tt ...
      Arec_ttd_tt imrel_ttd_tt errA_ttd_tt U_rep_729 S_rep_729 V_rep_729 Urec_729 Srec_729 Vrec_729 ...
      Arec_729 err_729_vs_730 r12v r1v r2v r3v r4v U1v U2v U3v U4v B12v B34v Brootv ...
      Av_full_ht U_rep_ht S_rep_ht V_rep_ht Urec_ht Srec_ht Vrec_ht Arec_htd imrel_htd errA_htd ...
      Udef Sdef Vdef Arec_def imrel_def errA_def val_errs n1v n2v val_tol val_tt_rank

%% ------------------ CASE 2: Low TT-rank tensor ------------------
tt_rank = [1, 2, 2, 2, 1];

t_def_tt = zeros(n_trial, numel(n_array));
t_htd_tt = zeros(n_trial, numel(n_array));
t_ttd_tt = zeros(n_trial, numel(n_array));

for trial = 1:n_trial
  fprintf('[Low TT-rank] Trial %d/%d\n', trial, n_trial);
  for i = 1:numel(n_array)
    n1 = n_array(i); n2 = n_array(i);

    TT_A = tt_rand([n1, n2, n3, n4], 4, tt_rank);
    A_full = reshape(full(TT_A), [n1, n2, n3, n4]);
    [G1, G2, G3, G4] = extract_tt_cores_dim4(TT_A, n1, n2, n3, n4);

    r12 = min([ht_rmax, n1*n2, n3*n4]);
    r1  = min(ht_rmax, n1);
    r2  = min(ht_rmax, n2);
    r3  = min(ht_rmax, n3);
    r4  = min(ht_rmax, n4);
    [U1,U2,U3,U4,B12,B34,Broot] = numeric_to_ht_factors_dim4(A_full, r1,r2,r3,r4,r12, ht_rel_eps);

    if n1 <= n_full_max_def
      try
        tic;
        [~,~,~] = definition_tsvd_dim4(A_full, tol_svd);
        t_def_tt(trial,i) = toc;
      catch ME
        fprintf('[Low TT-rank] definition_tsvd_dim4 failed at n1=%d: %s\n', n1, ME.message);
        t_def_tt(trial,i) = NaN;
      end
    else
      t_def_tt(trial,i) = NaN;
    end

    tic;
    [~,~,~] = tsvd_htd_dim4(U1,U2,U3,U4,B12,B34,Broot,tol_svd);
    t_htd_tt(trial,i) = toc;

    tic;
    [~,~,~] = tsvd_ttd_dim4(G1,G2,G3,G4,n1,n2,n3,n4,tol_svd);
    t_ttd_tt(trial,i) = toc;
    fprintf('trial=%d | i=%d | def=%.6f s | htd=%.6f s | ttd=%.6f s\n', ...
            trial, i, t_def_tt(trial,i), t_htd_tt(trial,i), t_ttd_tt(trial,i));
  end
  fprintf('Trial %d/%d done.\n', trial, n_trial);
end

t_def_tt_avg = nanmean(t_def_tt,1);
t_htd_tt_avg = nanmean(t_htd_tt,1);
t_ttd_tt_avg = nanmean(t_ttd_tt,1);

%% ------------------ CASE 3: Low HTD-rank tensor ------------------
ht_ranks_vec = [2 2 2 2 2 2 2];

t_def_ht = zeros(n_trial, numel(n_array));
t_htd_ht = zeros(n_trial, numel(n_array));
t_ttd_ht = zeros(n_trial, numel(n_array));

for trial = 1:n_trial
  fprintf('[Low HT-rank] Trial %d/%d\n', trial, n_trial);
  for i = 1:numel(n_array)
    n1 = n_array(i); n2 = n_array(i);

    A_ht = htenrandn([n1,n2,n3,n4], '', ht_ranks_vec);
    A_ht = orthog(A_ht);
    A_full = full(A_ht);

    [U1,U2,U3,U4,B12,B34,Broot] = extract_ht_components_dim4(A_ht);

    TT_A = tt_tensor(reshape(A_full,[n1,n2,n3,n4]));
    try
      TT_A = round(TT_A, tt_eps, tt_rmax);
    catch
    end
    [G1, G2, G3, G4] = extract_tt_cores_dim4(TT_A, n1, n2, n3, n4);

    if n1 <= n_full_max_def
      try
        tic;
        [~,~,~] = definition_tsvd_dim4(A_full, tol_svd);
        t_def_ht(trial,i) = toc;
      catch ME
        fprintf('[Low HT-rank] definition_tsvd_dim4 failed at n1=%d: %s\n', n1, ME.message);
        t_def_ht(trial,i) = NaN;
      end
    else
      t_def_ht(trial,i) = NaN;
    end

    tic;
    [~,~,~] = tsvd_htd_dim4(U1,U2,U3,U4,B12,B34,Broot,tol_svd);
    t_htd_ht(trial,i) = toc;

    tic;
    [~,~,~] = tsvd_ttd_dim4(G1,G2,G3,G4,n1,n2,n3,n4,tol_svd);
    t_ttd_ht(trial,i) = toc;
    fprintf('trial=%d | i=%d | def=%.6f s | htd=%.6f s | ttd=%.6f s\n', ...
            trial, i, t_def_ht(trial,i), t_htd_ht(trial,i), t_ttd_ht(trial,i));
  end
  fprintf('Trial %d/%d done.\n', trial, n_trial);
end

t_def_ht_avg = nanmean(t_def_ht,1);
t_htd_ht_avg = nanmean(t_htd_ht,1);
t_ttd_ht_avg = nanmean(t_ttd_ht,1);

%% ------------------ CASE 1: Random sparse tensor (accuracy-fixed) ------------------
t_def_sparse = zeros(n_trial, numel(n_array));
t_htd_sparse = zeros(n_trial, numel(n_array));
t_ttd_sparse = zeros(n_trial, numel(n_array));
err_htd_sparse = nan(n_trial, numel(n_array));
err_ttd_sparse = nan(n_trial, numel(n_array));

for trial = 1:n_trial
  fprintf('[Sparse] Trial %d/%d\n', trial, n_trial);
  for i = 1:numel(n_array)
    n1 = n_array(i); n2 = n_array(i);

    N = n1 * n2 * n3 * n4;
    density_sparse = min(1, target_nnz_sparse / N);
    nnz_min = max(1, round(density_sparse * N));
    idx = randperm(N, nnz_min);
    vals = randn(nnz_min, 1);

    A_full = zeros(N, 1);
    A_full(idx) = vals;
    A_full = reshape(A_full, [n1, n2, n3, n4]);
    A_full = A_full / max(1, norm(A_full(:)));

    TT_A = tt_tensor(A_full);
    try
      TT_A = round(TT_A, tt_eps, tt_rmax);
    catch
    end
    [G1, G2, G3, G4] = extract_tt_cores_dim4(TT_A, n1, n2, n3, n4);

    [U1,U2,U3,U4,B12,B34,Broot] = numeric_to_ht_factors_dim4(A_full, ...
        ht_rmax, ht_rmax, ht_rmax, ht_rmax, ht_rmax, ht_rel_eps);

    if n1 <= n_full_max_def
      try
        tic;
        [~,~,~] = definition_tsvd_dim4(A_full, tol_svd);
        t_def_sparse(trial,i) = toc;
      catch ME
        fprintf('[Sparse] definition_tsvd_dim4 failed at n1=%d: %s\n', n1, ME.message);
        t_def_sparse(trial,i) = NaN;
      end
    else
      t_def_sparse(trial,i) = NaN;
    end

    % Memory guard (dynamic) + hard cap (NEW, n1 <= n_full_max_ttd_sparse):
    % the achieved TT rank r1 (analogous to TERA's near-full-rank finding
    % for fixed-density sparse tensors) drives the per-frequency SVD cost;
    % _730's mode-2 QR doesn't help once r1,r2 are themselves large, so the
    % explicit n1 cutoff is kept as a first line of defense ahead of the
    % dynamic estimated-memory check.
    r1_tt = TT_A.r(2);
    mem_gb_ttd = r1_tt * n2 * (n3*n4) * 16 / 1e9;
    r1_ht = size(U1,2); r2_ht = size(U2,2);
    mem_gb_htd = r1_ht * r2_ht * (n3*n4) * 16 / 1e9;

    if n1 > n_full_max_ttd_sparse
      fprintf('[Sparse] TTD-based SKIPPED at n1=%d: exceeds hard cap n_full_max_ttd_sparse=%d\n', ...
          n1, n_full_max_ttd_sparse);
      t_ttd_sparse(trial,i) = NaN;
    elseif mem_gb_ttd > mem_budget_gb
      fprintf('[Sparse] TTD-based SKIPPED at n1=%d: estimated memory %.1f GB > budget %.1f GB (TT rank r1=%d)\n', ...
          n1, mem_gb_ttd, mem_budget_gb, r1_tt);
      t_ttd_sparse(trial,i) = NaN;
    else
      tic;
      [U_rep_t, S_rep_t, V_rep_t] = tsvd_ttd_dim4(G1,G2,G3,G4,n1,n2,n3,n4,tol_svd);
      t_ttd_sparse(trial,i) = toc;
    end

    if mem_gb_htd > mem_budget_gb
      fprintf('[Sparse] HTD-based SKIPPED at n1=%d: estimated memory %.1f GB > budget %.1f GB (HT ranks r1=%d r2=%d)\n', ...
          n1, mem_gb_htd, mem_budget_gb, r1_ht, r2_ht);
      t_htd_sparse(trial,i) = NaN;
    else
      tic;
      [U_rep_h, S_rep_h, V_rep_h] = tsvd_htd_dim4(U1,U2,U3,U4,B12,B34,Broot,tol_svd);
      t_htd_sparse(trial,i) = toc;
    end

    % ---- Accuracy check (outside timing): reconstruction residual vs A_full ----
    if mem_gb_htd <= mem_budget_gb
      try
        Uh = htd_reconstruct_729(U_rep_h); Sh = htd_reconstruct_729(S_rep_h); Vh = htd_reconstruct_729(V_rep_h);
        [Arec_h, ~] = reconstruct_tensor_from_svd_729(Uh, Sh, Vh);
        err_htd_sparse(trial,i) = norm(Arec_h(:)-A_full(:))/max(1,norm(A_full(:)));
      catch ME
        fprintf('[Sparse] HTD accuracy check failed at n1=%d: %s\n', n1, ME.message);
      end
    end
    if n1 <= n_full_max_ttd_sparse && mem_gb_ttd <= mem_budget_gb
      try
        Ut = tt_contract4_729(U_rep_t.cores); St = tt_contract4_729(S_rep_t.cores); Vt = tt_contract4_729(V_rep_t.cores);
        [Arec_t, ~] = reconstruct_tensor_from_svd_729(Ut, St, Vt);
        err_ttd_sparse(trial,i) = norm(Arec_t(:)-A_full(:))/max(1,norm(A_full(:)));
      catch ME
        fprintf('[Sparse] TTD accuracy check failed at n1=%d: %s\n', n1, ME.message);
      end
    end

    fprintf('trial=%d | i=%d | n1=%d | def=%.6f s | htd=%.6f s | ttd=%.6f s | errHTD=%.3e | errTTD=%.3e\n', ...
            trial, i, n1, ...
            t_def_sparse(trial,i), t_htd_sparse(trial,i), t_ttd_sparse(trial,i), ...
            err_htd_sparse(trial,i), err_ttd_sparse(trial,i));
  end
  fprintf('Trial %d/%d done.\n', trial, n_trial);
end

t_def_sparse_avg = nanmean(t_def_sparse,1);
t_htd_sparse_avg = nanmean(t_htd_sparse,1);
t_ttd_sparse_avg = nanmean(t_ttd_sparse,1);
err_htd_sparse_max = max(err_htd_sparse,[],1,'omitnan');
err_ttd_sparse_max = max(err_ttd_sparse,[],1,'omitnan');

fprintf('\nMax relative reconstruction error (Sparse case) across trials, per size:\n');
for i = 1:numel(n_array)
    fprintf('  n=%6d : errHTD=%.3e  errTTD=%.3e\n', n_array(i), err_htd_sparse_max(i), err_ttd_sparse_max(i));
end

%% save
outdir = fullfile(pwd, 'results');
if ~exist(outdir, 'dir'); mkdir(outdir); end

jobid = getenv('SLURM_JOB_ID');
if isempty(jobid); jobid = datestr(now,'yyyymmdd_HHMMSS'); end

save(fullfile(outdir, ['tsvd_dim4_803_' jobid '.mat']), ...
     'n_array', 'n3', 'n4', 'n_trial', ...
     't_def_sparse_avg', 't_htd_sparse_avg', 't_ttd_sparse_avg', ...
     'err_htd_sparse_max', 'err_ttd_sparse_max', ...
     't_def_tt_avg', 't_htd_tt_avg', 't_ttd_tt_avg', ...
     't_def_ht_avg', 't_htd_ht_avg', 't_ttd_ht_avg');

%% ------------------ plots ------------------
fig = figure('Position', [80, 120, 1320, 420]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing','compact', 'Padding','compact');
ylabel(tl, 'Time (s)', 'FontSize', 16);

ax1 = nexttile(tl, 1);
plot_one_case_ax(ax1, n_array, t_def_sparse_avg, t_htd_sparse_avg, t_ttd_sparse_avg, ...
    sprintf('Case 1: Random Sparse'), false);

ax2 = nexttile(tl, 2);
plot_one_case_ax(ax2, n_array, t_def_tt_avg, t_htd_tt_avg, t_ttd_tt_avg, ...
    sprintf('Case 2: Low TT-Rank'), true);

ax3 = nexttile(tl, 3);
plot_one_case_ax(ax3, n_array, t_def_ht_avg, t_htd_ht_avg, t_ttd_ht_avg, ...
    sprintf('Case 3: Low HT-Rank'), false);

yl = [min([ax1.YLim(1), ax2.YLim(1), ax3.YLim(1)]), max([ax1.YLim(2), ax2.YLim(2), ax3.YLim(2)])];
ax1.YLim = yl; ax2.YLim = yl; ax3.YLim = yl;
ax2.YTickLabel = [];
ax3.YTickLabel = [];

savefig(fig, fullfile(outdir, sprintf('tsvd_dim4_panel_803_%s.fig', jobid)));

%% ======================= Local Functions =======================

function [u_tensor, s_tensor, v_tensor] = definition_tsvd_dim4(A_full, tol)
[n1,n2,n3,n4] = size(A_full);
Ahat = fft(fft(A_full,[],3),[],4);

U_fft = cell(n3,n4);
S_fft = cell(n3,n4);
V_fft = cell(n3,n4);

for k3 = 1:n3
  for k4 = 1:n4
    Ak = Ahat(:,:,k3,k4);
    [Uk,Sk,Vk] = svd(Ak,'econ');
    s = max(1, sum(diag(Sk) > tol));
    U_fft{k3,k4} = Uk(:,1:s);
    S_fft{k3,k4} = Sk(1:s,1:s);
    V_fft{k3,k4} = Vk(:,1:s);
  end
end

rmax = max(cellfun(@(x) size(x,2), U_fft(:)));
U_big = zeros(n1, rmax, n3, n4);
S_big = zeros(rmax, rmax, n3, n4);
V_big = zeros(n2, rmax, n3, n4);

for k3 = 1:n3
  for k4 = 1:n4
    rk = size(U_fft{k3,k4},2);
    U_big(:,1:rk,k3,k4) = U_fft{k3,k4};
    S_big(1:rk,1:rk,k3,k4) = S_fft{k3,k4};
    V_big(:,1:rk,k3,k4) = V_fft{k3,k4};
  end
end

u_tensor = ifft(ifft(U_big,[],3),[],4,'symmetric');
s_tensor = ifft(ifft(S_big,[],3),[],4,'symmetric');
v_tensor = ifft(ifft(V_big,[],3),[],4,'symmetric');
end

function [G1,G2,G3,G4] = extract_tt_cores_dim4(TT_A, n1, n2, n3, n4)
crA = TT_A.core; psA = TT_A.ps;
G1 = reshape(crA(psA(1):psA(2)-1), [TT_A.r(1), n1, TT_A.r(2)]);
G2 = reshape(crA(psA(2):psA(3)-1), [TT_A.r(2), n2, TT_A.r(3)]);
G3 = reshape(crA(psA(3):psA(4)-1), [TT_A.r(3), n3, TT_A.r(4)]);
G4 = reshape(crA(psA(4):psA(5)-1), [TT_A.r(4), n4, TT_A.r(5)]);
end

function [U1,U2,U3,U4,B12,B34,Broot] = extract_ht_components_dim4(A_ht)
leaf_indices = A_ht.dim2ind;
U1 = A_ht.U{leaf_indices(1)};
U2 = A_ht.U{leaf_indices(2)};
U3 = A_ht.U{leaf_indices(3)};
U4 = A_ht.U{leaf_indices(4)};

B12_idx = A_ht.parent(leaf_indices(1));
B12 = A_ht.B{B12_idx};

B34_idx = A_ht.parent(leaf_indices(3));
B34 = A_ht.B{B34_idx};

Broot = A_ht.B{1};
end

function [U1,U2,U3,U4,B12,B34,Broot] = numeric_to_ht_factors_dim4(A, r1,r2,r3,r4,r12, tol)
% Tolerance-adaptive HTucker factors for a 4D numeric tensor A, tree
% {1,2} -- {3,4}. r1,r2,r3,r4,r12 here are CEILINGS; the actual rank kept
% at each split is chosen via a relative-singular-value threshold (tol),
% mirroring the TT side's tolerance-driven round(..., tt_eps, tt_rmax).
if nargin < 7 || isempty(tol)
    tol = 1e-10;
end
[n1,n2,n3,n4] = size(A);

A_mat = reshape(A, n1*n2, n3*n4);
[U,S,V] = svd(A_mat, 'econ');
sv = diag(S);
r12_auto = max(1, sum(sv > tol*sv(1)));
r12 = min([r12, r12_auto, size(S,1)]);
U12 = U(:,1:r12); V34 = V(:,1:r12); S12 = S(1:r12,1:r12);
Broot = reshape(S12, [r12, r12, 1]);

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

T34 = reshape(V34, [n3, n4, r12]);
Y3 = reshape(T34, n3, n4*r12);
[U3,S3,~] = svd(Y3, 'econ');
sv3 = diag(S3);
r3 = min([r3, max(1, sum(sv3 > tol*sv3(1))), size(U3,2)]);
U3 = U3(:, 1:r3);

Y4 = reshape(permute(T34,[2 1 3]), n4, n3*r12);
[U4,S4,~] = svd(Y4, 'econ');
sv4 = diag(S4);
r4 = min([r4, max(1, sum(sv4 > tol*sv4(1))), size(U4,2)]);
U4 = U4(:, 1:r4);

r3 = size(U3,2); r4 = size(U4,2);
B34 = zeros(r3, r4, r12);
for k = 1:r12
    B34(:,:,k) = U3' * T34(:,:,k) * U4;
end

if any(~isfinite(U1(:))) || any(~isfinite(U2(:))) || any(~isfinite(U3(:))) || any(~isfinite(U4(:))) || ...
   any(~isfinite(B12(:))) || any(~isfinite(B34(:))) || any(~isfinite(Broot(:)))
    error('numeric_to_ht_factors_dim4 produced NaN/Inf (unexpected). Check A for NaN/Inf.');
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

lgd = legend(ax, {'Definition-based','HTD-based T-SVD','TTD-based T-SVD'}, ...
    'Location', 'northwest', 'FontSize', FS_legend, 'Box', 'off');
lgd.Units = 'normalized';

grid(ax, 'off');
hold(ax, 'off');
ax.XTickMode = 'manual';
ax.XTickLabelMode = 'manual';
end

%% ======================= _729 validation-only helper functions =======================

function A = resolve_core_729(C)
if isstruct(C)
    switch C.type
        case 'identity_tt_core'
            n = C.physical_size; m = C.rank_size;
            A = zeros(1, n, m);
            for k = 1:min(n,m)
                A(1,k,k) = 1;
            end
        case 'pairing_transfer'
            L = C.left_size; R = C.right_size; mult = C.left_multiplier;
            A = zeros(L, R, C.output_size);
            for b = 1:R
                for a = 1:L
                    idx = a + mult*(b-1);
                    A(a,b,idx) = 1;
                end
            end
        otherwise
            error('resolve_core_729:unknownType', 'Unknown core struct type "%s".', C.type);
    end
else
    A = C;
end
end

function T = tt_contract4_729(cores)
C1 = resolve_core_729(cores{1});
C2 = resolve_core_729(cores{2});
C3 = resolve_core_729(cores{3});
C4 = resolve_core_729(cores{4});

[~, d1, r1] = size(C1);
[~, d2, r2] = size(C2);
[~, d3, r3] = size(C3);
[~, d4, ~ ] = size(C4);

M1  = reshape(C1, d1, r1);
T12 = reshape(M1 * reshape(C2, r1, d2*r2), d1, d2, r2);

T12r  = reshape(T12, d1*d2, r2);
T123  = reshape(T12r * reshape(C3, r2, d3*r3), d1, d2, d3, r3);

T123r = reshape(T123, d1*d2*d3, r3);
T     = reshape(T123r * reshape(C4, r3, d4), d1, d2, d3, d4);
end

function T = htd_reconstruct_729(rep)
L1 = rep.leaves{1}; L2 = rep.leaves{2}; L3 = rep.leaves{3}; L4 = rep.leaves{4};
B12 = resolve_core_729(rep.transfers.B12);
B34 = resolve_core_729(rep.transfers.B34);
Broot = rep.transfers.Broot;

[n1_, r1_] = size(L1);
[n2_, r2_] = size(L2);
[n3_, r3_] = size(L3);
[n4_, r4_] = size(L4);
r12 = size(B12,3);  Nf = size(B34,3);

U12 = zeros(n1_, n2_, r12);
for rho12 = 1:r12
    for a2 = 1:r2_
        for a1 = 1:r1_
            c = B12(a1,a2,rho12);
            if c ~= 0
                U12(:,:,rho12) = U12(:,:,rho12) + c * (L1(:,a1) * L2(:,a2).');
            end
        end
    end
end

U34 = zeros(n3_, n4_, Nf);
for rhoF = 1:Nf
    for a4 = 1:r4_
        for a3 = 1:r3_
            c = B34(a3,a4,rhoF);
            if c ~= 0
                U34(:,:,rhoF) = U34(:,:,rhoF) + c * (L3(:,a3) * L4(:,a4).');
            end
        end
    end
end

T = zeros(n1_, n2_, n3_, n4_);
for rhoF = 1:Nf
    for rho12 = 1:r12
        c = Broot(rho12,rhoF,1);
        if c ~= 0
            T = T + reshape(c * U12(:,:,rho12), n1_,n2_,1,1) .* reshape(U34(:,:,rhoF), 1,1,n3_,n4_);
        end
    end
end
end

function [A, im_rel] = reconstruct_tensor_from_svd_729(U, S, V)
[n1_, ~, n3_, n4_] = size(U);
n2_ = size(V,1);

Uhat = fft(fft(U,[],3),[],4);
Shat = fft(fft(S,[],3),[],4);
Vhat = fft(fft(V,[],3),[],4);

Chat = complex(zeros(n1_, n2_, n3_, n4_));
for beta4 = 1:n4_
    for beta3 = 1:n3_
        Chat(:,:,beta3,beta4) = Uhat(:,:,beta3,beta4) * Shat(:,:,beta3,beta4) * Vhat(:,:,beta3,beta4)';
    end
end

Acplx = ifft(ifft(Chat,[],3),[],4);
im_rel = norm(imag(Acplx(:))) / max(1, norm(Acplx(:)));
A = real(Acplx);
end

function A = htd4_to_full_729(U1,U2,U3,U4,B12,B34,Broot)
[n1_,r1_] = size(U1); [n2_,r2_] = size(U2);
[n3_,r3_] = size(U3); [n4_,r4_] = size(U4);
r12 = size(B12,3); r34 = size(B34,3);

G = zeros(r1_,r2_,r3_,r4_);
for a = 1:r12
    for b = 1:r34
        coeff = Broot(a,b,1);
        if coeff ~= 0
            G = G + coeff .* (reshape(B12(:,:,a),r1_,r2_,1,1) .* reshape(B34(:,:,b),1,1,r3_,r4_));
        end
    end
end

A = zeros(n1_,n2_,n3_,n4_);
for a3 = 1:r3_
    for a4 = 1:r4_
        Smat = U1 * G(:,:,a3,a4) * U2.';
        for j3 = 1:n3_
            for j4 = 1:n4_
                A(:,:,j3,j4) = A(:,:,j3,j4) + U3(j3,a3)*U4(j4,a4)*Smat;
            end
        end
    end
end
end
