%% ---------------------------------------------------------------
%  T-ERA timing benchmark (ERA vs T-ERA baseline vs T-ERA+TTD/HTD tsvd(H))
%  + three system cases: random sparse, low-TT rank, low-HT rank
%
%  Successor to Test_TERA_3_730.m. Change in this version (_803):
%    Sparse case (case 1) Hankel generation: nonzero COUNT is now capped
%    at a fixed target_nnz (~30), independent of N=nRows*nCols*s, instead
%    of a fixed density (1e-4) that let nnz -- and the achieved TT/HT
%    rank -- grow with problem size. Same fix validated for T-product/
%    T-SVD (pinning nnz keeps rank flat, restoring TTD/HTD's speedup over
%    the baseline at large scale). All other logic, including H_list/
%    n_array (unchanged -- only T-product/T-SVD had their size sweep
%    extended), is carried forward unchanged from _730.
%
%  Prior successor note (Test_TERA_3_729_10.m -> _730): the 'ttd' branch now goes through
%  tera_reduce (-> tsvd_ttd_dim3), which additionally
%  compresses mode 2 of the Hankel tensor onto an orthogonal basis via QR
%  of the mode-2 unfolding of G2 -- exact, not approximate (see that
%  function's header for the derivation: a TT network-cut bound plus the
%  isometry-preserves-SVD identity). In _729/_729_10, TTD's per-frequency
%  reduced matrix was r1-by-n2 (mode 2 uncompressed, an implicit n2 x n2
%  identity core), so each of the s per-frequency SVDs cost O(r1^2 n2);
%  HTD, by contrast, already compresses BOTH modes via its own leaf
%  factors. Here nCols = nRows by construction (m=l, T=L below), so this
%  gap grows with H just as fast as the mode-1 gap does -- unlike the
%  4th-order T-SVD timing experiment where n3=n4=4 were fixed constants.
%  Validated on small asymmetric sizes before this run: _729 vs _730 T-ERA
%  end-to-end outputs (A_red/B_red/C_red) agree to ~1e-14 relative error.
%  The 'htd' branch is UNCHANGED (tera_reduce's htd case still calls
%  tsvd_htd_dim3).
%
%  All other logic carried forward unchanged from Test_TERA_3_729_10.m:
%    (1) Sparse case Hankel density 1e-4 (nnz floor guards against an
%        all-zero tensor at small N).
%    (2) opts.tt_rmax / opts.ht_r1 / opts.ht_r2 / opts.ht_r12 = 1e5
%        (tolerance-driven rank, not a fixed cap -- a fixed cap=30
%        previously caused a structurally WRONG Sparse-case reduced model).
%    (3) The runtime memory guard before the Sparse case's 'ttd'/'htd'
%        calls (estimated dominant intermediate vs preset budget).
%
%  All other parameters (n=100, s=9, H_list=[10000], k_frac=0.8,
%  ntrials=10) are unchanged from Test_TERA_3_729_10.m.
%  ---------------------------------------------------------------

clear; clc;

%% Problem sizes (match paper defaults)
n_array = 100;
numN = numel(n_array);

s = 9;                   % tubal length

%% Hankel block sizes to sweep
H_list = [10000]
numH   = numel(H_list);

k_frac = 0.8;

w = linspace(0, pi, 600);

case_names = {'Sparse', 'LowTT', 'LowHT'};
numCases = numel(case_names);
ntrials = 10;

% Time_H(case, H_index, method, trial)
Time_H = zeros(numCases, numH, 3, ntrials);

%% Options for T_ERA_fast
opts = struct();
opts.svd_tol = 1e-12;
opts.tt_eps  = 1e-10;
opts.tt_rmax = 1e5;    % was 30 -- effectively uncapped, tolerance governs
opts.ht_r1   = 1e5;
opts.ht_r2   = 1e5;
opts.ht_r3   = min(9,s);
opts.ht_r12  = 1e5;

% Memory guard: skip (NaN) a method if its estimated dominant intermediate
% array would exceed this budget (GB). Set well below the SLURM job's
% --mem allocation to leave headroom for MATLAB overhead.
mem_budget_gb = 100;

for trial = 1:ntrials
    for nn = 1:numel(n_array)
        n = n_array(nn);

        for hIdx = 1:numH
            H = H_list(hIdx);

            [l, L] = pick_lL(H);
            m = l;
            T = L;

            nRows = l*(L+1);
            nCols = m*(T+1);

            k = min(round(k_frac * min(nRows,nCols)), min(nRows,nCols)-1);
            r = min(nRows,nCols) - k;  %#ok<NASGU>

            [A, B, C] = gen_ABC_base(n, m, l, s);
            A = stabilize_tensor_A(A, 0.95);

            for c = 1:numCases
                fprintf('\n=== Case: %s | H=%d | (l=%d,L=%d,m=%d,T=%d) | k=%d (r=%d) | trial %d/%d ===\n', ...
                    case_names{c}, H, l, L, m, T, k, min(nRows,nCols)-k, trial, ntrials);

                Hpack = gen_H_case_prebuilt(c, nRows, nCols, s, opts);

                tStart = tic;
                tera_reduce(A, B, C, k, T, L, 't', opts, Hpack.full);
                Time_H(c,hIdx,1,trial) = toc(tStart);

                % Memory guard (only meaningfully large for the Sparse
                % case; LowTT/LowHT ranks stay tiny so this never binds
                % there, matching the same reasoning used for T-product
                % and T-SVD in this investigation).
                r1_tt = Hpack.ttd.tt_cores.G1;
                r1_tt = size(r1_tt, 3);
                mem_gb_ttd = s * r1_tt^2 * 16 / 1e9;

                r1_ht = size(Hpack.htd.ht_factors.U1, 2);
                mem_gb_htd = s * r1_ht^2 * 16 / 1e9;

                if mem_gb_ttd > mem_budget_gb
                    fprintf('  TTD SKIPPED: estimated memory %.1f GB > budget %.1f GB (TT rank r1=%d)\n', ...
                        mem_gb_ttd, mem_budget_gb, r1_tt);
                    Time_H(c,hIdx,2,trial) = NaN;
                else
                    tStart = tic;
                    tera_reduce(A, B, C, k, T, L, 'ttd', opts, Hpack.ttd);
                    Time_H(c,hIdx,2,trial) = toc(tStart);
                end

                if mem_gb_htd > mem_budget_gb
                    fprintf('  HTD SKIPPED: estimated memory %.1f GB > budget %.1f GB (HT rank r1=%d)\n', ...
                        mem_gb_htd, mem_budget_gb, r1_ht);
                    Time_H(c,hIdx,3,trial) = NaN;
                else
                    tStart = tic;
                    tera_reduce(A, B, C, k, T, L, 'htd', opts, Hpack.htd);
                    Time_H(c,hIdx,3,trial) = toc(tStart);
                end

                fprintf('Time | TERA %.3fs | TTD %.3fs | HTD %.3fs\n', ...
                    Time_H(c,hIdx,1,trial), Time_H(c,hIdx,2,trial), Time_H(c,hIdx,3,trial));
            end

        end
    end
end


Time_ERA_avg = mean(Time_H, 4, 'omitnan');
Time_ERA_std = std(Time_H, 0, 4, 'omitnan');

if ~exist('results','dir'); mkdir('results'); end

tag = sprintf('TERA_scaling_H%d-%d_kfrac%.2f_trials%d_803', ...
    H_list(1), H_list(end), k_frac, ntrials);

save(fullfile('results', [tag '.mat']), ...
    'H_list','k_frac','case_names','n_array','s','ntrials', ...
    'Time_ERA_avg','Time_ERA_std');


%% ------------------ plots ------------------
H_target = 10000;
hIdx = find(H_list == H_target, 1);
if isempty(hIdx)
    [~,hIdx] = max(H_list);
    H_target = H_list(hIdx);
end

Y = squeeze(Time_ERA_avg(:, hIdx, :));
E = squeeze(Time_ERA_std(:, hIdx, :));

c_tera = [0.0, 0.45, 0.70];
c_ttd  = [0.90, 0.60, 0.00];
c_htd  = [0.0, 0.60, 0.50];

method_labels = {'TERA','TTD','HTD'};
C = [c_tera; c_ttd; c_htd];

case_titles = {'Random Sparse','Low TT-Rank','Low HT-Rank'};

fig = figure('Position',[80,120,1320,420]);
tl  = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
ylabel(tl,'Time (s)','FontSize',16);

FS_tick  = 14;
FS_title = 18;

axs = gobjects(1,3);

for c = 1:3
    ax = nexttile(tl,c);
    axs(c) = ax;
    hold(ax,'on');

    b = bar(ax, 1:3, Y(c,:), 0.75);
    b.FaceColor = 'flat';
    b.CData = C;

    errorbar(ax, 1:3, Y(c,:), E(c,:), 'k', 'LineStyle','none', 'LineWidth',1);

    ax.FontSize   = FS_tick;
    ax.LineWidth  = 1.1;
    ax.TickDir    = 'in';
    ax.TickLength = [0.01 0.01];
    ax.XMinorTick = 'off';
    ax.YMinorTick = 'off';
    box(ax,'on');
    grid(ax,'off');

    xticks(ax,1:3);
    xticklabels(ax,method_labels);

    title(ax, case_titles{c}, 'FontSize', FS_title);

    hold(ax,'off');
end

axs(2).YTickLabel = [];
axs(3).YTickLabel = [];

xlabel(tl, sprintf('H = %d', H_target), 'FontSize', 16.5);

if ~exist('results','dir'); mkdir('results'); end
savefig(fig, fullfile('results', [tag '.fig']));


%% ---------------- local helpers ----------------
function [A,B,C] = gen_ABC_base(n, m, l, s)
density = 1e-3;
A = sprandn(n*n*s, 1, density);
A = reshape(full(A), [n,n,s]) / 200;
B = randn(n,m,s);
C = randn(l,n,s);
end

function Hpack = gen_H_case_prebuilt(case_id, nRows, nCols, s, opts)

Hpack = struct();
TT_H_gen = [];

U1_gen = []; U2_gen = []; U3_gen = [];
B12_gen = []; Broot_gen = [];

switch case_id
    case 1   % Sparse random -- _803: FIXED nnz (~30), independent of N
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
        r1 = 2;
        r2 = 2;
        r3 = 2;
        r12 = 2;

        U1_gen = orth(randn(nRows, r1));
        U2_gen = orth(randn(nCols, r2));
        U3_gen = orth(randn(s, r3));

        B12_gen = randn(r1, r2, r12);
        Broot_gen = randn(r12, r3);

        H = htd3_to_full(U1_gen, U2_gen, U3_gen, B12_gen, Broot_gen);

    otherwise
        error('Unknown case_id');
end

% -------- full representation --------
Hpack.full = struct('type','full', 'H', H, 'Hh', H);

% -------- TT representation (precompute, not timed later) --------
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

Hpack.ttd = struct( ...
    'type','ttd', ...
    'tt_cores', struct('G1',G1,'G2',G2,'G3',G3), ...
    'Hh', H);

% -------- HT representation (precompute, not timed later) --------
if case_id == 3 && ~isempty(U1_gen)
    U1 = U1_gen;
    U2 = U2_gen;
    U3 = U3_gen;
    B12 = B12_gen;
    Broot = Broot_gen;
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

Hpack.htd = struct( ...
    'type','htd', ...
    'ht_factors', struct('U1',U1,'U2',U2,'U3',U3,'B12',B12,'Broot',Broot), ...
    'Hh', H);

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

function ninf = hinf_approx(sys, w)
resp = freqresp(sys, w);
ninf = 0;
for i = 1:length(w)
    s = svd(resp(:,:,i));
    ninf = max(ninf, s(1));
end
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

function [G1,G2,G3] = extract_tt_cores_dim3(TT, n1, n2, n3)
r = TT.r;
r1 = r(2); r2 = r(3);
core = TT.core;
ps = TT.ps;
G1 = reshape(core(ps(1):ps(2)-1), [1, n1, r1]);
G2 = reshape(core(ps(2):ps(3)-1), [r1, n2, r2]);
G3 = reshape(core(ps(3):ps(4)-1), [r2, n3, 1]);
end

function [U1,U2,U3,B12,Broot] = numeric_to_ht_factors_dim3(H, r1, r2, r3, r12, tol)
% Tolerance-adaptive HT-like factorization for 3D tensor H (n1 x n2 x n3),
% tree {1,2}-{3}. r1,r2,r3,r12 here are CEILINGS; the actual rank kept at
% each split is chosen via a relative-singular-value threshold (tol),
% mirroring the TT side's tolerance-driven round(..., opts.tt_eps, opts.tt_rmax).
if nargin < 6 || isempty(tol)
    tol = 1e-10;
end
[n1,n2,n3_full] = size(H);
r1  = min(r1, n1);
r2  = min(r2, n2);
r3  = min(r3, n3_full);

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
