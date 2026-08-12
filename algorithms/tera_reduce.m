function [A_red, B_red, C_red, prof] = tera_reduce(A, B, C, k, T, L, method, opts, H_input)
%TERA_REDUCE  Eigensystem Realization Algorithm (ERA) reduced-order model
% construction for a T-product (transform-domain) LTI system, with a
% swappable T-SVD backend for factoring the underlying Hankel tensor.
%
% Given either a system (A,B,C) or a precomputed Hankel-tensor
% representation, forms/reads the Hankel tensor H of Markov parameters
% (and its one-step shift Hh), computes a rank-r T-SVD of H via one of
% three backends (definition-based full SVD, TTD-based, or HTD-based --
% all three compute the SAME T-SVD of the SAME H, only the computational
% path differs), and assembles the reduced-order model via the standard
% ERA realization formulas
%   A_red = S^{-1/2} U1' Hh V1 S^{-1/2}
%   B_red = S^{1/2} V1'(:, 1:m)
%   C_red = U1(1:l,:) S^{1/2}
%
% Inputs:
%   A,B,C     system tensors, sizes [n x n x s], [n x m x s], [l x n x s]
%             (only used to build H/Hh when H_input is empty)
%   k         truncation parameter; target reduced order is
%             r = min(nRows,nCols) - k, with nRows=l*(L+1), nCols=m*(T+1)
%   T, L      number of column/row Hankel blocks
%   method    't' (definition-based, full T-SVD via tsvd(H)), 'ttd'
%             (tsvd_ttd_dim3), or 'htd' (tsvd_htd_dim3) -- selects the
%             T-SVD backend
%   opts      struct; relevant field: svd_tol (rank-detection tolerance
%             for the ttd/htd backends)
%   H_input   (optional) precomputed Hankel representation, struct with
%             field .type = 'full' | 'ttd' | 'htd' and the corresponding
%             data (.H/.Hh for 'full'; .tt_cores.G1/G2/G3 + .Hh for 'ttd';
%             .ht_factors.U1/U2/U3/B12/Broot + .Hh for 'htd'). If empty,
%             H and Hh are built directly from A,B,C's Markov parameters.
%
% Outputs:
%   A_red, B_red, C_red   reduced system tensors, sizes [r x r x s],
%                         [r x m x s], [l x r x s], where r is the ACTUAL
%                         achieved order: r = min(k's implied order,
%                         size(U,2)) for 't', or r = min(k's implied
%                         order, U_rep.max_svd_rank) for 'ttd'/'htd' --
%                         i.e. 'ttd'/'htd' silently cap to the Hankel
%                         tensor's own numerical rank, while 't' does
%                         not, so reduced models from different methods
%                         can end up with DIFFERENT state dimension r for
%                         the same requested k.
%   prof                  struct of stage timings (buildH, tsvdH,
%                         buildAred, buildBred, buildCred, total)
%
% Requires:
%   - tproduct toolbox: tprod, tran, tsvd
%   - tsvd_ttd_dim3.m   (used when method='ttd')
%   - tsvd_htd_dim3.m   (used when method='htd')
%
% Notes:
%   - Because of the order-capping asymmetry above, DO NOT compare
%     A_red/B_red/C_red across methods directly (elementwise) unless you
%     have first verified all three achieved the same r -- otherwise the
%     comparison will error on a dimension mismatch or, worse, silently
%     compare two systems that are valid realizations in different bases
%     (T-SVD is only unique up to sign per singular vector). Compare via
%     a basis-independent metric instead (e.g. resolvent/H-infinity
%     response), or compare the underlying T-SVD's reconstruction
%     accuracy directly (bypassing this function entirely, as
%     tera_relerr_hinf.m does).

if nargin < 7 || isempty(method), method = 't'; end
if nargin < 8 || isempty(opts), opts = struct(); end
if nargin < 9, H_input = []; end


svd_tol = getfield_default(opts, 'svd_tol', 1e-12);

[n, n2, s] = size(A);
if n ~= n2, error('A must be square on modes 1-2.'); end


% ---------- Build / read Hankel tensor representation ----------
prof = struct();
t_total = tic;

l_dim = size(C,1);
m_dim = size(B,2);

nRows = l_dim * (L+1);
nCols = m_dim * (T+1);

r_keep_rows = nRows - k;
r_keep_cols = nCols - k;
r = min(r_keep_rows, r_keep_cols);

if r <= 0
    error('Invalid k=%d: nRows=%d, nCols=%d => r=%d. Choose k < min(nRows,nCols).', k, nRows, nCols, r);
end

H = [];
Hh = [];

tt_cores = [];
ht_factors = [];

if ~isempty(H_input)
    if ~isstruct(H_input) || ~isfield(H_input, 'type')
        error('H_input must be a struct with field .type');
    end

    switch lower(H_input.type)
        case 'full'
            H = H_input.H;
            if ~isequal(size(H), [nRows, nCols, s])
                error('H_input.H must have size [%d,%d,%d].', nRows, nCols, s);
            end
            if isfield(H_input, 'Hh') && ~isempty(H_input.Hh)
                Hh = H_input.Hh;
            else
                Hh = H;
            end

        case 'ttd'
            tt_cores = H_input.tt_cores;
            if ~isfield(tt_cores, 'G1') || ~isfield(tt_cores, 'G2') || ~isfield(tt_cores, 'G3')
                error('TT input must contain tt_cores.G1, G2, G3');
            end
            if isfield(H_input, 'Hh') && ~isempty(H_input.Hh)
                Hh = H_input.Hh;
            else
                error('For method=ttd, please provide H_input.Hh (dense shifted Hankel surrogate).');
            end

        case 'htd'
            ht_factors = H_input.ht_factors;
            req = {'U1','U2','U3','B12','Broot'};
            for ii = 1:numel(req)
                if ~isfield(ht_factors, req{ii})
                    error('HT input missing ht_factors.%s', req{ii});
                end
            end
            if isfield(H_input, 'Hh') && ~isempty(H_input.Hh)
                Hh = H_input.Hh;
            else
                error('For method=htd, please provide H_input.Hh (dense shifted Hankel surrogate).');
            end

        otherwise
            error('Unknown H_input.type = %s', H_input.type);
    end

else
    % ---------- original build-from-(A,B,C) branch ----------
    A_hat = fft(A, [], 3);
    B_hat = fft(B, [], 3);
    C_hat = fft(C, [], 3);

    max_idx = L + T + 2;
    M_hat = zeros(l_dim, m_dim, s, max_idx);

    for j = 1:s
        Aj = A_hat(:,:,j);
        Bj = B_hat(:,:,j);
        Cj = C_hat(:,:,j);

        X = Bj;
        M_hat(:,:,j,1) = Cj * X;

        for t = 2:max_idx
            X = Aj * X;
            M_hat(:,:,j,t) = Cj * X;
        end
    end

    H_hat_full  = zeros(nRows, nCols, s);
    Hh_hat_full = zeros(nRows, nCols, s);

    for q = 1:s
        Hq  = zeros(nRows, nCols);
        Hhq = zeros(nRows, nCols);
        for ii = 0:L
            for jj = 0:T
                Mt  = M_hat(:,:,q, ii+jj+1);
                Mt1 = M_hat(:,:,q, ii+jj+2);
                r_idx = (ii*l_dim+1):((ii+1)*l_dim);
                c_idx = (jj*m_dim+1):((jj+1)*m_dim);
                Hq(r_idx, c_idx)  = Mt;
                Hhq(r_idx, c_idx) = Mt1;
            end
        end
        H_hat_full(:,:,q)  = Hq;
        Hh_hat_full(:,:,q) = Hhq;
    end

    H  = ifft(H_hat_full, [], 3);
    Hh = ifft(Hh_hat_full, [], 3);
end


prof.buildH = toc(t_total);


% ---------- t-SVD of H (swappable backend), truncated to rank r ----------
% For 'ttd'/'htd', truncation happens on the CORE-FORM representation
% before reconstruction, so the reconstructed U1/S1/V1 are already at the
% needed rank r (never materializing the full max-rank tensor).
t_svd = tic;
switch lower(method)
    case {'t','def','baseline'}
        if isempty(H)
            error('Full tensor H is required for method=''t''.');
        end
        [U,S,V] = tsvd(H);
        r = min(r, size(U,2));
        U1 = U(:, 1:r, :);
        S1 = S(1:r, 1:r, :);
        V1 = V(:, 1:r, :);

    case 'ttd'
        if isempty(tt_cores)
            error('Precomputed TT cores required for method=''ttd''.');
        end
        G1 = tt_cores.G1;
        G2 = tt_cores.G2;
        G3 = tt_cores.G3;
        [U_rep, S_rep, V_rep] = tsvd_ttd_dim3(G1, G2, G3, nRows, nCols, s, svd_tol);
        r = min(r, U_rep.max_svd_rank);
        [U1, S1, V1] = reconstruct_truncated_ttd_dim3_730(U_rep, S_rep, V_rep, r);

    case 'htd'
        if isempty(ht_factors)
            error('Precomputed HT factors required for method=''htd''.');
        end
        [U_rep, S_rep, V_rep] = tsvd_htd_dim3( ...
            ht_factors.U1, ht_factors.U2, ht_factors.U3, ...
            ht_factors.B12, ht_factors.Broot, svd_tol);
        r = min(r, U_rep.max_svd_rank);
        [U1, S1, V1] = reconstruct_truncated_htd_dim3_729(U_rep, S_rep, V_rep, r);

    otherwise
        error('Unknown method: %s', method);
end
prof.tsvdH = toc(t_svd);


% ---------- Build S^{+/-1/2} slice-wise in Fourier domain ----------
S1_hat = fft(S1, [], 3);
Sinvhalf_hat = zeros(r,r,s);
Shalf_hat    = zeros(r,r,s);

for j = 1:s
    Sj = S1_hat(:,:,j);
    Shalf_hat(:,:,j)    = Sj^(1/2);
    Sinvhalf_hat(:,:,j) = Sj^(-1/2);
end

Shalf    = ifft(Shalf_hat, [], 3);
Sinvhalf = ifft(Sinvhalf_hat, [], 3);

% ---------- Reduced model ----------
tA = tic;
A_red = tprod( tprod( tprod(Sinvhalf, tran(U1)), Hh), tprod(V1, Sinvhalf) );
prof.buildAred = toc(tA);

tB = tic;
V1T   = tran(V1);
Bfull = tprod(Shalf, V1T);
B_red = Bfull(:, 1:m_dim, :);
prof.buildBred = toc(tB);

tC = tic;
Cfull = tprod(U1, Shalf);
C_red = Cfull(1:l_dim, :, :);
prof.buildCred = toc(tC);

prof.total = toc(t_total);

A_red = real(A_red); B_red = real(B_red); C_red = real(C_red);
end

% ----------------- helpers -----------------
function v = getfield_default(s, name, default)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end

function [U1, S1, V1] = reconstruct_truncated_ttd_dim3_730(U_rep, S_rep, V_rep, r)
%RECONSTRUCT_TRUNCATED_TTD_DIM3_730  Truncate the TT cores of U,S,V to rank
% r, THEN reconstruct dense tensors (mode-3 inverse DFT + mode-1/mode-2
% leaf products). Truncating before reconstruction is what avoids ever
% forming the full max-svd-rank tensor. Differs from
% reconstruct_truncated_ttd_dim3_729 only in that V_rep.cores{1} (R1) is
% now a genuine mode-2 basis (n2 x q) instead of an implicit n2 x n2
% identity, so V1 needs the same leaf-product treatment U1 already got.
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

Qmat = reshape(P1, n1, size(P1,3));   % QR factor Q1, [n1 x r1]
Q2mat = reshape(R1, n2, size(R1,3));  % QR factor Q2, [n2 x q]

Uphys = ifft(P2t, [], 3);             % [r1, r, n3]
Sphys = ifft(Q2t, [], 3);             % [r,  r, n3]
Vphys = ifft(R2t, [], 3);             % [q,  r, n3]

r1 = size(Qmat,2);
q = size(Q2mat,2);
U1 = real(reshape(Qmat * reshape(Uphys, r1, r*n3), n1, r, n3));
S1 = real(Sphys);
V1 = real(reshape(Q2mat * reshape(Vphys, q, r*n3), n2, r, n3));
end

function [U1, S1, V1] = reconstruct_truncated_htd_dim3_729(U_rep, S_rep, V_rep, r)
%RECONSTRUCT_TRUNCATED_HTD_DIM3_729  Truncate the Tucker-style core of
% U,S,V to rank r, THEN reconstruct dense tensors (mode-3 inverse DFT +
% mode-1 leaf product, Tucker convention per prop:htd_tsvd).
U1leaf = U_rep.leaf;    % [n1 x r1_ht]
V1leaf = V_rep.leaf;    % [n2 x r2_ht]

Pcore = U_rep.core(:, 1:r, :);   % [r1_ht, r, n3]
Qcore = S_rep.core(1:r, 1:r, :); % [r, r, n3]
Rcore = V_rep.core(:, 1:r, :);   % [r2_ht, r, n3]

n1 = size(U1leaf,1); r1_ht = size(U1leaf,2);
n2 = size(V1leaf,1); r2_ht = size(V1leaf,2);
n3 = size(Pcore,3);

Uphys = ifft(Pcore, [], 3);   % [r1_ht, r, n3]
Sphys = ifft(Qcore, [], 3);   % [r, r, n3]
Vphys = ifft(Rcore, [], 3);   % [r2_ht, r, n3]

U1 = real(reshape(U1leaf * reshape(Uphys, r1_ht, r*n3), n1, r, n3));
S1 = real(Sphys);
V1 = real(reshape(V1leaf * reshape(Vphys, r2_ht, r*n3), n2, r, n3));
end
