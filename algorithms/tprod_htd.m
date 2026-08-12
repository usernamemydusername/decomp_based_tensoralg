function [U1C,U2C,U3C,B12C,BrootC] = tprod_htd(U1A,U2A,U3A,B12A,BrootA, U1B,U2B,U3B,B12B,BrootB)
%TPROD_HTD  HTD-based T-product of two third-order tensors, computed
% directly from their HTD factors, without forming either input tensor
% or the block-circulant matrix densely.
%
% Algorithm: the transform-mode leaves U3A,U3B are FFT'd; per frequency,
% the small mode-{1,2} cores are combined (via the shared mode-2/mode-1
% contraction M = U2A' * U1B) and multiplied together, then the result is
% inverse-FFT'd back to a time-domain small core B12C. The output HTD
% keeps A's mode-1 leaf and B's mode-2 leaf unchanged (U1C=U1A, U2C=U2B)
% and uses identity leaves/root on the transform mode, so B12C alone
% carries all of the product's information.
%
% Inputs:
%   U1A,U2A,U3A,B12A,BrootA   HTD factors of A (tree {1,2}-{3}): leaves
%                             [n x r1A],[n2A x r2A],[r x r3A], transfer
%                             core B12A [r1A x r2A x r12A], root BrootA
%                             [r12A x r3A]
%   U1B,U2B,U3B,B12B,BrootB   HTD factors of B, same convention, with
%                             size(U1B,1) == size(U2A,1) (shared inner
%                             T-product dimension) and size(U3B,1) ==
%                             size(U3A,1) (matching transform-mode length)
%
% Outputs:
%   U1C,U2C,U3C,B12C,BrootC   HTD factors of C = A*B: U1C=U1A, U2C=U2B,
%                             U3C=BrootC=identity (size r x r, r = the
%                             shared transform-mode length), and B12C
%                             [r1A x r2B x r] the only newly-computed
%                             factor
%
% Requires:
%   - fft, ifft (base MATLAB only -- no external toolbox)
%
% Notes:
%   - Automatically detects and preserves realness: if all inputs are
%     real, only ceil((r+1)/2) frequencies are computed explicitly and
%     the rest filled in by conjugate symmetry, and B12C is returned real.

% ---- sizes / checks ----
[n,  r1A] = size(U1A);
[n2A,r2A] = size(U2A);
[r,  r3A] = size(U3A);

[nB, r1B] = size(U1B);
[m,  r2B] = size(U2B);
[rB, r3B] = size(U3B);

assert(nB==n, 'Mode-1 sizes must match.');
assert(rB==r, 'Tube length r must match.');
assert(n2A==n, 'For A: size(U2A,1) should equal n.');
assert(size(B12A,1)==r1A && size(B12A,2)==r2A, 'B12A size mismatch.');
assert(size(B12B,1)==r1B && size(B12B,2)==r2B, 'B12B size mismatch.');

r12A = size(B12A,3);  r12B = size(B12B,3);
assert(all(size(BrootA)==[r12A, r3A]), 'BrootA must be (r12A x r3A).');
assert(all(size(BrootB)==[r12B, r3B]), 'BrootB must be (r12B x r3B).');

is_all_real = isreal(U1A)&&isreal(U2A)&&isreal(U3A)&&isreal(B12A)&&isreal(BrootA) && ...
              isreal(U1B)&&isreal(U2B)&&isreal(U3B)&&isreal(B12B)&&isreal(BrootB);

% ---- FFT weights ----
U3A_tilde = fft(U3A, [], 1);
U3B_tilde = fft(U3B, [], 1);

coeffA = BrootA * (U3A_tilde.'); % (r12A x r)
coeffB = BrootB * (U3B_tilde.'); % (r12B x r)

B12A_mat = reshape(B12A, r1A*r2A, r12A);
B12B_mat = reshape(B12B, r1B*r2B, r12B);

M = (U2A).' * U1B; % (r2A x r1B)

H_fft = zeros(r1A, r2B, r);

if is_all_real
    halfn3 = ceil((r + 1)/2);
    kmax = halfn3;
else
    kmax = r;
end

for k = 1:kmax
    GAk = reshape(B12A_mat * coeffA(:,k), r1A, r2A);
    GBk = reshape(B12B_mat * coeffB(:,k), r1B, r2B);
    H_fft(:,:,k) = (GAk * M) * GBk;
end

if is_all_real
    for k = halfn3+1:r
        H_fft(:,:,k) = conj(H_fft(:,:,r+2-k));
    end
end

B12C = ifft(H_fft, [], 3);
if is_all_real
    B12C = real(B12C);
end

% ---- pack as HTD cores ----
U1C = U1A;
U2C = U2B;
U3C = eye(r);        % (r x r)
BrootC = eye(r);     % (r x r)
end
