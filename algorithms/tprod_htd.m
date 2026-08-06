function [U1C,U2C,U3C,B12C,BrootC] = tprod_htd(U1A,U2A,U3A,B12A,BrootA, U1B,U2B,U3B,B12B,BrootB)
%COMPUTE_TPROD_HTD_FAST_CORE  Fast 3rd-order HTD-based T-product (returns HTD cores).
%
% Output HTD uses the same tree {1,2}-{3} and is constructed so that
%   U1C = U1A,  U2C = U2B,  U3C = I_r,
%   B12C(:,:,t) = H_time(:,:,t),  BrootC = I_r,
% where H_time is the time-domain small core obtained via inverse FFT.

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
