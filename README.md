# TTD/HTD-based Transform-Domain Tensor Algebra — Numerical Examples

Numerical example drivers and supporting algorithm implementations for the
TTD- and HTD-based T-product / T-SVD framework (block diagonalization,
T-product, T-SVD, and multilinear model order reduction via T-ERA).

## Layout

```
drivers/       Top-level experiment scripts (produce the paper's timing /
                accuracy tables and figures)
algorithms/    Core TTD-/HTD-/definition-based routines called by the drivers
```

### drivers/

| File | Description |
|---|---|
| `tsvd_dim4_timing.m` | 4th-order T-SVD: definition-based vs. TTD-based vs. HTD-based, timing comparison across problem size, for Sparse / Low-TT-rank / Low-HT-rank test tensors. |
| `tprod_timing.m` | 3rd-order T-product: definition-based vs. TTD-based vs. HTD-based, timing comparison. |
| `tera_timing.m` | T-ERA (Eigensystem Realization Algorithm on transform-domain Hankel tensors): timing comparison of the three T-SVD backends inside the reduced-order model construction. |
| `tera_relerr_hinf.m` | T-ERA T-SVD-backend accuracy comparison: relative H-infinity error between each backend's (definition/TTD/HTD) T-SVD reconstruction and the exact input Hankel tensor, evaluated as a per-frequency spectral-norm (sup-over-frequency operator-norm) distance. |

### algorithms/

| File | Implements |
|---|---|
| `tsvd_ttd_dim3.m` / `tsvd_ttd_dim4.m` | TTD-based T-SVD (core-form, 3rd/4th order). |
| `tsvd_ttd_dim4_ref.m` | Earlier TTD-based 4th-order T-SVD variant, kept only as an internal self-consistency reference check inside `tsvd_dim4_timing.m` (not part of the reported results). |
| `tsvd_htd_dim3.m` / `tsvd_htd_dim4.m` | HTD/TD-based T-SVD (core-form, 3rd/4th order). |
| `tprod_definition.m` | Definition-based (block-circulant) T-product baseline. |
| `tprod_ttd.m` | TTD-based T-product (core-form). |
| `tprod_htd.m` | HTD-based T-product (core-form). |
| `tera_reduce.m` | T-ERA reduced-order model construction with a swappable T-SVD backend (`'t'`/`'ttd'`/`'htd'`). |

## Dependencies

These scripts require the following third-party toolboxes on the MATLAB path
(not included in this repo):
- **TT-Toolbox** (`tt_tensor`, `tt_rand`, etc.)
- **htucker** (`htenrandn`, `orthog`, etc.)
- A T-product toolbox providing `tsvd`, `tprod`, `tran`, `bcirc` (definition-based
  baseline operators used for validation/comparison).

## Note on `tera_relerr_hinf.m`'s methodology

An earlier version of this comparison built each backend's ERA reduced
model `(A,B,C)` via `tera_reduce.m` and compared them through a
resolvent/H-infinity evaluation, with `B_red`/`C_red` extracted from the
Hankel tensor's `(1:l, 1:m, :)` corner block. That corner is a meaningful
"first Markov parameter" only for a Hankel tensor derived from a real
`(A,B,C)` system; the Sparse/Low-TT-rank/Low-HT-rank test tensors used
here are instead synthetic tensors fed directly as a Hankel-matrix
surrogate, and for the Sparse case specifically (30 nonzeros scattered
over ~9e8 entries at H=10000) that corner block is essentially always
empty, making the old comparison uninformative for that case. The current
version instead compares each backend's T-SVD reconstruction directly
against the exact input tensor (see the file's header), which uses the
full tensor and needs no state-space realization step at all.
