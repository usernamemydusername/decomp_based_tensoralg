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
| `tera_relerr_hinf.m` | T-ERA accuracy comparison: relative H-infinity error between TTD-/HTD-/definition-based reduced models and a common higher-order reference, evaluated via per-frequency resolvent response. |

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

## Known issues

- `tera_relerr_hinf.m`, **Sparse case**: the current sparse-tensor generator
  (`target_nnz = 30`, independent of tensor size) scatters nonzeros uniformly
  over the *entire* Hankel tensor. At the problem size used here (H=10000,
  Hankel block size l=m=100), the probability that any of the 30 nonzeros
  lands inside the `(1:l, 1:m, :)` corner block used to build the reduced
  model's B/C matrices is ~0.3% per trial — so that corner is effectively
  always zero/noise, and the reported relative H-infinity error for the
  Sparse case is not currently meaningful (tends to saturate near 1.0). This
  does not affect the Low-TT-rank / Low-HT-rank cases. A fix (targeted
  nonzero placement inside the corner block) is planned but not yet applied
  in this version.
