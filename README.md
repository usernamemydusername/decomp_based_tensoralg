# Transform-Based Multilinear Algebra via Tensor Decompositions

MATLAB implementations and numerical experiment drivers accompanying the paper
*Transform-Based Multilinear Algebra via Tensor Decompositions*. This repository
contains the proposed TTD- and HTD-based algorithms and scripts for reproducing
the numerical experiments reported in the paper.

**Authors:** Yidan Mei, Shenghan Mei, Ziqin He, and Can Chen

## Layout

```
drivers/       Top-level experiment scripts that produce the paper's timing
               and accuracy tables and figures
algorithms/    Core TTD-/HTD-/definition-based routines called by the drivers
```

### drivers/

| File | Description |
|---|---|
| `tsvd_dim4_timing.m` | 4th-order T-SVD: definition-based vs. TTD-based vs. HTD-based, timing comparison across problem size, for Sparse / Low-TT-rank / Low-HT-rank test tensors. |
| `tprod_timing.m` | 3rd-order T-product: definition-based vs. TTD-based vs. HTD-based, timing comparison. |
| `tera_timing.m` | T-ERA (Eigensystem Realization Algorithm on transform-domain Hankel tensors): timing comparison of the three T-SVD backends inside the reduced-order model construction. |
| `tera_relerr_hinf_reduced.m` | T-ERA reduced-system accuracy comparison: relative H-infinity error between the TTD-/HTD-based and definition-based reduced-order models (`A_red`, `B_red`, `C_red`), computed exactly via block-circulant unfolding and `hinfnorm`. |

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
- **TT-Toolbox** (`tt_tensor`, `tt_rand`, etc.) — [https://github.com/oseledets/TT-Toolbox](https://github.com/oseledets/TT-Toolbox)
- **htucker** (`htenrandn`, `orthog`, etc.) — [(https://anchp.epfl.ch/index-html/software/htucker/)](https://anchp.epfl.ch/index-html/software/htucker/),
  Kressner & Tobler, "Algorithm 941: htucker — A Matlab toolbox for tensors in
  hierarchical Tucker format," *ACM TOMS* 40(3), 2014, [https://doi.org/10.1145/2538688](https://doi.org/10.1145/2538688)
- **Tensor-tensor product toolbox** (`tsvd`, `tprod`, `tran`, `bcirc` — definition-based
  baseline operators used for validation/comparison) — [[GitHub](https://github.com/canyilu/Tensor-tensor-product-toolbox)](https://github.com/canyilu/Tensor-tensor-product-toolbox)
