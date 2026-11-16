# SemiSL_Consistency_FlexMatch

**FlexMatch semi-supervised classification on Crop36 (FashionLetters36) - PAI-SG SemiSL**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the semi-supervised
comparison in [SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36).

## Method

FixMatch with class-wise thresholds 0.95 x M(beta_c), M(x) = x / (2 - x), from the per-class count of confident unlabeled predictions, with threshold warm-up (Zhang et al. 2021).

**Common setup** (identical for the four classification repositories; all methods side by side:
[SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36)). Crop36 (FashionLetters36 object crops, 36 classes): labeled = the crops of the labeled images of the
nested image splits (seed 777; 1 % = 41 images / 258 crops, 5 % = 205 / 1,340, 10 % = 410 / 2,653; lists in
`splits/`), unlabeled = all 26,716 train crops with labels never read, test = 6,699 crops. ResNet-18 from scratch or
from the VICReg 200-epoch SSL backbone ([SSL_Redundancy_VICReg](https://github.com/Statera-Workshop/SSL_Redundancy_VICReg)). Microsoft USB
classic_cv defaults: SGD lr 0.03 nesterov, wd 5e-4 (not on BN / bias), labeled batch 64, cosine 7/16, EMA 0.999
model evaluated, 10k steps, last step reported. GPU augmentation; no horizontal flip. Supervised-only reference
(same recipe): 0.502 / 0.833 / 0.873 from scratch,
0.780 / 0.892 / 0.908 from VICReg (1 / 5 / 10 %); 100 % labels 0.926 / 0.932.

## Results (Crop36 EMA test top-1)

| Init | 1 % | 5 % | 10 % |
| --- | --- | --- | --- |
| from scratch | 0.868 | 0.912 | 0.913 |
| VICReg 200 | 0.843 | 0.913 | 0.918 |
| supervised only, from scratch | 0.502 | 0.833 | 0.873 |
| supervised only, VICReg 200 | 0.780 | 0.892 | 0.908 |

Best classification method: 0.868 with 1 % labels and 0.918 with 10 % (full supervision 0.926 / 0.932). The class-wise thresholds limit the loss FixMatch shows with the SSL initialisation at 1 %.
Macro F1 and the evaluation curve of every run: `results/`.

## Run

```bash
python code/make_label_splits.py           # nested labeled splits (needs FL36 at ~/SSL_Wave1/datasets)
python code/semisl_cls.py --method flexmatch --ratio p10 --init scratch --tag cls_p10_flexmatch_scratch
python code/semisl_cls.py --method flexmatch --ratio p10 --init <ssl_vicreg_e200_backbone.pth> --tag cls_p10_flexmatch_vicreg200
```

Crop36 is built by `prep_ssl_data.py` of [SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36). `code/semisl_cls.py` holds all five methods; this repository's
runs use `--method flexmatch` (the supervised-only runs are included as the baseline).
Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md) (USB MIT notice in `licenses/`).
