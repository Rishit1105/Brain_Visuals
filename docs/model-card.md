# NeuroSeg model card

**Research and teaching only. Never use model output as a diagnosis.**

Model: MONAI `brats_mri_segmentation`, SegResNet, bundle metadata version **0.5.4**, Apache-2.0, MONAI Consortium. Immutable source revision `370f7f9d062745fbac445e7fe6d6616d35df04ec`. Checkpoint SHA-256 `860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae`.

Training source: BraTS 2018. Official card describes 285 studies partitioned into 200 training, 42 validation, 43 testing. This project performs no training or fine-tuning. Pretrained patient IDs and overlap with future evaluation cases have not been audited.

Inputs: four co-registered, skull-stripped adult brain MRI channels in order T1Gd, T1, T2, FLAIR. Adapter uses RAS+ axis-aligned 1 mm world-space linear resampling, nonzero per-channel z-score normalization. SegResNet architecture uses initial filters 16, encoder blocks [1,2,2,4], decoder blocks [1,1,1], 3 sigmoid outputs TC, WT, ET. Prototype window 128³, overlap 0.5; this differs from the upstream inference window and needs accuracy/latency validation. Weights load strictly; hash mismatch refuses inference.

Outputs: thresholds >0.5 with official region precedence ET→4, other TC→1, other WT→2, background→0. Restore by nearest neighbour to original grid, preserve affine/qform/sform. Code 1 is residual predicted core under this conversion; it is not an independently validated necrosis tissue classification. Region containment follows from the exported codes.

Measurements: count original-grid region voxels × absolute affine determinant / 1,000 mL. Expert comparison reports per-region Dice and sensitivity. Both-empty Dice=1; sensitivity undefined/null for empty reference, excluded from sensitivity means.

Evidence: checkpoint loading and an end-to-end synthetic phantom software smoke test passed. Eleven synthetic imaging tests passed. This project has **no held-out case accuracy results**, no verified pretraining-overlap audit, no T4 latency benchmark and no clinical validation. Upstream historical scores are not presented as this project's results.

Limits: expected adult glioma research distribution only; other disease/anatomy/acquisition protocols may fail. Input checks cannot certify a brain or skull stripping. Intensity-derived surfaces are approximate. Uncertainty is not implemented or calibrated. No diagnosis, treatment advice, grading, prognosis, genetics, surgical or radiotherapy planning.

Privacy: loopback local inference; no scan transmission to third-party APIs. Source uploads removed after successful processing, failed job contents removed, remaining artifacts expire within 24 hours while service runs and are swept on restart. Users must consider browser/OS backups separately.

Source: https://huggingface.co/MONAI/brats_mri_segmentation/tree/370f7f9d062745fbac445e7fe6d6616d35df04ec
