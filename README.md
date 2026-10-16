# NeuroSeg research workspace

A dark imaging workspace inspired by the supplied video and screenshots, with interactive cortical anatomy, three linked MRI planes, region visibility/opacity controls, explicit volume calculations, expert comparison and printable reports.

Research and teaching only. Never use model output as a diagnosis. Public, de-identified research data only.

## Start the complete local platform

On this Windows machine, double-click `start.cmd` or run `.\start.cmd` from this folder. It selects the bundled Python 3.12 runtime, falling back to the Windows Python 3.12 launcher. On other systems use Python 3.11 or 3.12:

```sh
python start.py
```

The first run creates an isolated environment, installs pinned dependencies, downloads the official checkpoint at the revision in `models/model-lock.json`, verifies its SHA-256 and starts http://127.0.0.1:8000. Internet is required for initial installation only. Subsequent inference is entirely local. A CUDA-compatible PyTorch installation and sufficient NVIDIA GPU memory are recommended for actual BraTS-sized studies. The default PyPI installation may be CPU-only on Windows; install the official CUDA build appropriate to your system in `.venv` when using the organiser GPU.

Open the **local** address for uploads and inference. The private hosted website provides the reference demonstration; it has no GPU inference service and never receives MRI uploads. It explicitly refuses live processing. This is intentional separation of hosting and local scan processing.

Import T1, T1Gd, T2 and FLAIR using the four named file slots. An optional expert mask uses BraTS labels 0, 1, 2, 4. Confirm public de-identified data, then validate and run. The local service uses the uploaded images, with no dataset lookup and no stored-mask fallback. Prediction NIfTI is downloadable under Study details; report is printable to PDF. Expert comparison switches both meshes and slice overlays and shows reference volumes and per-case metrics beside estimates.

## Judge command

With the isolated environment active:

```sh
python predict.py --T1 case_t1.nii.gz --T1Gd case_t1ce.nii.gz --T2 case_t2.nii.gz --FLAIR case_flair.nii.gz --output prediction.nii.gz
```

The output preserves original shape, affine, qform/sform and uses uint8 values 0, 1, 2, 4. Masks derive as WT > 0, TC = 1 or 4, ET = 4. The precedence conversion follows the official MONAI bundle configuration; code 1 represents remaining predicted core under this conversion, rather than a separately validated necrosis classification.

## API

- `GET /api/health`: service/checkpoint presence.
- `POST /api/segment`: multipart fields `T1`, `T1Gd`, `T2`, `FLAIR`, optional `expert`; returns a random completed-job ID or clear error.
- `GET /api/jobs/{id}/metadata`, `/meshes`, `/labels`, `/volume/{sequence}`.
- `GET /api/jobs/{id}/expert-labels`, `/expert-meshes` when supplied.
- `GET /api/jobs/{id}/prediction`: original-grid NIfTI.
- `GET /api/jobs/{id}/report`: printable report with provenance.

The API is loopback only, same origin, one active study at a time. It does not enable third-party CORS. API outputs are marked no-store. Upload filenames are never used as storage paths. Source uploads are removed after success; failed jobs are removed; derived files expire after 23 hours with a one-minute sweeper and startup cleanup. No scans or labels are sent to cloud APIs. OS and browser caches are outside this application's deletion guarantee.

## Validation and geometry

The service rejects unreadable/corrupt files, non-3D inputs, NaN/Inf, constant sequences, invalid/singular affine, incompatible shapes/affines and unsupported expert labels. Per-file on-disk and expanded payload limits are 1 GB; there is an additional conservative 64-million-voxel per-sequence cap and 32-million-voxel resampled inference cap. Affines use absolute tolerance 1e-4, zero relative tolerance. Metre/micron inputs are refused; unknown header units trigger an explicit millimetre assumption warning.

Foreground, edge occupancy and physical field-of-view checks warn/refuse unsuitable inputs. These are heuristics, not a validated brain or skull-stripping classifier, and may accept non-brain phantoms or reject unusual genuine images. This remains a limitation against the full judge sanity pack.

Live preprocessing canonicalizes to RAS+, resamples to an axis-aligned 1 mm world grid, normalizes nonzero voxels per modality and runs real MONAI SegResNet sliding-window inference. Original-grid export uses nearest-neighbour restoration. Viewer data use a separate axis-aligned RAS+ 1.5 mm grid, shared by meshes, slices and crosshair. Measurements are calculated on the **original** output grid with `abs(det(affine[:3,:3]))`, independent of viewer sampling or mesh approximation.

The 128³ inference window is smaller than the upstream full inference window (240×240×160), chosen for prototype memory use. Orientation/resampling and window changes require evaluation; we do not claim baseline-equivalent performance.

## Reference demonstration

Reference MRI comes from NiiVue's public `mni152.nii.gz`. The illustrative cortex uses the public ICBM152 left hemisphere mesh with a mirrored right hemisphere and MRI-derived lower anatomy. It is an illustrative composite, not a verified registered or patient-specific cortical reconstruction. The teaching regions are generated synthetically and labelled as such throughout. Live studies never use this reference anatomy or synthetic mask; their surfaces are extracted from their own volumes and predicted masks using marching cubes. Live intensity-derived brain surfaces are approximate and are not cortical parcellations.

Only reference T1 is available in demo mode; unavailable sequences explain this rather than substituting fabricated imagery. Demo reports state that no model was run. All numeric demo volumes are calculated from the synthetic voxel mask and reference affine.

## Verification and remaining acceptance work

Run `python tests.py` for synthetic validation, volume, metric, geometry and export tests. Eleven tests pass, including the organiser's 8,000 voxels × 2 mm³ = **16.0 mL** cube. Real checkpoint loading was verified on CPU. A complete real-model execution on a synthetic four-channel phantom completed in approximately 13.5 seconds on this machine, with original-grid export verified; this is a software smoke test, not a medical accuracy result or T4 benchmark.

Browser checks cover anatomy rendering, slice movement/crosshair coordinates, layer toggles, opacity, volume explanation, import failure messaging and evidence navigation. Full judge orientation/anisotropic predictions, two-voxel marker tolerance, real blind-case performance, repeat-run Dice ≥0.99 and ≤120-second T4 timing remain to be measured.

No organiser dataset, starter notebook, scorer or ten labelled held-out studies were provided. Evaluation scores are therefore **not invented**. `evaluate.py manifest.json --output evaluation.json` evaluates at least ten unique patient IDs, refuses known training overlap, and reports Dice/sensitivity with empty-mask conventions. See `docs/evaluation-manifest.example.json`. Unknown pretrained training overlap must be disclosed separately. No training/fine-tuning was performed. HD95, TTA uncertainty, DICOM, longitudinal comparison and agreement heatmaps are not implemented.

## Credits and model sources

- MONAI Consortium, [brats_mri_segmentation model card](https://huggingface.co/MONAI/brats_mri_segmentation/blob/370f7f9d062745fbac445e7fe6d6616d35df04ec/docs/README.md), bundle 0.5.4, Apache-2.0. See the pinned official documents in `models/` and `docs/model-card.md`.
- [Official inference conversion](https://huggingface.co/MONAI/brats_mri_segmentation/blob/370f7f9d062745fbac445e7fe6d6616d35df04ec/configs/inference.json).
- [NiiVue reference MRI and ICBM152 mesh assets](https://github.com/niivue/niivue/tree/main/packages/niivue/demos/images). Reference assets are used for research demonstration; verify their underlying atlas/data redistribution terms before broader distribution.
- Three.js authors, MIT licence; copyright notice retained in the supplied library.
- BraTS 2021 credit as requested: Baid et al., [DOI 10.7937/jc8x-9874](https://doi.org/10.7937/jc8x-9874). BraTS 2021 images were not downloaded or used here.
- Medical Segmentation Decathlon was referenced by the supplied specification but its data were not used. If added, follow its dataset.json, label mapping and dataset licence.

See `docs/architecture.md`, `docs/model-card.md` and `docs/acceptance.md` for the handoff.
