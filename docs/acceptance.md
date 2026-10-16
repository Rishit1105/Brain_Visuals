# Acceptance status

Verified: eleven synthetic imaging tests, official checkpoint hash/load on CPU, real-model synthetic phantom execution, uint8 0/1/2/4 original-grid export, 16.0 mL cube, affine mismatch refusal, corrupt/constant/nonfinite/2D refusal, empty-mask metric conventions, reference viewer rendering and linked slice control.

Implemented but needs organiser fixtures: expert annotation comparison, actual four-modality case inference, full-size memory behaviour, hidden-case scorer compatibility, same-anatomy and mirrored-anatomy prediction tests, anisotropic prediction volume consistency, repeat-run ≥0.99 Dice, two-voxel marker precision, non-brain/skull-stripping sanity pack, clean-environment startup and full ≤120-second T4 benchmark.

Not completed: ten patient-held-out labelled evaluations and independent pretraining overlap verification. No data or target GPU were supplied. These are required before claiming the challenge is complete.

Optional extensions not implemented: HD95, TTA/ensemble uncertainty, colour-coded agreement, DICOM, longitudinal comparison and modality ablation.

Failure policy: no prediction or downloadable result for a rejected or failed job. Browser may continue showing the visibly labelled reference demo behind an error dialog; it is never renamed to the failed input or presented as its output.
