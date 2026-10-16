# Architecture

Browser workspace → same-origin local FastAPI API → streamed upload staging → NIfTI validation → pinned SegResNet adapter → original-grid label map → measurements/expert comparison → RAS+ viewer assets/report → expiry cleanup.

The private hosted Site is a static reference viewer. It cannot run PyTorch in a Cloudflare Worker and does not accept scan uploads. The full platform runs locally with Python, with optional CUDA acceleration. This keeps research scan processing on the judge's machine and honours the no-third-party-upload requirement.

`backend/imaging.py` owns geometry and arithmetic. `backend/inference.py` owns checkpoint verification and inference, loading once per service process with serialized access. `backend/main.py` owns lifecycle, uploads, job isolation, API artifacts and reporting. The browser consumes canonical geometry rather than applying undisclosed flips. Three.js operates in a right-handed scene basis x=R, y=S, z=P; this is a rigid rotation of RAS world coordinates, with a shared scene translation. Original world coordinates remain visible.

Input shapes and effective affines must match; no hidden registration occurs. Viewer and inference grids are explicitly separate. Labels are always resampled with nearest neighbour. NIfTI output and quantitative volumes remain on the original grid. Mesh smoothing affects display only.

No database is needed for a single-workstation prototype. Opaque random job directories are available only after completed metadata exists. Source inputs disappear on success; all output artifacts expire after 23 hours with a minute sweep, or startup cleanup after shutdown. One active inference protects memory. No permissive CORS, user filenames in paths or external model APIs. Bind to loopback; this service is not hardened for internet exposure.

Provenance includes immutable checkpoint revision/hash, dependency/model information, preprocessing, input hashes, effective geometry, timestamp, code commit when available and elapsed processing time. Runtime measurement currently starts before validation and ends after viewer construction, excluding multipart upload transfer and the final tiny report write; full upload-to-report T4 benchmark is still required.

The 3D brain surface is a display approximation. Live meshes come from the uploaded image intensities and mask. No reference mesh is inserted into live results. The reference demonstration is an explicitly labelled composite with synthetic teaching regions, and should not be used to validate spatial accuracy.
