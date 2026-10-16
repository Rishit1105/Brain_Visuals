"""Download official public checkpoint; resolve and lock immutable source revision."""
import hashlib,json,urllib.request
from pathlib import Path
root=Path(__file__).parent/'models';root.mkdir(exist_ok=True)
repo='MONAI/brats_mri_segmentation'
existing=json.loads((root/'model-lock.json').read_text()) if (root/'model-lock.json').exists() else None
revision=existing['revision'] if existing else json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo))['sha']
base=f'https://huggingface.co/{repo}/resolve/{revision}/'
for remote,local in [('models/model.pt','model.pt'),('configs/inference.json','upstream-inference.json'),('configs/metadata.json','upstream-metadata.json'),('docs/README.md','UPSTREAM-MODEL-CARD.md')]:
    print('Downloading',local);urllib.request.urlretrieve(base+remote,root/local)
sha=hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()
if existing and sha!=existing['sha256']:raise SystemExit('Checkpoint hash differs from the pinned lock. Refusing installation.')
(root/'model-lock.json').write_text(json.dumps({'repository':repo,'revision':revision,'sha256':sha,'licence':'Apache-2.0'},indent=2))
print('Checkpoint locked:',revision,sha)
