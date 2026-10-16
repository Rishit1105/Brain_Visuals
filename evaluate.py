"""Evaluate a patient-held-out manifest. Never tunes thresholds or trains."""
import argparse,json
from pathlib import Path
import numpy as np
from backend.imaging import validate,metrics,digest
from backend.inference import predict
p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--output',default='evaluation.json');args=p.parse_args()
manifest=json.loads(Path(args.manifest).read_text());cases=manifest['cases']
if len(cases)<10:raise SystemExit('At least ten held-out labelled cases are required.')
ids=[c['patient_id'] for c in cases]
if len(ids)!=len(set(ids)):raise SystemExit('Duplicate patient IDs: evaluation must be split by patient.')
if set(ids)&set(manifest.get('training_patient_ids',[])):raise SystemExit('Known patient overlap with training split.')
rows=[]
for c in cases:
    images,truth,warnings=validate(c['modalities'],c['expert']);labels,model=predict(images)
    rows.append({'patient_id':c['patient_id'],'metrics':metrics(labels,truth),'hashes':{k:digest(v) for k,v in c['modalities'].items()}})
summary={k:{m:float(np.mean([r['metrics'][k][m] for r in rows if r['metrics'][k][m] is not None])) if any(r['metrics'][k][m] is not None for r in rows) else None for m in ['dice','sensitivity']} for k in ['WT','TC','ET']}
Path(args.output).write_text(json.dumps({'cases':rows,'mean':summary,'model':model,'pretraining_overlap':manifest.get('pretraining_overlap','Unverified'),'empty_mask_policy':'Dice=1 if both empty; sensitivity null/excluded if reference empty'},indent=2))
print(json.dumps(summary,indent=2))
