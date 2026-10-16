import argparse, json
from pathlib import Path
from backend.imaging import validate
from backend.inference import predict
import nibabel as nib
import numpy as np
p=argparse.ArgumentParser(description='Original-grid BraTS 2021 NIfTI prediction. Research only.')
for name in ['T1','T1Gd','T2','FLAIR']:p.add_argument('--'+name,required=True)
p.add_argument('--output',required=True);args=p.parse_args()
images,_,warnings=validate({k:getattr(args,k) for k in ['T1','T1Gd','T2','FLAIR']})
labels,model=predict(images);ref=images['T1'];h=ref.header.copy();h.set_data_dtype(np.uint8)
out=nib.Nifti1Image(labels,ref.affine,h);out.set_qform(ref.get_qform(),int(ref.header['qform_code']));out.set_sform(ref.get_sform(),int(ref.header['sform_code']));nib.save(out,args.output)
print(json.dumps({'output':args.output,'model':model,'warnings':warnings}))
