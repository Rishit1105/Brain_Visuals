"""Validated NIfTI geometry, measurement and viewer preparation."""
from pathlib import Path
import hashlib, json
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import gaussian_filter, label
from skimage.measure import marching_cubes

LIMIT=1_000_000_000
MODALITIES=('T1Gd','T1','T2','FLAIR')
class InputError(ValueError):
    pass

def load_volume(path, name, mask=False):
    path=Path(path)
    if path.stat().st_size>LIMIT: raise InputError(f'{name}: file exceeds 1 GB.')
    if not str(path).lower().endswith(('.nii','.nii.gz')): raise InputError(f'{name}: expected .nii or .nii.gz.')
    try:
        im=nib.load(path)
        if not isinstance(im,(nib.Nifti1Image,nib.Nifti2Image)): raise InputError(f'{name}: unsupported image format.')
        if len(im.shape)!=3: raise InputError(f'{name}: must be exactly 3D.')
        if np.prod(im.shape)*max(im.get_data_dtype().itemsize,4)>LIMIT: raise InputError(f'{name}: expanded image exceeds 1 GB.')
        if np.prod(im.shape)>64_000_000: raise InputError(f'{name}: this prototype supports at most 64 million voxels per sequence.')
        if not np.isfinite(im.affine).all() or abs(np.linalg.det(im.affine[:3,:3]))<1e-8: raise InputError(f'{name}: spatial affine is invalid.')
        if im.header.get_xyzt_units()[0] not in ('mm','unknown'): raise InputError(f'{name}: spatial units must be millimetres.')
        data=im.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all(): raise InputError(f'{name}: contains NaN or infinite intensities.')
        if mask:
            if not np.isin(data,[0,1,2,4]).all(): raise InputError(f'{name}: expert labels must use BraTS codes 0, 1, 2, 4.')
        elif data.max()==data.min(): raise InputError(f'{name}: volume is constant (including all-zero scans).')
        return im,data
    except InputError: raise
    except Exception as e: raise InputError(f'{name}: unreadable or corrupt NIfTI/gzip file.') from e

def validate(paths,expert=None):
    if set(paths)!=set(MODALITIES): raise InputError('All four sequences are required: T1, T1Gd, T2, FLAIR.')
    images={};arrays={};warnings=[]
    for name in MODALITIES:images[name],arrays[name]=load_volume(paths[name],name)
    ref=images['T1']
    for name,im in images.items():
        if im.shape!=ref.shape: raise InputError(f'{name}: shape differs from T1.')
        if not np.allclose(im.affine,ref.affine,atol=1e-4,rtol=0): raise InputError(f'{name}: affine differs from T1; co-register the sequences first.')
    exp=None
    if expert:
        ei,exp=load_volume(expert,'Expert mask',mask=True)
        if ei.shape!=ref.shape or not np.allclose(ei.affine,ref.affine,atol=1e-4,rtol=0): raise InputError('Expert mask: shape or affine differs from T1.')
    a=arrays['T1']; foreground=np.abs(a)>max(float(np.max(np.abs(a)))*.001,1e-8)
    fraction=foreground.mean();edges=np.concatenate([foreground[0].ravel(),foreground[-1].ravel(),foreground[:,0].ravel(),foreground[:,-1].ravel(),foreground[:,:,0].ravel(),foreground[:,:,-1].ravel()])
    if fraction<.01 or fraction>.90: raise InputError('Input is clearly unsuitable: foreground occupancy is inconsistent with a skull-stripped brain MRI.')
    if edges.mean()>.12: raise InputError('Input touches too much of the image boundary. Provide a skull-stripped, uncropped brain MRI.')
    extent=nib.affines.voxel_sizes(ref.affine)*np.array(ref.shape)
    if min(extent)<65 or max(extent)>400: raise InputError('Physical field of view is inconsistent with an adult brain study.')
    warnings.append('Anatomical and skull-stripping checks are heuristics; they cannot certify that this is a valid brain MRI.')
    if any(im.header.get_xyzt_units()[0]=='unknown' for im in images.values()):warnings.append('Spatial units are unspecified in an input header; millimetres were assumed.')
    return images,exp,warnings

def regions(labels): return {'WT':labels>0,'TC':(labels==1)|(labels==4),'ET':labels==4}
def volumes(labels,affine):
    voxel=float(abs(np.linalg.det(affine[:3,:3])))
    return {k:{'voxel_count':int(v.sum()),'voxel_volume_mm3':voxel,'volume_ml':float(v.sum()*voxel/1000)} for k,v in regions(labels).items()}
def metrics(pred,truth):
    out={}
    for k,p in regions(pred).items():
        g=regions(truth)[k];tp=int((p&g).sum());pn=int(p.sum());gn=int(g.sum())
        out[k]={'dice':2*tp/(pn+gn) if pn+gn else 1.,'sensitivity':tp/gn if gn else None,'reference_empty':gn==0}
    return out
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def surface(data,affine,level):
    if data.min()==data.max() or not data.min()<level<data.max():return {'vertices':[],'indices':[]}
    verts,faces,_,_=marching_cubes(data.astype('float32'),level,allow_degenerate=False)
    verts=nib.affines.apply_affine(affine,verts)
    # Affine reflections reverse winding; normals are recomputed by the viewer.
    if np.linalg.det(affine[:3,:3])<0:faces=faces[:,::-1]
    return {'vertices':np.round(verts,3).ravel().tolist(),'indices':faces.ravel().tolist()}
def prepare_viewer(images,prediction,out):
    out=Path(out);ref=images['T1']
    # Axis-aligned RAS+ display grid also handles oblique originals correctly.
    target=resample_to_output(ref,voxel_sizes=(1.5,1.5,1.5),order=1)
    a=target.get_fdata(dtype=np.float32);grid=(target.shape,target.affine)
    for name,im in images.items():
        data=resample_from_to(im,grid,order=1).get_fdata(dtype=np.float32)
        nz=data[data!=0];lo,hi=np.percentile(nz,[1,99]) if len(nz) else (0,1)
        u=np.clip((data-lo)/max(hi-lo,1e-6)*255,0,255).astype('uint8');u[data==0]=0
        u.ravel(order='F').tofile(out/f'{name}.bin')
    predim=nib.Nifti1Image(prediction.astype('uint8'),ref.affine)
    lab=resample_from_to(predim,grid,order=0).get_fdata().astype('uint8');lab.ravel(order='F').tofile(out/'labels.bin')
    nz=a[a>0];level=float(np.percentile(nz,20)) if nz.size else .5
    meshes={'brain':surface(gaussian_filter(a,.5),target.affine,level)}
    for name,mask in regions(lab).items():meshes[name]=surface(gaussian_filter(mask.astype(float),.6),target.affine,.5)
    (out/'meshes.json').write_text(json.dumps(meshes,separators=(',',':')))
    return {'shape':list(target.shape),'affine':target.affine.tolist(),'spacing':nib.affines.voxel_sizes(target.affine).tolist(),'original_shape':list(ref.shape),'original_affine':ref.affine.tolist()}
