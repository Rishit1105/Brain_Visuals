"""Real SegResNet inference. No sample, heuristic or stored-label fallback."""
import json, os, threading, time
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to,resample_to_output
from .imaging import MODALITIES,digest,InputError
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'model.pt'
LOCK=ROOT/'models'/'model-lock.json'
_network=None
_mutex=threading.Lock()

def model_ready():return MODEL.exists() and LOCK.exists()
def load_model():
    global _network
    if _network is not None:return _network
    if not model_ready():raise InputError('Verified model checkpoint is missing. Run python setup_model.py before inference.')
    import torch
    from monai.networks.nets import SegResNet
    lock=json.loads(LOCK.read_text())
    if digest(MODEL)!=lock['sha256']:raise InputError('Checkpoint hash does not match model-lock.json. Inference refused.')
    torch.manual_seed(0);np.random.seed(0);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':torch.set_num_threads(max(1,int(os.environ.get('NEUROSEG_CPU_THREADS','6'))))
    network=SegResNet(blocks_down=[1,2,2,4],blocks_up=[1,1,1],init_filters=16,in_channels=4,out_channels=3,dropout_prob=.2)
    weights=torch.load(MODEL,map_location='cpu',weights_only=True)
    if isinstance(weights,dict) and 'state_dict' in weights:weights=weights['state_dict']
    network.load_state_dict(weights,strict=True);network.to(device).eval();_network=(network,device,lock)
    return _network

def predict(images):
    import torch
    from monai.inferers import sliding_window_inference
    with _mutex:
        network,device,lock=load_model();ref=images['T1']
        # Continuous inputs are resampled in world space. Discrete outputs return via nearest neighbour.
        canonical=nib.as_closest_canonical(ref)
        gridim=resample_to_output(canonical,voxel_sizes=(1,1,1),order=1)
        if np.prod(gridim.shape)>32_000_000:raise InputError('Resampled grid exceeds the supported inference memory budget.')
        channels=[]
        for name in MODALITIES:
            a=resample_from_to(images[name],(gridim.shape,gridim.affine),order=1).get_fdata(dtype=np.float32)
            mask=a!=0
            if mask.any():a[mask]=(a[mask]-a[mask].mean())/max(float(a[mask].std()),1e-8)
            channels.append(a)
        x=torch.from_numpy(np.stack(channels)[None]).to(device)
        with torch.inference_mode():
            p=sliding_window_inference(x,(128,128,128),1,network,overlap=.5,mode='constant').sigmoid()[0].cpu().numpy()>.5
        # Official bundle precedence: ET (4), remaining TC (1), remaining WT (2).
        labels=np.where(p[2],4,np.where(p[0],1,np.where(p[1],2,0))).astype('uint8')
        output=resample_from_to(nib.Nifti1Image(labels,gridim.affine),(ref.shape,ref.affine),order=0).get_fdata().astype('uint8')
        return output,{'name':'MONAI brats_mri_segmentation','bundle_version':'0.5.4','architecture':'SegResNet','licence':'Apache-2.0','revision':lock['revision'],'checkpoint_sha256':lock['sha256'],'device':str(device),'torch':torch.__version__,'preprocessing':'RAS+; 1mm world-space linear resampling; per-channel nonzero z-score; T1Gd,T1,T2,FLAIR','inference':'128 cubed sliding window, overlap 0.5, sigmoid > 0.5; ET > TC > WT precedence; nearest-neighbour original-grid restoration','uncertainty':'Not measured','training_data':'BraTS 2018; pretraining/evaluation overlap unverified'}
