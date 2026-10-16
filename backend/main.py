from pathlib import Path
import asyncio, contextlib, datetime, hashlib, html, json, os, shutil, subprocess, tempfile, threading, time, uuid
import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from .imaging import LIMIT,MODALITIES,InputError,validate,volumes,metrics,prepare_viewer,digest,surface,regions
from .inference import predict,model_ready
ROOT=Path(__file__).resolve().parents[1]
JOBS=Path(os.environ.get('NEUROSEG_JOB_DIR',str(ROOT/'.runtime'/'jobs')))
JOBS.mkdir(parents=True,exist_ok=True)
RETENTION=23*3600
busy=threading.Lock()
DISCLAIMER='Research and teaching only. Never use model output as a diagnosis.'

def cleanup():
    for p in JOBS.iterdir():
        if p.is_dir() and time.time()-p.stat().st_mtime>RETENTION:shutil.rmtree(p,ignore_errors=True)

@contextlib.asynccontextmanager
async def lifespan(app):
    cleanup()
    async def sweeper():
        while True:
            await asyncio.sleep(60);await run_in_threadpool(cleanup)
    task=asyncio.create_task(sweeper())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):await task
app=FastAPI(title='NeuroSeg local research API',version='1.0.0',lifespan=lifespan)

@app.middleware('http')
async def local_only(request:Request,call_next):
    # No cross-origin upload or permissive CORS: same-origin localhost UI only.
    origin=request.headers.get('origin')
    if origin and origin!=str(request.base_url).rstrip('/'):
        return HTMLResponse('Cross-origin access refused.',status_code=403)
    length=request.headers.get('content-length')
    if length and int(length)>5*LIMIT+100_000: return HTMLResponse('Upload exceeds request limit.',status_code=413)
    response=await call_next(request)
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store';response.headers['X-Content-Type-Options']='nosniff'
    return response

@app.get('/api/health')
def health():return {'status':'ok','model_ready':model_ready(),'disclaimer':DISCLAIMER}

def process(paths,expert,out):
    started=time.perf_counter();images,truth,warnings=validate(paths,expert)
    labels,model=predict(images)
    ref=images['T1'];header=ref.header.copy();header.set_data_dtype(np.uint8)
    export=nib.Nifti1Image(labels,ref.affine,header=header)
    export.set_qform(ref.get_qform(),int(ref.header['qform_code']));export.set_sform(ref.get_sform(),int(ref.header['sform_code']))
    nib.save(export,out/'prediction.nii.gz')
    geometry=prepare_viewer(images,labels,out)
    if truth is not None:
        from nibabel.processing import resample_from_to
        from scipy.ndimage import gaussian_filter
        display_truth=resample_from_to(nib.Nifti1Image(truth.astype('uint8'),ref.affine),(geometry['shape'],np.array(geometry['affine'])),order=0).get_fdata().astype('uint8')
        display_truth.ravel(order='F').tofile(out/'expert-labels.bin')
        em={key:surface(gaussian_filter(mask.astype(float),.6),np.array(geometry['affine']),.5) for key,mask in regions(display_truth).items()}
        (out/'expert-meshes.json').write_text(json.dumps(em,separators=(',',':')))
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:commit='uncommitted-local-source'
    metadata={**geometry,'name':'Research study '+out.name[:8],'demo':False,'description':'Suspected tumour regions predicted by the model.','regions':volumes(labels,ref.affine),'expert':volumes(truth,ref.affine) if truth is not None else None,'metrics':metrics(labels,truth) if truth is not None else None,'model':model,'warnings':warnings,'input_sha256':{k:digest(v) for k,v in paths.items()},'expert_sha256':digest(expert) if expert else None,'code_commit':commit,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime_seconds':time.perf_counter()-started,'disclaimer':DISCLAIMER,'limitations':['No clinical validation.','No project-level held-out evaluation yet.','No uncertainty estimate.','Target T4 two-minute budget not benchmarked.','Input sanity checks cannot certify brain anatomy.']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'report.html').write_text(report_html(metadata),encoding='utf-8')
    # Remove source uploads immediately after successful output preparation.
    for p in paths.values():Path(p).unlink(missing_ok=True)
    if expert:Path(expert).unlink(missing_ok=True)
    return metadata

@app.post('/api/segment')
async def segment(T1:UploadFile|None=File(None),T1Gd:UploadFile|None=File(None),T2:UploadFile|None=File(None),FLAIR:UploadFile|None=File(None),expert:UploadFile|None=File(None)):
    uploads={'T1':T1,'T1Gd':T1Gd,'T2':T2,'FLAIR':FLAIR}
    if any(v is None for v in uploads.values()):raise HTTPException(422,'All four sequences are required: T1, T1Gd, T2, FLAIR.')
    if not busy.acquire(blocking=False):raise HTTPException(409,'One study is already processing. Wait for it to finish.')
    job=uuid.uuid4().hex;out=JOBS/job;out.mkdir();paths={};expert_path=None
    try:
        for name,upload in {**uploads,**({'expert':expert} if expert else {})}.items():
            suffix='.nii.gz' if upload.filename.lower().endswith('.nii.gz') else '.nii' if upload.filename.lower().endswith('.nii') else None
            if not suffix:raise InputError(f'{name}: expected .nii or .nii.gz.')
            path=out/(name+suffix);total=0
            with path.open('wb') as f:
                while chunk:=await upload.read(1024*1024):
                    total+=len(chunk)
                    if total>LIMIT:raise InputError(f'{name}: file exceeds 1 GB.')
                    f.write(chunk)
            if name=='expert':expert_path=path
            else:paths[name]=path
        await run_in_threadpool(process,paths,expert_path,out)
        return {'job_id':job,'status':'completed'}
    except InputError as e:
        shutil.rmtree(out,ignore_errors=True);raise HTTPException(422,str(e))
    except Exception:
        shutil.rmtree(out,ignore_errors=True);raise HTTPException(500,'Processing failed. No result is available. Check the local model installation and available memory.')
    finally:
        busy.release()
        for upload in [*uploads.values(),expert]:
            if upload:await upload.close()

def jobdir(job):
    if len(job)!=32 or any(c not in '0123456789abcdef' for c in job):raise HTTPException(404,'Study not found.')
    path=JOBS/job
    if not (path/'metadata.json').exists():raise HTTPException(404,'No completed study is available.')
    if time.time()-path.stat().st_mtime>RETENTION:raise HTTPException(410,'Study has expired and is being removed.')
    return path
@app.get('/api/jobs/{job}/metadata')
def metadata(job):return FileResponse(jobdir(job)/'metadata.json',media_type='application/json')
@app.get('/api/jobs/{job}/meshes')
def meshes(job):return FileResponse(jobdir(job)/'meshes.json',media_type='application/json')
@app.get('/api/jobs/{job}/labels')
def labels(job):return FileResponse(jobdir(job)/'labels.bin',media_type='application/octet-stream')
@app.get('/api/jobs/{job}/expert-labels')
def expert_labels(job):
    p=jobdir(job)/'expert-labels.bin'
    if not p.exists():raise HTTPException(404,'No expert mask was provided.')
    return FileResponse(p,media_type='application/octet-stream')
@app.get('/api/jobs/{job}/expert-meshes')
def expert_meshes(job):
    p=jobdir(job)/'expert-meshes.json'
    if not p.exists():raise HTTPException(404,'No expert mask was provided.')
    return FileResponse(p,media_type='application/json')
@app.get('/api/jobs/{job}/volume/{seq}')
def volume(job,seq):
    if seq not in MODALITIES:raise HTTPException(404,'Sequence not found.')
    return FileResponse(jobdir(job)/(seq+'.bin'),media_type='application/octet-stream')
@app.get('/api/jobs/{job}/prediction')
def prediction(job):return FileResponse(jobdir(job)/'prediction.nii.gz',filename='prediction.nii.gz',media_type='application/gzip')
@app.get('/api/jobs/{job}/report')
def report(job):return FileResponse(jobdir(job)/'report.html',media_type='text/html')

def report_html(m):
    rows=''.join(f'<tr><td>{k}</td><td>{v["voxel_count"]:,}</td><td>{v["voxel_volume_mm3"]:.6f}</td><td>{v["volume_ml"]:.3f}</td></tr>' for k,v in m['regions'].items())
    return '<!doctype html><html><head><meta charset="utf-8"><title>NeuroSeg research report</title><style>body{font:14px system-ui;max-width:880px;margin:40px auto;padding:20px;color:#182a42}p{line-height:1.7}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;font-size:11px;background:#f0f4f8;padding:18px;overflow-wrap:anywhere}.note{background:#edf2f7;padding:16px}@media print{button{display:none}}</style></head><body><h1>NeuroSeg research report</h1><p class="note"><b>'+DISCLAIMER+'</b></p><p>Suspected tumour regions predicted by the model. Estimated volumes below do not establish a diagnosis.</p><h2>Estimated volumes</h2><table><tr><th>Region</th><th>Voxels</th><th>mm³ per voxel</th><th>mL</th></tr>'+rows+'</table><p>Volume = voxel count × |det(original affine 3×3)| / 1,000.</p><h2>Expert comparison</h2><pre>'+html.escape(json.dumps({'expert_volumes':m['expert'],'per_case_metrics':m['metrics'],'empty_mask_convention':'Dice=1 when both empty; sensitivity undefined (null) when reference empty'},indent=2))+'</pre><h2>Provenance, warnings and limitations</h2><pre>'+html.escape(json.dumps(m,indent=2))+'</pre><button onclick="print()">Print / Save as PDF</button></body></html>'

app.mount('/',StaticFiles(directory=ROOT/'dist',html=True),name='workspace')
