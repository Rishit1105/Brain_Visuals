"""Synthetic tests: no dataset or model-performance claims."""
import gzip,json,tempfile,unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from backend.imaging import load_volume,validate,volumes,metrics,InputError
class ImagingTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.p=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,a,aff=None):
        p=self.p/(name+'.nii.gz');nib.save(nib.Nifti1Image(a,np.eye(4) if aff is None else aff),p);return p
    def test_cube(self):
        a=np.zeros((30,30,30),dtype='uint8');a[2:22,2:22,2:22]=2
        r=volumes(a,np.diag([1,1,2,1]));self.assertAlmostEqual(r['WT']['volume_ml'],16);self.assertEqual(r['TC']['volume_ml'],0);self.assertEqual(r['ET']['volume_ml'],0)
    def test_shear_volume(self):
        a=np.ones((10,10,10),dtype='uint8');aff=np.eye(4);aff[0,1]=.5
        self.assertAlmostEqual(volumes(a,aff)['WT']['volume_ml'],1)
    def test_bad_files(self):
        for ext,content in [('x.nii.gz',b'not nifti'),('y.nii.gz',b'\x1f\x8bgarbage')]:
            p=self.p/ext;p.write_bytes(content)
            with self.assertRaises(InputError):load_volume(p,'T1')
    def test_2d_constant_nan(self):
        for a in [np.ones((3,3),dtype='float32'),np.zeros((3,3,3),dtype='float32'),np.full((3,3,3),np.nan,dtype='float32')]:
            p=self.write('bad',a)
            with self.assertRaises(InputError):load_volume(p,'T1')
    def test_mask_codes(self):
        p=self.write('mask',np.full((4,4,4),3,dtype='uint8'))
        with self.assertRaises(InputError):load_volume(p,'Expert',True)
    def test_missing(self):
        with self.assertRaises(InputError):validate({})
    def test_geometry_mismatch(self):
        a=np.random.default_rng(42).random((10,10,10)).astype('float32');paths={k:self.write(k,a) for k in ['T1','T1Gd','T2','FLAIR']}
        paths['T2']=self.write('other',a,np.diag([-1,1,1,1]))
        with self.assertRaisesRegex(InputError,'affine'):validate(paths)
        paths['T2']=self.write('other',a[:5])
        with self.assertRaisesRegex(InputError,'shape'):validate(paths)
    def test_orientation_same_anatomy(self):
        a=np.zeros((8,9,10),dtype='uint8');a[2,4,5]=4;aff=np.diag([1,2,3,1]);flip=np.eye(4);flip[0,0]=-1;flip[0,3]=7
        original=nib.affines.apply_affine(aff,[2,4,5]);changed=nib.affines.apply_affine(aff@flip,[5,4,5]);np.testing.assert_allclose(original,changed)
        self.assertAlmostEqual(volumes(a,aff)['WT']['volume_ml'],volumes(a[::-1],aff@flip)['WT']['volume_ml'])
    def test_metric(self):
        p=np.zeros(30000,dtype='uint8');g=p.copy();p[:12000]=2;g[:9000]=2;g[15000:16000]=2;r=metrics(p,g)['WT'];self.assertAlmostEqual(r['dice'],18000/22000);self.assertEqual(r['sensitivity'],.9)
    def test_empty(self):
        a=np.zeros((3,3,3),dtype='uint8');r=metrics(a,a);self.assertEqual(r['WT']['dice'],1);self.assertIsNone(r['WT']['sensitivity'])
    def test_export_geometry(self):
        a=np.zeros((8,9,10),dtype='uint8');a[2:5]=4;aff=np.array([[-1,0,0,7],[0,2,0,-10],[0,0,3,8],[0,0,0,1.]])
        p=self.write('export',a,aff);im=nib.load(p);np.testing.assert_allclose(im.affine,aff);self.assertEqual(im.shape,a.shape);self.assertEqual(set(np.unique(im.get_fdata())),{0,4})
if __name__=='__main__':unittest.main(verbosity=2)
