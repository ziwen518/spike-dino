"""Full-shard supervised AugReg reference; not a DINO or SpikeCLR replication."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,gc,hashlib,json,math,random,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models import resnet18,vit_b_16
from spike_dino.benchmark_manifest import load_manifest
from scripts.run_dvs_standard_ann_triage import RepresentationPairs

ROOT=Path('runs/generalization_reference_20261103')
DATA=Path('dataset/mini_nimagenet')

def atomic(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
 for attempt in range(50):
  try:temp.replace(path);return
  except PermissionError:
   if attempt==49:raise
   time.sleep(.1)

def gpu_idle():
 used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 if used>2300:raise RuntimeError(f'GPU0 occupied: {used}MiB')

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 ready=ROOT/'cache_ready.json'
 if ready.exists():return json.loads(ready.read_text())
 if shutil.disk_usage(ROOT).free<15*1024**3:raise RuntimeError('Less than15GiB free; inspect storage before caching')
 manifest=load_manifest(DATA/'metadata/part1_manifest.json','mini_nimagenet');rows=manifest['train']
 if len(rows)!=13000 or len(manifest['class_synsets'])!=10:raise ValueError('Unexpected full shard')
 labels=np.array([r['label'] for r in rows],dtype=np.int64);rng=np.random.default_rng(31415);fit=[];dev=[]
 for label in range(10):
  ix=np.flatnonzero(labels==label);rng.shuffle(ix);dev.extend(ix[:130].tolist());fit.extend(ix[130:].tolist())
 if set(fit)&set(dev) or len(set(fit+dev))!=13000:raise RuntimeError('Split invalid')
 spec={'manifest_sha256':manifest['sha256'],'fit_indices':fit,'development_indices':dev,'rows':rows,
  'split_seed':31415,'fit_count':len(fit),'development_count':len(dev),'official_validation_used_for_selection':False,
  'representation':'identical prior triage max-pool T4 counts collapsed2channels, unit max; float16 cache; full13000 training shard'}
 spec['split_sha256']=hashlib.sha256(json.dumps({'fit':[rows[i] for i in fit],'dev':[rows[i] for i in dev]},sort_keys=True).encode()).hexdigest()
 if (ROOT/'split.json').exists():
  if json.loads((ROOT/'split.json').read_text())['split_sha256']!=spec['split_sha256']:raise RuntimeError('Resume split differs')
 else:atomic(ROOT/'split.json',spec)
 target=ROOT/'training_histograms.npy';partial=ROOT/'training_histograms.partial.npy'
 if target.exists():raise RuntimeError('Completed cache without ready marker; inspect')
 index=0
 if partial.exists():
  progress=json.loads((ROOT/'preparation_status.json').read_text());index=progress['records']
  mapped=np.load(partial,mmap_mode='r+')
  if mapped.shape!=(len(rows),2,128,128) or mapped.dtype!=np.float16 or not 0<=index<=len(rows):raise RuntimeError('Invalid partial cache')
  for start in range(0,index,256):
   block=mapped[start:min(index,start+256)]
   if not np.isfinite(block).all() or np.any(block.reshape(len(block),-1).sum(1)==0):raise RuntimeError('Invalid cached prefix')
  if index:del block
  print('RESUME_CACHE',index,flush=True)
 else:mapped=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=(len(rows),2,128,128))
 loader=DataLoader(RepresentationPairs(DATA,rows[index:]),batch_size=32,shuffle=False,num_workers=4)
 for x,y in loader:
  n=len(y)
  if not np.array_equal(y.numpy(),labels[index:index+n]):raise RuntimeError('Cache label mismatch')
  values=x[:,0].numpy()
  if not np.isfinite(values).all() or np.any(values.reshape(n,-1).sum(1)==0):raise RuntimeError('Invalid image')
  mapped[index:index+n]=values;index+=n
  atomic(ROOT/'preparation_status.json',{'state':'caching','records':index,'total':len(rows)})
  if index%1024==0:print('CACHE',index,flush=True)
 mapped.flush();mapping_handle=mapped._mmap;del mapped;mapping_handle.close();del mapping_handle
 gc.collect()
 for attempt in range(50):
  try:partial.replace(target);break
  except PermissionError:
   if attempt==49:raise
   time.sleep(.1)
 np.save(ROOT/'labels.npy',labels)
 atomic(ready,{'records':index,'shape':[len(rows),2,128,128],'split_sha256':spec['split_sha256'],'cache_bytes':target.stat().st_size})
 atomic(ROOT/'preparation_status.json',{'state':'completed','records':index});return spec

class CachedViews(Dataset):
 def __init__(self,indices,augment):
  self.indices=indices;self.augment=augment;self.images=None;self.labels=None;self.epoch=torch.zeros(1,dtype=torch.int64).share_memory_()
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  if self.images is None:
   self.images=np.load(ROOT/'training_histograms.npy',mmap_mode='r');self.labels=np.load(ROOT/'labels.npy')
  index=self.indices[i];x=torch.from_numpy(np.array(self.images[index],dtype=np.float32,copy=True))
  if self.augment:
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42+index*1009+int(self.epoch[0])*1000003)
    top,left,h,w=T.RandomResizedCrop.get_params(x,scale=(.5,1.),ratio=(.75,4/3))
    x=TF.resized_crop(x,top,left,h,w,[224,224],T.InterpolationMode.BILINEAR,antialias=True)
    if torch.rand(())<.5:x=x.flip(-1)
    if torch.rand(())<.5:x=x.flip(0)
  else:x=TF.resize(x,[224,224],T.InterpolationMode.BILINEAR,antialias=True)
  return x,int(self.labels[index])

def model_for(arm):
 if arm=='resnet18':
  m=resnet18(weights=None,num_classes=10);m.conv1=nn.Conv2d(2,64,7,stride=2,padding=3,bias=False)
 else:
  m=vit_b_16(weights=None,image_size=224,num_classes=10,dropout=.1)
  m.conv_proj=nn.Conv2d(2,768,16,stride=16)
  if arm=='vit_conv_stem':
   from spike_dino.event_conv_stem import EventConvStem
   m.conv_proj=EventConvStem(2,768)
 return m

def assess(m,loader):
 m.eval();n=c1=c5=0;ce=0.
 with torch.no_grad():
  for x,y in loader:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x)
   z=z.float();n+=len(y);c1+=int(z.argmax(-1).eq(y).sum());c5+=int(z.topk(5,-1).indices.eq(y[:,None]).any(-1).sum());ce+=float(F.cross_entropy(z,y,reduction='sum'))
 return {'acc1':c1/n,'acc5':c5/n,'ce':ce/n,'records':n}

def mixed_batch(x,y,rng):
 targets=F.one_hot(y,10).float()*.9+.01
 order=torch.randperm(len(y),device=y.device)
 if rng.random()<.5:
  lam=float(rng.beta(.8,.8));return x*lam+x[order]*(1-lam),targets*lam+targets[order]*(1-lam)
 lam=float(rng.beta(1.,1.));h,w=x.shape[-2:];ratio=math.sqrt(1-lam)
 cx,cy=rng.integers(w),rng.integers(h);bw,bh=int(w*ratio),int(h*ratio)
 x0,x1=max(0,cx-bw//2),min(w,cx+bw//2);y0,y1=max(0,cy-bh//2),min(h,cy+bh//2)
 x[:,:,y0:y1,x0:x1]=x[order,:,y0:y1,x0:x1]
 lam=1-(x1-x0)*(y1-y0)/(h*w)
 return x,targets*lam+targets[order]*(1-lam)

def train(arm,epochs):
 gpu_idle();torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.cuda.manual_seed_all(42)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
 spec=json.loads((ROOT/'split.json').read_text());folder=ROOT/arm;folder.mkdir(exist_ok=True)
 if (folder/'status.json').exists():raise RuntimeError('Existing arm; explicit resume needed')
 generator=torch.Generator().manual_seed(42);mix_rng=np.random.default_rng(42)
 loader=DataLoader(CachedViews(spec['fit_indices'],True),batch_size=64,shuffle=True,generator=generator,num_workers=4,pin_memory=True,persistent_workers=True)
 dev=DataLoader(CachedViews(spec['development_indices'],False),batch_size=64,num_workers=2,pin_memory=True,persistent_workers=True)
 fit_probe=DataLoader(CachedViews(spec['fit_indices'][::10],False),batch_size=64,num_workers=2,pin_memory=True,persistent_workers=True)
 base_lr=.001 if arm=='resnet18' else .000125
 m=model_for(arm).cuda();opt=torch.optim.AdamW(m.parameters(),lr=base_lr,weight_decay=.05)
 state={'state':'running','arm':arm,'model_parameters':sum(p.numel() for p in m.parameters()),'epoch':0,
  'tokenizer':'four3x3 stride2 conv+BN+GELU and1x1 projection' if arm=='vit_conv_stem' else 'standard',
  'maximum_epochs':epochs,'fit_records':len(spec['fit_indices']),'development_records':len(spec['development_indices']),
  'split_sha256':spec['split_sha256'],'seed':42,'supervised':True,'RGB_pretraining':False,'official_validation_read':False,
  'base_lr':base_lr,'recipe':'batch64 accumulation2; AdamW CNNlr.001/ViTlr.000125 wd.05; warmup5 cosine->.002x; RRC.5-1 flip polarity; mixup.8/cutmix1; smoothing.1; ViTdropout.1; BF16; clip1',
  'selection':'devAcc1 then CE; minimum30 epochs; patience3 evaluations =15epochs; maximum100; new reference, not old Base resume',
  'history':[],'started':time.time()};best=(-1.,float('-inf'));stale=0;step=0
 def checkpoint(epoch):return {'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,'state':state,
  'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state(),
  'numpy_rng':mix_rng.bit_generator.state,'python_rng':random.getstate()}
 try:
  atomic(folder/'status.json',state)
  for epoch in range(1,epochs+1):
   loader.dataset.epoch[0]=epoch
   m.train();opt.zero_grad(set_to_none=True);total=0.;count=0;norms=[];start=time.time()
   for batch,(x,y) in enumerate(loader):
    progress=(epoch-1)+batch/len(loader)
    factor=min(1.,(progress+1/len(loader))/5.) if progress<5 else .002+.998*.5*(1+math.cos(math.pi*(progress-5)/max(epochs-5,1)))
    for group in opt.param_groups:group['lr']=base_lr*factor
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);x,target=mixed_batch(x,y,mix_rng)
    with torch.autocast('cuda',dtype=torch.bfloat16):scores=m(x);loss=-(target*F.log_softmax(scores.float(),-1)).sum(-1).mean()
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    # The last singleton accumulation group is scaled by its actual length.
    divisor=1 if batch==len(loader)-1 and len(loader)%2 else 2
    (loss/divisor).backward();total+=float(loss)*len(y);count+=len(y)
    if (batch+1)%2==0 or batch==len(loader)-1:
     norm=nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);norms.append(float(norm));opt.step();opt.zero_grad(set_to_none=True);step+=1
    if batch%40==0:
     state.update(epoch=epoch,batch=batch,batches=len(loader),optimizer_step=step,heartbeat=time.time());atomic(folder/'status.json',state)
   record={'epoch':epoch,'train_mixed_CE':total/count,'seconds':time.time()-start,'gradient_norm_max_before_clip':max(norms),'lr':opt.param_groups[0]['lr']}
   if epoch==1 or epoch%5==0:
    record['development']=assess(m,dev);record['unaugmented_fit_subset']=assess(m,fit_probe)
    metric=(record['development']['acc1'],-record['development']['ce'])
    if metric>best:
     best=metric;stale=0;state['best_epoch']=epoch;state['best_development']=record['development'];torch.save(checkpoint(epoch),folder/'best.pt')
    else:stale+=1
   state['history'].append(record);state.update(epoch=epoch,optimizer_step=step,heartbeat=time.time());atomic(folder/'status.json',state)
   print(arm,json.dumps(record),flush=True)
   # Save a restart point every five epochs, plus the first completed epoch.
   if epoch==1 or epoch%5==0:torch.save(checkpoint(epoch),folder/'last.pt')
   if epoch>=30 and stale>=3:
    state['stop_reason']='15epochs_without_development_improvement';break
  state['state']='completed';state.setdefault('stop_reason','maximum_epochs')
 except BaseException as exc:state['state']='failed';state['error']=repr(exc);raise
 finally:
  state['finished']=time.time();atomic(folder/'status.json',state)
  del m,opt,loader,dev,fit_probe;gc.collect();torch.cuda.empty_cache()

def queue(epochs,resume_preparation=False):
 ROOT.mkdir(parents=True,exist_ok=True)
 attempts=[]
 if (ROOT/'queue.json').exists():
  prior=json.loads((ROOT/'queue.json').read_text())
  if not resume_preparation or prior.get('state')!='failed' or prior.get('stage')!='prepare':raise RuntimeError('Existing queue cannot be resumed automatically')
  attempts=prior.get('previous_attempts',[])+[{k:v for k,v in prior.items() if k!='previous_attempts'}]
 state={'state':'running','stage':'prepare','pid':os.getpid(),'started':time.time(),'arms':['resnet18','vit_base'],'previous_attempts':attempts};atomic(ROOT/'queue.json',state)
 try:
  prepare()
  for arm in state['arms']:
   gpu_idle();state['stage']=arm;atomic(ROOT/'queue.json',state)
   with (ROOT/(arm+'.log')).open('w',encoding='utf-8') as log:
    child=subprocess.Popen([__import__('sys').executable,'-u','-m','scripts.train_generalization_reference','--arm',arm,'--epochs',str(epochs)],stdout=log,stderr=subprocess.STDOUT)
    state['child_pid']=child.pid;atomic(ROOT/'queue.json',state);code=child.wait()
   state['last_exit_code']=code;state['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],text=True).strip()
   atomic(ROOT/'queue.json',state)
   if code:raise RuntimeError(f'{arm} failed with {code}')
  state['state']='completed'
 except BaseException as exc:state['state']='failed';state['error']=repr(exc);raise
 finally:state['finished']=time.time();atomic(ROOT/'queue.json',state)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['resnet18','vit_base','vit_conv_stem','queue','prepare'],default='queue');parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--resume-preparation',action='store_true');args=parser.parse_args()
 if args.arm=='queue':queue(args.epochs,args.resume_preparation)
 elif args.arm=='prepare':prepare()
 else:train(args.arm,args.epochs)
