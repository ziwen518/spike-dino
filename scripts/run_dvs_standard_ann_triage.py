"""Short supervised standard-ANN diagnostic; not an SSL/SpikeCLR benchmark."""
import argparse,gc,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision.models import resnet18,vit_b_16
from spike_dino.data import read_nimagenet_events
from spike_dino.benchmark_manifest import load_manifest


class RepresentationPairs(Dataset):
 def __init__(self,root,rows):self.root=root;self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  row=self.rows[index];events=read_nimagenet_events(self.root/'raw'/row['path'])
  t=np.asarray(events['t'],dtype=np.int64);x=events['x'];y=events['y'];p=events['p']
  bins=np.minimum((t-t[0])*4//max(int(t[-1]-t[0]),1),3)
  flat=((bins*2+p)*480+y)*640+x
  frames=np.bincount(flat,minlength=4*2*480*640).reshape(4,2,480,640).astype(np.float32)
  tensor=torch.from_numpy(frames);tensor=tensor/tensor.max().clamp_min(1)
  maximum=F.adaptive_max_pool2d(tensor,(128,128)).sum(0)
  average=F.adaptive_avg_pool2d(tensor,(128,128)).sum(0)
  # Same post-pool amplitude normalization in BOTH arms, so spatial
  # morphology rather than overall pooling amplitude is the changed factor.
  pair=torch.stack([image/image.max().clamp_min(1e-12) for image in (maximum,average)])
  return pair,row['label']


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--train-limit',type=int,default=2600);p.add_argument('--epochs',type=int,default=3)
 p.add_argument('--output',type=Path,default=Path('runs/standard_ann_triage_20261103'))
 a=p.parse_args();torch.set_num_threads(2)
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 root=Path('dataset/mini_nimagenet');manifest=load_manifest(root/'metadata/part1_manifest.json','mini_nimagenet')
 if len(manifest['class_synsets'])!=10 or a.train_limit%10:raise ValueError('Ten balanced classes required')
 rng=np.random.default_rng(42);selected=[]
 for label in range(10):
  candidates=[row for row in manifest['train'] if row['label']==label]
  ix=rng.choice(len(candidates),a.train_limit//10,replace=False)
  selected += [candidates[int(i)] for i in ix]
 validation=manifest['test']
 for rows in (selected,validation):
  if any(row['path'].split('/')[1]!=manifest['class_synsets'][row['label']] for row in rows):
   raise ValueError('Path synset and label mapping differ')
 if {r['path'] for r in selected}&{r['path'] for r in validation}:raise ValueError('Split overlap')
 status={'state':'running','stage':'cache','train_records':len(selected),'validation_records':len(validation),
  'manifest_sha256':manifest['sha256'],'selected_train_sha256':hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
  'epochs':a.epochs,'batch_size':32,'seed':42,'started':time.time(),'completed':[],
  'claim':'short full-supervised standard ANN diagnostic; no pretraining, no publication score',
  'representation':'T4 counts minmax; max/avg spatial pool, sumT, identical per-image max normalization',
  'arms':{}}
 def save():(a.output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 def cache(rows,name):
  loader=DataLoader(RepresentationPairs(root,rows),batch_size=16,shuffle=False,num_workers=4,
    generator=torch.Generator().manual_seed(42))
  images=[];labels=[]
  for batch,(x,y) in enumerate(loader):
   images.append(x);labels.append(y)
   if batch%25==0:print('CACHE',name,batch*16,len(rows),flush=True)
  x=torch.cat(images);y=torch.cat(labels)
  if not torch.isfinite(x).all() or not x.any():raise RuntimeError('Invalid cached representation')
  torch.save({'inputs':x,'labels':y},a.output/(name+'_cache.pt'))
  return x,y
 def assess(model,x,y):
  model.eval();correct=0;correct5=0;ce=0.;n=len(y)
  with torch.no_grad():
   for start in range(0,n,32):
    scores=model(x[start:start+32].cuda());target=y[start:start+32].cuda()
    correct+=int(scores.argmax(-1).eq(target).sum())
    correct5+=int(scores.topk(5,-1).indices.eq(target[:,None]).any(-1).sum())
    ce+=float(F.cross_entropy(scores,target,reduction='sum'))
  model.train();return {'acc1':correct/n,'acc5':correct5/n,'ce':ce/n,'records':n}
 try:
  save();train,ty=cache(selected,'train');valid,vy=cache(validation,'validation')
  status['input_statistics']={}
  for i,name in enumerate(('max','avg')):
   x=train[:,i]
   status['input_statistics'][name]={'nonzero_pixel_fraction':float(x.ne(0).float().mean()),
    'mean':float(x.mean()),'std':float(x.std()),'mean_per_image_occupied':float(x.ne(0).float().flatten(1).mean(1).mean())}
  for arm,pool,kind,lr in [('resnet18_max',0,'cnn',1e-3),('resnet18_avg',1,'cnn',1e-3),('vit_base_max',0,'vit',1e-4)]:
   used=int(__import__('subprocess').check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if used>2200:raise RuntimeError(f'GPU0 busy before {arm}: {used}')
   torch.manual_seed(42);random.seed(42);np.random.seed(42)
   if kind=='cnn':
    model=resnet18(weights=None,num_classes=10)
    model.conv1=nn.Conv2d(2,64,7,stride=2,padding=3,bias=False)
   else:
    model=vit_b_16(weights=None,image_size=128,num_classes=10)
    model.conv_proj=nn.Conv2d(2,768,16,stride=16)
   model=model.cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
   generator=torch.Generator().manual_seed(42)
   loader=DataLoader(TensorDataset(train[:,pool],ty),batch_size=32,shuffle=True,
      generator=generator,num_workers=0,drop_last=True)
   history=[];status['arms'][arm]={'parameters':sum(p.numel() for p in model.parameters()),'lr':lr,'history':history}
   status['stage']=arm;save();print('START_ARM',arm,flush=True)
   for epoch in range(a.epochs+1):
    history.append({'epoch':epoch,'train':assess(model,train[:,pool],ty),
                     'validation':assess(model,valid[:,pool],vy)})
    save();print(arm,json.dumps(history[-1]),flush=True)
    if epoch==a.epochs:break
    for images,labels in loader:
     opt.zero_grad(set_to_none=True);scores=model(images.cuda());loss=F.cross_entropy(scores,labels.cuda())
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised CE')
     loss.backward()
     if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite supervised gradient')
     opt.step()
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,
       'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
       'loader_rng':generator.get_state(),'arm':arm,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'selected_train':selected},a.output/(arm+'_last.pt'))
   status['completed'].append(arm);save()
   del model,opt,loader;gc.collect();torch.cuda.empty_cache()
  status['state']='completed'
 except BaseException as exc:
  status['state']='failed';status['error']=repr(exc);raise
 finally:
  status['finished']=time.time()
  status['gpu_after']=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],text=True).strip()
  save()

if __name__=='__main__':main()

