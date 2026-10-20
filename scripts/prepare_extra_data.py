"""Verify, safely extract and sample-read the supplementary event datasets."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np
try:
    from .download_cifar10dvs import digest,extract
except ImportError:
    from download_cifar10dvs import digest,extract


def read_5byte(path):
    payload=np.fromfile(path,dtype=np.uint8)
    if len(payload)%5:raise ValueError(f'Malformed event file {path}')
    values=payload.reshape(-1,5).astype(np.int64)
    overflow=values[:,1]==240
    timestamps=((values[:,2]&127)<<16)|(values[:,3]<<8)|values[:,4]
    timestamps+=np.cumsum(overflow)*8192
    keep=~overflow
    return {'x':values[keep,0],'y':values[keep,1],'p':values[keep,2]>>7,'t':timestamps[keep]}


def extract_tar(path,root):
    root=root.resolve()
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers()
        for member in members:
            if not (root/member.name).resolve().is_relative_to(root) or ':' in member.name or '\\' in member.name:
                raise ValueError('Unsafe tar member path')
            if not (member.isfile() or member.isdir()):
                raise ValueError('Only regular files and directories supported')
        tar.extractall(root,members=members,filter='data')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['NMNIST','DVSGesture','NCALTECH101'],required=True)
    parser.add_argument('--root',type=Path,default=Path('dataset'))
    args=parser.parse_args()
    root=args.root/args.dataset
    raw=root/'raw'
    if args.dataset!='NCALTECH101':
        key='nmnist' if args.dataset=='NMNIST' else 'dvsgesture'
        manifest=json.loads((Path(__file__).resolve().parents[1]/f'configs/{key}_download.json').read_text())
        for item in manifest['files']:
            archive=root/'archives'/item['name']
            if archive.stat().st_size!=item['size'] or digest(archive)!=item['md5']:
                raise RuntimeError(f'Archive checksum mismatch: {archive}')
            print('verified',archive.name,flush=True)
            if archive.suffix=='.zip':extract(archive,raw)
            else:extract_tar(archive,raw)
        (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report={'dataset':args.dataset,'archives_verified':True}
    if args.dataset=='DVSGesture':
        subjects={}
        for split,folder in [('train','ibmGestureTrain'),('test','ibmGestureTest')]:
            files=sorted((raw/folder).rglob('*.npy'))
            if not files:raise RuntimeError('No gesture recordings')
            subjects[split]={p.parent.name.split('_')[0] for p in files}
            labels={int(p.stem) for p in files}
            for p in files[::max(len(files)//20,1)]:
                data=np.load(p,allow_pickle=False)
                if data.ndim!=2 or data.shape[1]!=4 or not len(data) or not np.isfinite(data).all():
                    raise RuntimeError(f'Invalid gesture array: {p}')
            report[split]={'recordings':len(files),'subjects':sorted(subjects[split]),'classes':sorted(labels)}
        if subjects['train']&subjects['test']:raise RuntimeError('Subjects overlap between splits')
        report['format']='Tonic pre-cut npy; [x,y,p,t]; timestamps in milliseconds'
    else:
        folders=['Train','Test'] if args.dataset=='NMNIST' else ['.']
        for folder in folders:
            files=sorted((raw/folder).rglob('*.bin'))
            expected=60000 if folder=='Train' else 10000 if folder=='Test' else 8709
            if len(files)!=expected:raise RuntimeError(f'{folder}: expected {expected}, found {len(files)}')
            checked=0
            for p in files[::max(len(files)//20,1)]:
                data=read_5byte(p)
                if not len(data['t']) or np.any(np.diff(data['t'])<0):
                    raise RuntimeError(f'Invalid event stream {p}')
                checked+=1
            report[folder]={'recordings':len(files),'classes':len({p.parent.name for p in files}),
                            'sampled_reads':checked}
        report['format']='original 5-byte events; timestamp overflow markers handled'
    (root/'read_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
