"""Validate original AEDAT recordings and prepare a small, training-only smoke NPZ.

Not the benchmark augmentation pipeline. Official class counts required.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spike_dino.data import bin_cifar10dvs_events, read_cifar10dvs_aedat

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Compatibility aliases retained for existing scripts and tests.
read_aedat = read_cifar10dvs_aedat
bin_events = bin_cifar10dvs_events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('dataset/CIFAR10DVS'))
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    raw=args.root/'raw'
    rng=np.random.default_rng(args.seed)
    manifest={'seed':args.seed,'source':'original AEDAT; spikingjelly CIFAR10DVS.load_origin_data',
              'protocol':'stratified 900 train / 100 test per class; not recovered author split',
              'train':[],'test':[]}
    selected=[]
    for label,name in enumerate(CLASSES):
        paths=sorted((raw/name).rglob('*.aedat'))
        if len(paths)!=1000:
            raise ValueError(f'{name}: expected 1000 files, got {len(paths)}')
        indices=rng.permutation(len(paths))
        for split,ids in [('train',indices[:900]),('test',indices[900:])]:
            manifest[split].extend({'path':p.relative_to(raw).as_posix(),'label':label}
                                   for p in (paths[i] for i in ids))
        selected.extend(paths[i] for i in indices[:2])
    manifest_path=args.root/f'split_seed{args.seed}.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
            raise RuntimeError('Existing split differs; refusing overwrite')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    frames=[]
    stats=[]
    for path in selected:
        events=read_aedat(path)
        x=bin_events(events).numpy()
        frames.append(x)
        stats.append({'path':path.relative_to(raw).as_posix(),'events':len(events['t']),
                      'nonzero_fraction':float(np.count_nonzero(x)/x.size)})
    data=np.stack(frames)
    np.savez_compressed(args.root/'train_smoke_views.npz',view0=data,view1=np.roll(data,1,axis=-1))
    report={'recordings':10000,'train':9000,'test':1000,'smoke_shape':list(data.shape),
            'reader':'project AEDAT2 frombuffer parser; SpikingJelly-compatible orientation/polarity',
            'purpose':'input validation only; shifted smoke views are not SpikeCLR augmentations','samples':stats}
    (args.root/'read_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)


if __name__=='__main__':
    main()
