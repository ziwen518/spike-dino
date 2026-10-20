"""Profile one representative strict Spike-DINO pretraining step on real events."""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from spike_dino.core import Experiment, independent_forward
from spike_dino.data import Cifar10DVSPilotDataset


def tick(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',required=True)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--batches',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--width',type=float,default=.125)
    parser.add_argument('--hidden',type=int,default=2048)
    parser.add_argument('--prototypes',type=int,default=128)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.batches < 3 or args.workers < 0 or args.batch_size < 2 or args.width <= 0:
        parser.error('Need at least three batches and nonnegative workers')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data=Cifar10DVSPilotDataset(args.dataset_root,seed=42,time_steps=4,size=48,
                                 augmentation='eventclr',local_crops=2)
    loader=DataLoader(data,batch_size=args.batch_size,shuffle=True,drop_last=True,
                      num_workers=args.workers,pin_memory=args.device.startswith('cuda'),
                      generator=torch.Generator().manual_seed(42))
    model=Experiment(method='spike_dino',width=args.width,hidden=args.hidden,
                     prototypes=args.prototypes,
                     input_coding='bernoulli',output_target_rate=.05).to(args.device).train()
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.student.parameters(),lr=1e-3)
    iterator=iter(loader)
    totals={key:[] for key in ('data','transfer','teacher','student','backward','optimizer','total')}
    for step in range(args.batches):
        t0=tick(args.device)
        batch=next(iterator)
        t1=tick(args.device)
        views=[view.transpose(0,1).contiguous().to(args.device,non_blocking=True)
               for view in batch]
        t2=tick(args.device)
        with torch.no_grad():
            teacher=[independent_forward(model.teacher,x)[1] for x in views[:2]]
        t3=tick(args.device)
        student=[independent_forward(model.student,x)[1] for x in views]
        t4=tick(args.device)
        loss=model.criterion(student,teacher)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t5=tick(args.device)
        optimizer.step()
        model.after_optimizer_step(teacher)
        t6=tick(args.device)
        if step:
            for key,value in zip(totals,(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t6-t0)):
                totals[key].append(value)
    result={'workers':args.workers,'batch_size':args.batch_size,'width':args.width,
            'hidden':args.hidden,'prototypes':args.prototypes,
            'batches_measured':args.batches-1,
            'mean_seconds':{key:sum(values)/len(values) for key,values in totals.items()},
            'total_seconds_per_batch':sum(totals['total'])/len(totals['total'])}
    if args.device.startswith('cuda'):
        result['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        result['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
