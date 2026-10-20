"""CIFAR10-DVS downstream LP, full fine-tuning, and supervised controls.

This research runner reserves validation records inside the training split.
Official test data stay untouched until the protocol is fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from .core import Experiment, TemporalBatchNorm1d
from .data import (eventclr_representation, random_crop_time, random_roll_events,
                   read_cifar10dvs_aedat)


def cohorts(split, shot, seed, validation_per_class=100):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    labels = sorted({record['label'] for record in split['train']})
    for label in labels:
        records = [record for record in split['train'] if record['label'] == label]
        order = rng.permutation(len(records))
        if len(records) <= validation_per_class:
            raise ValueError(f'Class {label} has too few records')
        validation.extend(records[int(i)] for i in order[:validation_per_class])
        available = order[validation_per_class:]
        amount = len(available) if shot == 'full' else int(shot)
        if amount < 1 or amount > len(available):
            raise ValueError(f'Invalid shot count for class {label}')
        train.extend(records[int(i)] for i in available[:amount])
    if {record['path'] for record in train} & {record['path'] for record in validation}:
        raise RuntimeError('Labelled train and validation cohorts overlap')
    rng.shuffle(train)
    return train, validation


class LabeledEvents(Dataset):
    def __init__(self, root, records, train, seed, time_steps=4, size=48):
        self.root = Path(root) / 'raw'
        self.records = records
        self.train = train
        self.seed = seed
        self.time_steps = time_steps
        self.size = size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        events = read_cifar10dvs_aedat(self.root / record['path'])
        if self.train:
            rng = np.random.default_rng(self.seed * 1_000_003 + index * 1000 +
                                        self.epoch * 10_000_019)
            events = random_crop_time(events, rng, low=.5, high=1.)
            if rng.random() < .5:
                events = random_roll_events(events, rng, max_shift=.2)
        return eventclr_representation(events, self.time_steps, self.size), record['label']


class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, features, classes=10, spiking_head=False, frozen=False):
        super().__init__()
        self.encoder = encoder
        self.frozen = frozen
        if frozen:
            self.encoder.requires_grad_(False)
        if spiking_head:
            self.head = nn.Sequential(
                layer.Linear(features, classes), TemporalBatchNorm1d(classes),
                neuron.LIFNode(tau=2., v_threshold=1., v_reset=0., detach_reset=True,
                               surrogate_function=surrogate.ATan(), backend='torch'))
            functional.set_step_mode(self.head, 'm')
        else:
            self.head = layer.Linear(features, classes)
            functional.set_step_mode(self.head, 'm')

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                features = self.encoder.encode(x)
        else:
            features = self.encoder.encode(x)
        return self.head(features).float().mean(0)


def build_encoder(config, checkpoint, mode):
    torch.manual_seed(config['seed'])
    model = Experiment(config['method'], width=config['width'], hidden=config['hidden'],
                       bottleneck=config['bottleneck'], prototypes=config['prototypes'],
                       output_target_rate=config.get('output_target_rate'),
                       input_coding=config.get('input_coding','occupancy'),
                       population=config.get('population',1),
                       teacher_input_coding=config.get('teacher_input_coding'))
    if mode != 'supervised':
        model.load_state_dict(checkpoint['model'])
    return model.student


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for frames, labels in loader:
            x = frames.transpose(0,1).contiguous().to(device,non_blocking=True)
            try:
                logits = model(x)
                correct += int((logits.argmax(-1).cpu() == labels).sum())
                total += len(labels)
            finally:
                functional.reset_net(model)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['lp','ft','supervised'], required=True)
    parser.add_argument('--shots', default='9', help='Examples per class or full')
    parser.add_argument('--subset-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error('Invalid training configuration')
    if args.shots != 'full' and (not args.shots.isdigit() or int(args.shots) < 1):
        parser.error('shots must be a positive integer or full')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.subset_seed)
    np.random.seed(args.subset_seed)
    torch.manual_seed(args.subset_seed)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config = payload['config']
    split = json.loads((args.dataset_root / f"split_seed{config['seed']}.json").read_text(encoding='utf-8'))
    train_records, validation_records = cohorts(split,args.shots,args.subset_seed)
    if {record['path'] for record in train_records + validation_records} & {
            record['path'] for record in split['test']}:
        raise RuntimeError('Downstream cohorts overlap the official test partition')
    cohort_manifest = {'train':train_records,'validation':validation_records,
                       'mode':args.mode,'shot':args.shots,'seed':args.subset_seed}
    (args.output / 'cohorts.json').write_text(json.dumps(cohort_manifest,indent=2),encoding='utf-8')
    train_data = LabeledEvents(args.dataset_root,train_records,True,args.subset_seed,
                               config['time_steps'],config['size'])
    validation_data = LabeledEvents(args.dataset_root,validation_records,False,args.subset_seed,
                                    config['time_steps'],config['size'])
    loader_kwargs={'batch_size':args.batch_size,'num_workers':args.workers,
                   'pin_memory':args.device.startswith('cuda')}
    train_loader=DataLoader(train_data,shuffle=True,drop_last=False,
                            generator=torch.Generator().manual_seed(args.subset_seed),**loader_kwargs)
    validation_loader=DataLoader(validation_data,shuffle=False,**loader_kwargs)
    encoder=build_encoder(config,payload,args.mode)
    classifier=DownstreamClassifier(
        encoder, int(512*config['width']), spiking_head=(config['method']=='spike_dino' and
                                                          args.mode!='lp'),
        frozen=(args.mode=='lp')).to(args.device)
    optimizer=torch.optim.AdamW((p for p in classifier.parameters() if p.requires_grad),
                                 lr=1e-3, weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5)
    best=0.
    started=time.monotonic()
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        classifier.train()
        loss_total=0.
        correct=total=0
        for frames,labels in train_loader:
            x=frames.transpose(0,1).contiguous().to(args.device,non_blocking=True)
            y=labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            try:
                logits=classifier(x)
                loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite downstream loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(),3.,error_if_nonfinite=True)
                optimizer.step()
                loss_total+=float(loss.detach())*len(labels)
                correct+=int((logits.detach().argmax(-1)==y).sum())
                total+=len(labels)
            finally:
                functional.reset_net(classifier)
        scheduler.step()
        validation_accuracy=evaluate(classifier,validation_loader,args.device)
        best=max(best,validation_accuracy)
        metrics={'epoch':epoch,'train_loss':loss_total/total,
                 'train_accuracy':correct/total,'validation_accuracy':validation_accuracy,
                 'best_validation_accuracy':best}
        with (args.output/'epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(metrics)+'\n')
        if epoch==0 or (epoch+1)%10==0 or epoch+1==args.epochs:
            print(json.dumps(metrics),flush=True)
    report={'mode':args.mode,'shots':args.shots,'subset_seed':args.subset_seed,
            'train_records':len(train_records),'validation_records':len(validation_records),
            'train_classes':len({r['label'] for r in train_records}),
            'validation_classes':len({r['label'] for r in validation_records}),
            'best_validation_accuracy':best,'final_validation_accuracy':validation_accuracy,
            'elapsed_seconds':time.monotonic()-started,
            'cohort_sha256':hashlib.sha256(json.dumps(cohort_manifest,sort_keys=True).encode()).hexdigest(),
            'official_test_accessed':False,
            'protocol_note':'SpikeCLR-style LP/FT/supervised tasks on an internal validation cohort; not exact paper numbers'}
    (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
