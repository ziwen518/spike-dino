"""Frozen-feature diagnostic on disjoint subsets of the CIFAR10-DVS training split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional

from .core import Experiment
from .data import Cifar10DVSPilotDataset, eventclr_representation, read_cifar10dvs_aedat


class FixedEventDataset(Dataset):
    def __init__(self, root, records, time_steps=4, size=48):
        self.root = Path(root) / 'raw'
        self.records = records
        self.time_steps = time_steps
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        events = read_cifar10dvs_aedat(self.root / record['path'])
        return eventclr_representation(events, self.time_steps, self.size), record['label']


@torch.no_grad()
def extract_features(model, loader, device):
    features, labels = [], []
    model.eval()
    for frames, targets in loader:
        x = frames.transpose(0, 1).contiguous().to(device)
        functional.reset_net(model)
        try:
            sequence = model.encode(x)
            features.append(sequence.float().mean(0).cpu())
            labels.append(targets.clone())
        finally:
            functional.reset_net(model)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def weighted_knn(train_features, train_labels, test_features, k=20, temperature=.07, classes=10):
    train = F.normalize(train_features.float(), dim=-1)
    test = F.normalize(test_features.float(), dim=-1)
    values, indices = (test @ train.T).topk(min(k, len(train)), dim=1)
    neighbours = train_labels[indices]
    weights = (values / temperature).softmax(dim=1)
    votes = weights.new_zeros(len(test), classes)
    votes.scatter_add_(1, neighbours, weights)
    return votes.argmax(dim=1)


def linear_probe(train_features, train_labels, test_features, seed=42, epochs=100):
    # Fixed hyperparameters. This is a diagnostic, not a tuned paper protocol.
    torch.manual_seed(seed)
    head = nn.Linear(train_features.shape[1], 10)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_features), train_labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return head(test_features).argmax(dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output file already exists')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    if config['method'] not in ('spike_dino', 'dino', 'spikeclr', 'temporal_ntxent'):
        parser.error('Unsupported checkpoint method')
    torch.manual_seed(config['seed'])
    model = Experiment(config['method'], width=config['width'], hidden=config['hidden'],
                       bottleneck=config['bottleneck'], prototypes=config['prototypes'],
                       output_target_rate=config.get('output_target_rate'),
                       input_coding=config.get('input_coding','occupancy'),
                       population=config.get('population',1),
                       teacher_input_coding=config.get('teacher_input_coding'))
    model.load_state_dict(payload['model'])
    encoder = model.student.to(args.device)
    split = json.loads((args.dataset_root / f"split_seed{config['seed']}.json").read_text(encoding='utf-8'))
    selected = Cifar10DVSPilotDataset(args.dataset_root, seed=config['seed'],
                                      limit=config.get('limit', 512)).records
    train_paths = {record['path'] for record in selected}
    test_paths = {record['path'] for record in split['test']}
    if train_paths & test_paths:
        raise RuntimeError('Training and test records overlap')
    # Reserve the official test partition. Select a balanced validation
    # cohort from the remainder of the training partition.
    remaining = [record for record in split['train'] if record['path'] not in train_paths]
    rng = np.random.default_rng(config['seed'] + 17)
    validation = []
    for label in range(10):
        candidates = [record for record in remaining if record['label'] == label]
        if len(candidates) < 100:
            raise RuntimeError(f'Not enough unselected training records for class {label}')
        indices = rng.choice(len(candidates), size=100, replace=False)
        validation.extend(candidates[int(index)] for index in indices)
    train_loader = DataLoader(FixedEventDataset(args.dataset_root, selected,
                                               config['time_steps'], config['size']),
                              batch_size=16, shuffle=False, num_workers=0)
    validation_loader = DataLoader(FixedEventDataset(args.dataset_root, validation,
                                              config['time_steps'], config['size']),
                             batch_size=16, shuffle=False, num_workers=0)
    train_features, train_labels = extract_features(encoder, train_loader, args.device)
    validation_features, validation_labels = extract_features(encoder, validation_loader, args.device)
    knn_prediction = weighted_knn(train_features, train_labels, validation_features)
    linear_prediction = linear_probe(train_features, train_labels, validation_features)
    report = {'method':config['method'], 'input_coding':config.get('input_coding','occupancy'),
              'output_target_rate':config.get('output_target_rate'),
              'checkpoint':str(args.checkpoint), 'train_samples':len(selected),
              'pretrain_records':config.get('train_records',config.get('limit')),
              'pretrain_updates':(payload['step'] if 'step' in payload else
                                  (payload['epoch']+1)*config['steps_per_epoch']),
              'validation_samples':len(validation), 'validation_source':'unused records from training partition',
              'official_test_accessed':False,
              'train_classes':len(set(train_labels.tolist())),
              'validation_classes':len(set(validation_labels.tolist())),
              'knn_top1':float((knn_prediction == validation_labels).float().mean()),
              'linear_top1':float((linear_prediction == validation_labels).float().mean()),
              'purpose':'fixed-protocol frozen-feature diagnostic; official test partition untouched'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
