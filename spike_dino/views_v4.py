"""Independent event multi-crop views for Spike-DINO v4.

Each view draws its own temporal window, event thinning, spatial
RandomResizedCrop, horizontal flip and polarity jitter from the *whole*
recording (DINO-style independent multi-crop), unlike the E2
``shared_core_nested_local`` policy whose two globals shared one spatial crop
(support IoU ~.965) and whose locals were nested in global view 0.

An event-support gate resamples crops that keep much less event mass than
their area share, so views stay observable without forcing high overlap.
Per-view crop/time/flip/mass metadata can be returned for audits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

from .benchmark_manifest import load_manifest
from .data import (SENSOR_SIZES, eventclr_representation, polarity_jitter,
                   read_event_record)


def crop_time(events, rng, low, high):
    """Random contiguous time window; returns (events, start_frac, ratio)."""
    times = events['t']
    if len(times) < 2 or times[-1] == times[0]:
        return events, 0., 1.
    ratio = float(rng.uniform(low, high))
    start, end = int(times[0]), int(times[-1])
    duration = end - start
    offset = float(rng.uniform(0., 1. - ratio))
    lower = start + offset * duration
    upper = lower + ratio * duration
    keep = (times >= lower) & (times <= upper)
    return {name: value[keep] for name, value in events.items()}, offset, ratio


def thin(events, rng, keep_low, keep_high):
    keep_probability = float(rng.uniform(keep_low, keep_high))
    keep = rng.random(len(events['t'])) < keep_probability
    return {name: value[keep] for name, value in events.items()}, keep_probability


class IndependentMulticropViews:
    """Callable producing [global0, global1, local0..] event-frame views."""

    def __init__(self, sensor_size, time_steps=4, render_size=320,
                 global_size=224, local_size=96, local_crops=6,
                 global_scale=(.35, 1.), local_scale=(.05, .35),
                 global_time=(.6, 1.), local_time=(.6, 1.),
                 thinning=(.8, 1.), flip_probability=.5,
                 jitter_probability=.8, jitter_strength=.3,
                 min_relative_mass=.5, attempts=8):
        self.sensor_size = sensor_size
        self.time_steps = time_steps
        self.render_size = render_size
        self.global_size = global_size
        self.local_size = local_size
        self.local_crops = local_crops
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.global_time = global_time
        self.local_time = local_time
        self.thinning = thinning
        self.flip_probability = flip_probability
        self.jitter_probability = jitter_probability
        self.jitter_strength = jitter_strength
        self.min_relative_mass = min_relative_mass
        self.attempts = attempts

    def _view(self, events, seed, size, scale, time_range):
        rng = np.random.default_rng(seed)
        generator = torch.Generator().manual_seed(seed)
        stream, time_offset, time_ratio = crop_time(events, rng, *time_range)
        stream, keep_probability = thin(stream, rng, *self.thinning)
        frame = eventclr_representation(stream, self.time_steps,
                                        self.render_size, self.sensor_size)
        mass_map = frame.sum((0, 1))
        total = float(mass_map.sum())
        best = None
        for attempt in range(self.attempts):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed * 31 + attempt)
                top, left, height, width = RandomResizedCrop.get_params(
                    frame, scale=scale, ratio=(3 / 4, 4 / 3))
            area = height * width / float(self.render_size ** 2)
            mass = (float(mass_map[top:top + height, left:left + width].sum())
                    / total if total > 0 else 0.)
            relative = mass / area if area > 0 else 0.
            candidate = (relative, attempt, top, left, height, width, mass, area)
            if best is None or relative > best[0]:
                best = candidate
            if relative >= self.min_relative_mass:
                break
        relative, attempt, top, left, height, width, mass, area = best
        view = TF.resized_crop(frame, top, left, height, width, (size, size),
                               InterpolationMode.NEAREST_EXACT)
        flipped = bool(torch.rand((), generator=generator) < self.flip_probability)
        if flipped:
            view = view.flip(-1)
        jittered = bool(torch.rand((), generator=generator) < self.jitter_probability)
        if jittered:
            view = polarity_jitter(view, generator, self.jitter_strength,
                                   self.jitter_strength).clamp(0, 1)
        metadata = {
            'box_in_render': [top, left, height, width],
            'render_size': self.render_size, 'output_size': size,
            'time_offset_fraction': time_offset, 'time_ratio': time_ratio,
            'keep_probability': keep_probability, 'flipped': flipped,
            'jittered': jittered, 'event_mass_fraction': mass,
            'area_fraction': area, 'relative_mass': relative,
            'gate_attempts': attempt + 1, 'events_in_view': int(len(stream['t'])),
        }
        return view, metadata

    def __call__(self, events, base_seed, return_metadata=False):
        views, metadata = [], []
        for index in range(2 + self.local_crops):
            is_global = index < 2
            view, meta = self._view(
                events, base_seed + 7919 * index,
                self.global_size if is_global else self.local_size,
                self.global_scale if is_global else self.local_scale,
                self.global_time if is_global else self.local_time)
            meta['role'] = 'global' if is_global else 'local'
            views.append(view)
            metadata.append(meta)
        return (views, metadata) if return_metadata else views


class MulticropEventDataset(Dataset):
    """Unlabeled manifest records -> independent multi-crop event views."""

    def __init__(self, root, manifest_path, transform, seed=42,
                 return_metadata=False):
        from pathlib import Path
        manifest = load_manifest(manifest_path)
        self.dataset = manifest['dataset']
        self.raw_root = Path(root) / 'raw'
        self.records = manifest['train']
        self.transform = transform
        self.seed = seed
        self.return_metadata = return_metadata
        if tuple(transform.sensor_size) != tuple(SENSOR_SIZES[self.dataset]):
            raise ValueError('View transform sensor size differs from dataset')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        index, epoch = item if isinstance(item, tuple) else (item, 0)
        events = read_event_record(self.raw_root / self.records[index]['path'],
                                   self.dataset)
        base = (self.seed * 1_000_003 + index * 1009 + epoch * 10_000_019) % (2 ** 31)
        result = self.transform(events, base, self.return_metadata)
        if self.return_metadata:
            views, metadata = result
            return tuple(views), metadata
        return tuple(result)


def support_iou(first, second):
    """Per-sample active-pixel IoU between two same-size view batches [B,T,C,H,W]."""
    a = first.float().abs().sum((1, 2)) > 0
    b = second.float().abs().sum((1, 2)) > 0
    intersection = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).clamp_min(1).float()
    return intersection / union
