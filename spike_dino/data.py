"""Event-data utilities used by the reproducible CIFAR10-DVS pilot."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomHorizontalFlip, RandomResizedCrop
from torchvision.transforms import functional as TF

from .benchmark_manifest import load_manifest


def read_cifar10dvs_aedat(path):
    """Read AEDAT2 with the coordinate/polarity convention used by SpikingJelly."""
    with Path(path).open('rb') as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line.startswith(b'#'):
                stream.seek(position)
                break
        payload = stream.read()
    if len(payload) % 8:
        raise ValueError(f'AEDAT2 payload does not contain address/timestamp pairs: {path}')
    words = np.frombuffer(payload, dtype='>u4')
    address = words[::2]
    x = ((address & 0xFE) >> 1).astype(np.int64)
    y = ((address & 0x7F00) >> 8).astype(np.int64)
    return {'t': words[1::2].astype(np.int64), 'x': 127 - y, 'y': 127 - x,
            'p': 1 - (address & 1).astype(np.int64)}


def bin_cifar10dvs_events(events, time_steps=4, size=48):
    if time_steps < 1 or size < 1:
        raise ValueError('time_steps and size must be positive')
    frames = np.zeros((time_steps, 2, 128, 128), dtype=np.float32)
    if len(events['t']):
        t = np.asarray(events['t'], dtype=np.int64)
        if np.any(np.diff(t) < 0):
            raise ValueError('Event timestamps must be sorted')
        x, y, p = (np.asarray(events[key], dtype=np.int64) for key in ('x', 'y', 'p'))
        if np.any((x < 0) | (x >= 128) | (y < 0) | (y >= 128) | (p < 0) | (p > 1)):
            raise ValueError('Invalid CIFAR10-DVS event coordinate or polarity')
        bins = np.minimum((t - t[0]) * time_steps // max(int(t[-1] - t[0]), 1), time_steps - 1)
        flat = ((bins * 2 + p) * 128 + y) * 128 + x
        frames = np.bincount(flat, minlength=frames.size).reshape(frames.shape).astype(np.float32)
        lower, upper = frames.min(), frames.max()
        if upper > lower:
            frames = (frames - lower) / (upper - lower)
    return F.interpolate(torch.from_numpy(frames), size=(size, size), mode='nearest')


def eventclr_representation(events, time_steps=4, size=48, sensor_size=(128, 128)):
    """Match the EventCLR frame path: counts, per-sample min-max, adaptive max pool."""
    width, height = sensor_size
    frames = np.zeros((time_steps, 2, height, width), dtype=np.float32)
    if len(events['t']):
        t = np.asarray(events['t'], dtype=np.int64)
        if np.any(np.diff(t) < 0):
            raise ValueError('Event timestamps must be sorted')
        x, y, p = (np.asarray(events[key], dtype=np.int64) for key in ('x', 'y', 'p'))
        if np.any((x < 0) | (x >= width) | (y < 0) | (y >= height) |
                  (p < 0) | (p > 1)):
            raise ValueError('Event coordinate or polarity exceeds sensor size')
        bins = np.minimum((t - t[0]) * time_steps // max(int(t[-1] - t[0]), 1), time_steps - 1)
        flat = ((bins * 2 + p) * height + y) * width + x
        frames = np.bincount(flat, minlength=frames.size).reshape(frames.shape).astype(np.float32)
        lower, upper = frames.min(), frames.max()
        if upper > lower:
            frames = (frames - lower) / (upper - lower)
    return F.adaptive_max_pool2d(torch.from_numpy(frames), (size, size))


def copy_events(events):
    return {name: np.asarray(value).copy() for name, value in events.items()}


def thin_events(events, rng, keep_probability=.9):
    """Independent event thinning without changing time or geometry."""
    if not 0 < keep_probability <= 1:
        raise ValueError('keep_probability must lie in (0, 1]')
    keep = rng.random(len(events['t'])) < keep_probability
    return {name: np.asarray(value)[keep] for name, value in events.items()}


def random_crop_time(events, rng, low=.1, high=1.0):
    if not 0 < low <= high <= 1:
        raise ValueError('Temporal crop range must lie inside (0, 1]')
    if not len(events['t']):
        return events
    ratio = rng.uniform(low, high)
    start, end = events['t'][0], events['t'][-1]
    duration = end - start
    crop_duration = duration * ratio
    crop_start = rng.uniform(start, max(start, end - crop_duration))
    keep = (events['t'] >= crop_start) & (events['t'] < crop_start + crop_duration)
    return {name: value[keep] for name, value in events.items()}


def random_roll_events(events, rng, max_shift=.5, sensor_size=(128, 128)):
    width, height = sensor_size
    shift_x = rng.integers(-int(max_shift * width), int(max_shift * width) + 1)
    shift_y = rng.integers(-int(max_shift * height), int(max_shift * height) + 1)
    events['x'] = (events['x'] + shift_x) % width
    events['y'] = (events['y'] + shift_y) % height
    return events


def polarity_jitter(frames, generator, contrast=.5, brightness=.5):
    contrast_factor = torch.empty((), dtype=frames.dtype).uniform_(1 - contrast, 1 + contrast, generator=generator)
    brightness_factor = torch.empty((), dtype=frames.dtype).uniform_(1 - brightness, 1 + brightness, generator=generator)
    def adjust_contrast(x):
        mean = x.mean(dim=1, keepdim=True)
        return (x - mean) * contrast_factor + mean
    def adjust_brightness(x):
        return x * brightness_factor
    return (adjust_brightness(adjust_contrast(frames)) if torch.rand((), generator=generator) < .5
            else adjust_contrast(adjust_brightness(frames)))


class EventCLRViewTransform:
    """EventCLR strong policy with independently copied event streams.

    Parameters follow the upstream public EventCLR commit: temporal crop 0.1-1,
    rolling p=0.5/max shift 0.5, RRC scale 0.1-1, flip p=0.5, jitter p=0.8,
    and polarity average p=0.2.
    """
    def __init__(self, time_steps=4, size=48, families=('temporal', 'spatial', 'polarity'),
                 crop_scale=(.1, 1.0), temporal_crop_range=(.1, 1.0),
                 sensor_size=(128, 128)):
        self.time_steps = time_steps
        self.size = size
        self.families = frozenset(families)
        self.temporal_crop_range = temporal_crop_range
        self.sensor_size = sensor_size
        self.crop = RandomResizedCrop((size, size), scale=crop_scale,
                                      interpolation=InterpolationMode.NEAREST_EXACT)
        self.flip = RandomHorizontalFlip(.5)

    def __call__(self, events, rng, generator):
        events = copy_events(events)
        if 'temporal' in self.families:
            events = random_crop_time(events, rng, *self.temporal_crop_range)
        if ('spatial' in self.families or 'roll' in self.families) and rng.random() < .5:
            events = random_roll_events(events, rng, sensor_size=self.sensor_size)
        frames = eventclr_representation(events, self.time_steps, self.size,
                                         self.sensor_size)
        if any(name in self.families for name in ('spatial', 'crop', 'flip')):
            # torchvision transforms use the process-global CPU RNG. Fork it
            # so views stay identical when DataLoader worker count changes.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(generator.get_state())
                if 'spatial' in self.families or 'crop' in self.families:
                    frames = self.crop(frames)
                if 'spatial' in self.families or 'flip' in self.families:
                    frames = self.flip(frames)
                generator.set_state(torch.get_rng_state())
        if 'polarity' in self.families:
            if torch.rand((), generator=generator) < .8:
                frames = polarity_jitter(frames, generator)
            if torch.rand((), generator=generator) < .2:
                frames = frames.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        return frames


def pilot_view_augmentation(frames, generator):
    """Weak frame-space two-view transform used only for pipeline pilots.

    This is deliberately separate from the event-level SpikeCLR augmentation policy.
    """
    x = frames.clone()
    if torch.rand((), generator=generator) < .5:
        x = x.flip(-1)
    dy, dx = torch.randint(-4, 5, (2,), generator=generator).tolist()
    x = x.roll((dy, dx), dims=(-2, -1))
    scale = torch.empty((), dtype=x.dtype).uniform_(.8, 1.2, generator=generator)
    keep = torch.rand(x.shape, generator=generator, dtype=x.dtype) > .05
    return (x * keep * scale).clamp_(0, 1)


class Cifar10DVSPilotDataset(Dataset):
    def __init__(self, root, split='train', seed=42, time_steps=4, size=48, limit=None,
                 augmentation='pilot', local_crops=0, local_size=None,
                 global_crop_scale=(.1, 1.0), local_crop_scale=(.05, .4),
                 global_temporal_crop_range=(.1, 1.0),
                 local_temporal_crop_range=(.1, 1.0), manifest_path=None):
        root = Path(root)
        split_data = (load_manifest(manifest_path, 'cifar10dvs')
                      if manifest_path is not None else
                      json.loads((root / f'split_seed{seed}.json').read_text(
                          encoding='utf-8')))
        if split not in split_data:
            raise ValueError(f'Unknown split: {split}')
        records = split_data[split]
        if limit is not None:
            if limit < 2:
                raise ValueError('limit must be at least 2')
            labels = sorted({record['label'] for record in records})
            if limit < len(labels):
                raise ValueError('limit must include at least one sample from each class')
            rng = np.random.default_rng(seed)
            selected = []
            for label_index, label in enumerate(labels):
                class_records = [record for record in records if record['label'] == label]
                quota = limit // len(labels) + int(label_index < limit % len(labels))
                if len(class_records) < quota:
                    raise ValueError(f'Class {label} has fewer than {quota} samples')
                indices = rng.choice(len(class_records), size=quota, replace=False)
                selected.extend(class_records[int(index)] for index in indices)
            rng.shuffle(selected)
            records = selected
        self.raw_root = root / 'raw'
        self.records = records
        self.seed = seed
        self.epoch = 0
        self.time_steps = time_steps
        self.size = size
        if augmentation not in {'pilot', 'eventclr'}:
            raise ValueError(f'Unknown augmentation: {augmentation}')
        self.augmentation = augmentation
        if local_crops < 0 or local_crops > 8 or (local_crops and augmentation != 'eventclr'):
            raise ValueError('Local crops require the EventCLR view policy')
        self.local_crops = local_crops
        self.eventclr = (EventCLRViewTransform(
            time_steps, size, crop_scale=global_crop_scale,
            temporal_crop_range=global_temporal_crop_range)
            if augmentation == 'eventclr' else None)
        self.local_transform = (EventCLRViewTransform(
            time_steps, local_size if local_size is not None else size // 2,
            crop_scale=local_crop_scale,
            temporal_crop_range=local_temporal_crop_range)
            if local_crops else None)

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = epoch

    def view_seed(self, index, view_index, epoch=None):
        if epoch is None:
            epoch = self.epoch
        return self.seed * 1_000_003 + index * 1000 + epoch * 10_000_019 + view_index

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, epoch = index
        else:
            epoch = self.epoch
        record = self.records[index]
        events = read_cifar10dvs_aedat(self.raw_root / record['path'])
        seed_base = self.view_seed(index, 0, epoch)
        first_seed, second_seed = seed_base, seed_base + 1
        first, second = torch.Generator().manual_seed(first_seed), torch.Generator().manual_seed(second_seed)
        if self.augmentation == 'pilot':
            frames = bin_cifar10dvs_events(events, self.time_steps, self.size)
            return pilot_view_augmentation(frames, first), pilot_view_augmentation(frames, second)
        views = [self.eventclr(events, np.random.default_rng(first_seed), first),
                 self.eventclr(events, np.random.default_rng(second_seed), second)]
        for crop_index in range(self.local_crops):
            local_seed = seed_base + crop_index + 2
            views.append(self.local_transform(events, np.random.default_rng(local_seed),
                                              torch.Generator().manual_seed(local_seed)))
        return tuple(views)


SENSOR_SIZES = {'cifar10dvs': (128, 128), 'ncaltech101': (240, 180),
                'nmnist': (34, 34), 'dvsgesture': (128, 128)}


def read_5byte_events(path):
    """Read original N-MNIST/N-Caltech events with overflow markers."""

    payload = np.fromfile(path, dtype=np.uint8)
    if len(payload) % 5:
        raise ValueError(f'Malformed 5-byte event file: {path}')
    data = payload.reshape(-1, 5).astype(np.int64)
    overflow = data[:, 1] == 240
    timestamps = ((data[:, 2] & 127) << 16) | (data[:, 3] << 8) | data[:, 4]
    timestamps += np.cumsum(overflow) * 8192
    keep = ~overflow
    return {'x': data[keep, 0], 'y': data[keep, 1],
            'p': data[keep, 2] >> 7, 't': timestamps[keep]}


def read_gesture_events(path):
    """Tonic pre-cut [x,y,p,t] NPY; stored t is milliseconds."""

    data = np.load(path, allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f'Malformed DVS-Gesture event file: {path}')
    return {'x': data[:, 0].astype(np.int64),
            'y': data[:, 1].astype(np.int64),
            'p': data[:, 2].astype(np.int64),
            't': data[:, 3].astype(np.int64) * 1000}


def read_event_record(path, dataset):
    if dataset == 'cifar10dvs':
        return read_cifar10dvs_aedat(path)
    if dataset in ('ncaltech101', 'nmnist'):
        return read_5byte_events(path)
    if dataset == 'dvsgesture':
        return read_gesture_events(path)
    raise ValueError(dataset)


class ManifestEventDataset(Dataset):
    """Shared fixed-record training loader for the four SpikeCLR event datasets."""

    def __init__(self, root, manifest_path, split='train', seed=42,
                 time_steps=4, size=128, local_crops=8, local_size=None,
                 global_crop_scale=(.32, 1.0), local_crop_scale=(.05, .32),
                 global_temporal_crop_range=(.5, 1.0),
                 local_temporal_crop_range=(.1, 1.0),
                 view_validity='none', max_view_attempts=4,
                 global_view_policy='independent'):
        manifest = load_manifest(manifest_path)
        if split not in ('train', 'test'):
            raise ValueError(split)
        self.dataset = manifest['dataset']
        self.raw_root = Path(root) / 'raw'
        self.records = manifest[split]
        self.seed = seed
        self.epoch = 0
        self.local_crops = local_crops
        if global_view_policy not in ('independent', 'shared_core',
                                      'shared_temporal_windows',
                                      'shared_time_independent_space',
                                      'shared_time_shifted_space',
                                      'shared_core_nested_local'):
            raise ValueError('Unknown global view policy')
        self.global_view_policy = global_view_policy
        if view_validity not in ('none', 'resample_empty') or max_view_attempts < 1:
            raise ValueError('Invalid event-view validity policy')
        self.view_validity = view_validity
        self.max_view_attempts = max_view_attempts
        sensor = SENSOR_SIZES[self.dataset]
        self.sensor_size = sensor
        self.time_steps = time_steps
        self.size = size
        self.global_crop_scale = global_crop_scale
        self.global_temporal_crop_range = global_temporal_crop_range
        self.global_transform = EventCLRViewTransform(
            time_steps, size, crop_scale=global_crop_scale,
            temporal_crop_range=global_temporal_crop_range, sensor_size=sensor)
        self.local_transform = EventCLRViewTransform(
            time_steps, local_size if local_size is not None else size // 2,
            crop_scale=local_crop_scale,
            temporal_crop_range=local_temporal_crop_range, sensor_size=sensor)

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch = epoch

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, epoch = index
        else:
            epoch = self.epoch
        record = self.records[index]
        events = read_event_record(self.raw_root / record['path'], self.dataset)
        base = self.seed * 1_000_003 + index * 1000 + epoch * 10_000_019
        views = (self._shared_core_views(events, base) if
                 self.global_view_policy in ('shared_core',
                                             'shared_temporal_windows',
                                             'shared_time_independent_space',
                                             'shared_time_shifted_space',
                                             'shared_core_nested_local') else [])
        for crop_index in range(len(views), 2 + self.local_crops):
            transform = (self.global_transform if crop_index < 2
                         else self.local_transform)
            attempts = (self.max_view_attempts if self.view_validity == 'resample_empty'
                        else 1)
            for attempt in range(attempts):
                view_seed = base + crop_index + attempt * 1_000_003
                view = transform(events, np.random.default_rng(view_seed),
                                 torch.Generator().manual_seed(view_seed))
                if view.any():
                    break
            views.append(view)
        return tuple(views)

    def _shared_core_views(self, events, base):
        """Two thinned observations of one event-space temporal/spatial core."""
        core = random_crop_time(events, np.random.default_rng(base + 71),
                                *self.global_temporal_crop_range)
        rendered = []
        for view_index in range(2):
            seed = base + view_index
            # Two windows of at least 75% of the shared core have at least
            # 50% temporal overlap, preserving the DVS positive's content
            # while making timing genuinely different between global views.
            stream = (random_crop_time(
                core, np.random.default_rng(base + 81 + view_index), .75, 1.)
                if self.global_view_policy == 'shared_temporal_windows'
                else core)
            thinned = thin_events(stream, np.random.default_rng(seed))
            rendered.append(eventclr_representation(
                thinned, self.time_steps, self.size, self.sensor_size))
        crops = []
        first_crop = None
        for view_index, view in enumerate(rendered):
            if (view_index == 1 and self.global_view_policy ==
                    'shared_time_shifted_space'):
                top, left, height, width = self._shift_global_crop(
                    first_crop, view.shape[-2:], base + 74)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(base + 73 +
                                      (view_index if self.global_view_policy ==
                                       'shared_time_independent_space' else 0))
                    top, left, height, width = RandomResizedCrop.get_params(
                        view, scale=self.global_crop_scale,
                        ratio=(3 / 4, 4 / 3))
                if view_index == 0:
                    first_crop = top, left, height, width
            crops.append(TF.resized_crop(
                view, top, left, height, width, (self.size, self.size),
                InterpolationMode.NEAREST_EXACT))
        if self.global_view_policy == 'shared_core_nested_local':
            top, left, height, width = first_crop
            local_size = self.local_transform.size
            for local_index in range(self.local_crops):
                seed = base + 201 + local_index
                # Keep at least 75% of the same temporal core used by both
                # teacher global views, then independently thin the events.
                stream = random_crop_time(
                    core, np.random.default_rng(seed + 17), .75, 1.)
                stream = thin_events(stream, np.random.default_rng(seed))
                local_frame = eventclr_representation(
                    stream, self.time_steps, self.size, self.sensor_size)
                parent = local_frame[..., top:top + height, left:left + width]
                attempts = (self.max_view_attempts
                            if self.view_validity == 'resample_empty' else 1)
                for attempt in range(attempts):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed + attempt * 1_000_003)
                        local_top, local_left, local_height, local_width = (
                            RandomResizedCrop.get_params(
                                parent, scale=self.local_transform.crop.scale,
                                ratio=(3 / 4, 4 / 3)))
                    view = TF.resized_crop(
                        parent, local_top, local_left, local_height,
                        local_width, (local_size, local_size),
                        InterpolationMode.NEAREST_EXACT)
                    if view.any():
                        break
                crops.append(view)
        return crops

    @staticmethod
    def _shift_global_crop(crop, frame_size, seed):
        """Displace one axis by 15% while preserving crop size and >=85% area."""
        top, left, height, width = crop
        frame_height, frame_width = frame_size
        rng = np.random.default_rng(seed)
        axes = [0, 1]
        rng.shuffle(axes)
        for axis in axes:
            origin = top if axis == 0 else left
            crop_length = height if axis == 0 else width
            maximum = ((frame_height - height) if axis == 0
                       else (frame_width - width))
            if maximum <= 0:
                continue
            distance = max(1, int(.15 * crop_length))
            signs = [1, -1]
            rng.shuffle(signs)
            for sign in signs:
                shifted = min(max(origin + sign * distance, 0), maximum)
                if shifted != origin:
                    return ((shifted, left, height, width) if axis == 0
                            else (top, shifted, height, width))
        return crop
