"""Event-aware view family for Spike-DINO v5: 2 global + 2 temporal views.

Why this replaces the v4 spatial locals
---------------------------------------
v4 followed DINO's RGB recipe (2 global + 6 local). For SEW-ResNet-18 at
128 px a 64 px local crop reaches layer 4 with a 2x2 feature map, and the six
locals carry 12 of the 14 equally-weighted teacher-student terms (85.7%), so
most of the objective supervised a spatially degenerate target. Events are not
RGB: their distinctive structure is *when* the pixels fire, and a recording is
a complete time series that can be sampled in time without shrinking space.

v5 therefore keeps the two global views, and replaces the six spatial locals
with two *temporal* views: the full spatial extent over a shorter time window,
selected so it keeps at least ``mass_min`` of the recording's events. The
student task becomes "recover the teacher's full-recording representation from
an incomplete temporal observation", which is only possible if the backbone
encodes event timing rather than a static frame statistic.

Every view still renders through ``eventclr_representation`` at the global
resolution, so the spatial receptive field is identical for all four views and
the only difference is the time window.
"""

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .data import eventclr_representation, polarity_jitter
from .views_v4 import thin


def temporal_window(events, rng, ratio_range, mass_min, attempts=8):
    """Contiguous time window holding >= ``mass_min`` of the events.

    ``events['t']`` must be sorted (DVS recordings are). Returns the kept
    events plus (offset fraction, ratio, event-mass fraction). If no candidate
    reaches ``mass_min`` the best of ``attempts`` samples is used, so a view is
    never empty.
    """
    times = events['t']
    if len(times) < 2 or times[-1] == times[0]:
        return events, 0., 1., 1.
    start, end = float(times[0]), float(times[-1])
    duration = end - start
    total = max(int(len(times)), 1)
    best = None
    for attempt in range(attempts):
        ratio = float(rng.uniform(*ratio_range))
        if attempt % 2:
            # Anchor on a random event so a narrow window can never be empty.
            lower = float(times[int(rng.integers(0, len(times)))])
        else:
            lower = start + float(rng.uniform(0., 1. - ratio)) * duration
        upper = min(lower + ratio * duration, end)
        low = int(np.searchsorted(times, lower, 'left'))
        high = int(np.searchsorted(times, upper, 'right'))
        mass = (high - low) / total
        candidate = (mass, low, high, (lower - start) / duration, ratio)
        if best is None or mass > best[0]:
            best = candidate
        if mass >= mass_min:
            break
    mass, low, high, offset, ratio = best
    if high <= low:  # unreachable: an anchored attempt always keeps its anchor
        raise RuntimeError('Temporal window selection produced an empty view')
    return ({name: value[low:high] for name, value in events.items()},
            offset, ratio, mass)


class TemporalDinoViews:
    """Callable producing [global0, global1, temporal0, temporal1].

    Globals: independent time window ``global_time``, spatial RRC
    ``global_scale``, flip, thinning, polarity jitter.
    Temporal: independent time window ``temporal_time`` gated on event mass,
    spatial RRC ``temporal_scale`` (default 1.0 = full frame), same output
    size as the globals, flip, thinning, polarity jitter.
    """

    def __init__(self, sensor_size, time_steps=4, render_size=128,
                 global_size=128, global_scale=(.35, 1.),
                 global_time=(.6, 1.), temporal_time=(.25, .6),
                 temporal_scale=(1., 1.), temporal_mass_min=.25,
                 temporal_crops=2, thinning=(.8, 1.), flip_probability=.5,
                 jitter_probability=.8, jitter_strength=.3,
                 min_relative_mass=.5, attempts=8):
        if temporal_crops < 1:
            raise ValueError('v5 needs at least one temporal view')
        self.sensor_size = sensor_size
        self.time_steps = time_steps
        self.render_size = render_size
        self.global_size = global_size
        self.global_scale = tuple(global_scale)
        self.global_time = tuple(global_time)
        self.temporal_time = tuple(temporal_time)
        self.temporal_scale = tuple(temporal_scale)
        self.temporal_mass_min = float(temporal_mass_min)
        self.temporal_crops = int(temporal_crops)
        self.thinning = thinning
        self.flip_probability = flip_probability
        self.jitter_probability = jitter_probability
        self.jitter_strength = jitter_strength
        self.min_relative_mass = min_relative_mass
        self.attempts = attempts

    def _spatial_crop(self, frame, rng_seed, scale):
        from torchvision.transforms import RandomResizedCrop
        mass_map = frame.sum((0, 1))
        total = float(mass_map.sum())
        if scale[0] >= 1. and scale[1] >= 1.:
            return 0, 0, int(self.render_size), int(self.render_size), 1., 1.
        best = None
        for attempt in range(self.attempts):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(rng_seed * 31 + attempt)
                top, left, height, width = RandomResizedCrop.get_params(
                    frame, scale=scale, ratio=(3 / 4, 4 / 3))
            area = height * width / float(self.render_size ** 2)
            mass = (float(mass_map[top:top + height, left:left + width].sum())
                    / total if total > 0 else 0.)
            relative = mass / area if area > 0 else 0.
            if best is None or relative > best[0]:
                best = (relative, attempt, top, left, height, width, mass, area)
            if relative >= self.min_relative_mass:
                break
        relative, attempt, top, left, height, width, mass, area = best
        return top, left, height, width, mass, area

    def _view(self, events, seed, is_global):
        rng = np.random.default_rng(seed)
        generator = torch.Generator().manual_seed(seed)
        if is_global:
            ratio = float(rng.uniform(*self.global_time))
            times = events['t']
            if len(times) >= 2 and times[-1] != times[0]:
                start, end = float(times[0]), float(times[-1])
                duration = end - start
                offset = float(rng.uniform(0., 1. - ratio))
                lower = start + offset * duration
                stream = {name: value[(times >= lower)
                                      & (times <= lower + ratio * duration)]
                          for name, value in events.items()}
                mass_fraction = float(len(stream['t']) / max(len(times), 1))
            else:
                stream, offset, mass_fraction = events, 0., 1.
            scale = self.global_scale
        else:
            stream, offset, ratio, mass_fraction = temporal_window(
                events, rng, self.temporal_time, self.temporal_mass_min,
                self.attempts)
            scale = self.temporal_scale
        stream, keep_probability = thin(stream, rng, *self.thinning)
        frame = eventclr_representation(stream, self.time_steps,
                                        self.render_size, self.sensor_size)
        top, left, height, width, mass, area = self._spatial_crop(
            frame, seed, scale)
        view = TF.resized_crop(frame, top, left, height, width,
                               (self.global_size, self.global_size),
                               InterpolationMode.NEAREST_EXACT)
        flipped = bool(torch.rand((), generator=generator) < self.flip_probability)
        if flipped:
            view = view.flip(-1)
        jittered = bool(torch.rand((), generator=generator) < self.jitter_probability)
        if jittered:
            view = polarity_jitter(view, generator, self.jitter_strength,
                                   self.jitter_strength).clamp(0, 1)
        metadata = {'role': 'global' if is_global else 'temporal',
                    'box_in_render': [top, left, height, width],
                    'render_size': self.render_size,
                    'output_size': self.global_size,
                    'time_offset_fraction': offset, 'time_ratio': ratio,
                    'event_mass_fraction': mass_fraction,
                    'keep_probability': keep_probability,
                    'spatial_mass_fraction': mass, 'area_fraction': area,
                    'flipped': flipped, 'jittered': jittered,
                    'events_in_view': int(len(stream['t']))}
        return view, metadata

    def __call__(self, events, base_seed, return_metadata=False):
        views, metadata = [], []
        for index in range(2 + self.temporal_crops):
            view, meta = self._view(events, base_seed + 7919 * index, index < 2)
            views.append(view)
            metadata.append(meta)
        return (views, metadata) if return_metadata else views
