import numpy as np
import torch
from spike_dino.data import (EventCLRViewTransform, bin_cifar10dvs_events as bin_events,
                             read_cifar10dvs_aedat as read_aedat, pilot_view_augmentation,
                             Cifar10DVSPilotDataset)


def test_temporal_bins_include_final_timestamp_and_keep_polarities():
    events={'t':np.array([0,1,2,3]),'x':np.array([0,0,0,127]),
            'y':np.array([0,0,0,127]),'p':np.array([0,1,0,1])}
    result=bin_events(events,time_steps=2,size=128)
    assert result[0,0,0,0] == 1
    assert result[0,1,0,0] == 1
    assert result[1,0,0,0] == 1
    assert result[1,1,127,127] == 1
    assert torch.count_nonzero(result) == 4


def test_empty_event_recording():
    events={k:np.array([],dtype=np.int64) for k in ['t','x','y','p']}
    result=bin_events(events)
    assert result.shape==(4,2,48,48)
    assert torch.isfinite(result).all() and not result.any()


def test_aedat_endianness_coordinates_polarity(tmp_path):
    path=tmp_path/'sample.aedat'
    words=np.array([(7<<8)|(5<<1)|1,12345,(127<<8)|(0<<1)|0,12346],dtype='>u4')
    path.write_bytes(b'#!AER-DAT2.0\r\n# synthetic fixture\r\n'+words.tobytes())
    events=read_aedat(path)
    np.testing.assert_array_equal(events['t'],[12345,12346])
    np.testing.assert_array_equal(events['x'],[120,0])
    np.testing.assert_array_equal(events['y'],[122,127])
    np.testing.assert_array_equal(events['p'],[0,1])


def test_nmnist_timestamp_overflow_marker(tmp_path):
    from scripts.prepare_extra_data import read_5byte
    path=tmp_path/'sample.bin'
    records=np.array([[3,4,0,31,164],[0,240,0,0,0],[5,6,128,0,100]],dtype=np.uint8)
    path.write_bytes(records.tobytes())
    events=read_5byte(path)
    np.testing.assert_array_equal(events['t'],[8100,8292])
    np.testing.assert_array_equal(events['x'],[3,5])
    np.testing.assert_array_equal(events['p'],[0,1])


def test_pilot_augmentation_is_two_view_safe_and_bounded():
    frames = torch.ones(4, 2, 16, 16)
    first = pilot_view_augmentation(frames, torch.Generator().manual_seed(1))
    second = pilot_view_augmentation(frames, torch.Generator().manual_seed(2))
    assert first.shape == second.shape == frames.shape
    assert first.data_ptr() != frames.data_ptr()
    assert 0 <= first.min() <= first.max() <= 1


def test_eventclr_view_transform_keeps_input_unchanged_and_has_expected_shape():
    events = {'t': np.array([0, 1, 2, 3], dtype=np.int64),
              'x': np.array([0, 1, 2, 3], dtype=np.int64),
              'y': np.array([0, 1, 2, 3], dtype=np.int64),
              'p': np.array([0, 1, 0, 1], dtype=np.int64)}
    original = {name: value.copy() for name, value in events.items()}
    transform = EventCLRViewTransform(time_steps=4, size=16)
    view = transform(events, np.random.default_rng(4), torch.Generator().manual_seed(4))
    assert view.shape == (4, 2, 16, 16)
    for name in events:
        np.testing.assert_array_equal(events[name], original[name])


def test_eventclr_view_does_not_depend_on_global_torch_rng():
    events = {'t':np.arange(100,dtype=np.int64),
              'x':np.arange(100,dtype=np.int64)%128,
              'y':np.arange(100,dtype=np.int64)%128,
              'p':np.arange(100,dtype=np.int64)%2}
    transform=EventCLRViewTransform(time_steps=4,size=32)
    torch.manual_seed(1)
    first=transform(events,np.random.default_rng(13),torch.Generator().manual_seed(13))
    torch.manual_seed(999)
    second=transform(events,np.random.default_rng(13),torch.Generator().manual_seed(13))
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_multicrop_keeps_global_views_first_and_locals_smaller(tmp_path):
    import json
    root = tmp_path / 'CIFAR10DVS'
    (root / 'raw' / 'airplane').mkdir(parents=True)
    sample = root / 'raw' / 'airplane' / 'sample.aedat'
    addresses = np.array([(y << 8) | (x << 1) | (i % 2) for i, (x, y)
                          in enumerate(zip(range(20), range(20)))], dtype='>u4')
    times = np.arange(20, dtype='>u4')
    words = np.empty(40, dtype='>u4')
    words[::2], words[1::2] = addresses, times
    sample.write_bytes(b'#!AER-DAT2.0\n' + words.tobytes())
    (root / 'split_seed42.json').write_text(json.dumps({'train':[
        {'path':'airplane/sample.aedat','label':0} for _ in range(4)]}),encoding='utf-8')
    dataset = Cifar10DVSPilotDataset(root, seed=42, size=48,
                                    augmentation='eventclr', local_crops=2)
    views = dataset[0]
    assert [tuple(view.shape) for view in views] == [
        (4,2,48,48),(4,2,48,48),(4,2,24,24),(4,2,24,24)]
    first_seed = dataset.view_seed(0, 0)
    dataset.set_epoch(1)
    assert dataset.view_seed(0, 0) != first_seed


def test_limited_pilot_samples_are_stratified_and_reproducible(tmp_path):
    import json
    root = tmp_path / 'CIFAR10DVS'
    root.mkdir()
    records = [{'path':f'{label}/{index}.aedat','label':label}
               for label in range(10) for index in range(20)]
    (root / 'split_seed42.json').write_text(json.dumps({'train':records}),encoding='utf-8')
    first = Cifar10DVSPilotDataset(root, limit=53)
    second = Cifar10DVSPilotDataset(root, limit=53)
    assert first.records == second.records
    counts = [sum(record['label'] == label for record in first.records) for label in range(10)]
    assert len(first) == 53
    assert max(counts) - min(counts) <= 1
    assert min(counts) >= 5


def test_eventclr_histogram_matches_direct_event_accumulation():
    from spike_dino.data import eventclr_representation
    rng = np.random.default_rng(11)
    count = 2000
    events = {'t':np.sort(rng.integers(0,1000,size=count)),
              'x':rng.integers(0,128,size=count),
              'y':rng.integers(0,128,size=count),
              'p':rng.integers(0,2,size=count)}
    bins = np.minimum((events['t'] - events['t'][0]) * 4 //
                      max(int(events['t'][-1] - events['t'][0]),1),3)
    reference = np.zeros((4,2,128,128),dtype=np.float32)
    np.add.at(reference,(bins,events['p'],events['y'],events['x']),1)
    reference /= reference.max()
    expected = torch.nn.functional.adaptive_max_pool2d(torch.from_numpy(reference),(48,48))
    torch.testing.assert_close(eventclr_representation(events,4,48),expected)
