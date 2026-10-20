import pytest
import torch
from spike_dino.pretrain_v3 import (FixedEpochBatchSampler,
                                    linear_warmup_value,
                                    partial_temporal_masks)


def test_linear_warmup_then_flat_values():
    assert [linear_warmup_value(step, 0., 1., 2)
            for step in range(5)] == [0., .5, 1., 1., 1.]
    assert [linear_warmup_value(step, .04, .07, 0)
            for step in range(2)] == [.07, .07]
    with pytest.raises(ValueError):
        linear_warmup_value(-1, 0., 1., 2)


def test_resumed_sampler_preserves_unseen_order():
    complete = list(FixedEpochBatchSampler(100, 4, 42, 0, 20))
    first = list(FixedEpochBatchSampler(100, 4, 42, 0, 7))
    resumed = list(FixedEpochBatchSampler(100, 4, 42, 7, 20))
    assert first + resumed == complete
    assert len({index for batch in complete for index, epoch in batch}) == 80


def test_sampler_repeats_records_with_new_epoch_context():
    batches = list(FixedEpochBatchSampler(12, 4, 42, 0, 6))
    assert all(epoch == 0 for batch in batches[:3] for _, epoch in batch)
    assert all(epoch == 1 for batch in batches[3:] for _, epoch in batch)
    assert sorted(index for batch in batches[:3] for index, _ in batch) == list(range(12))
    assert sorted(index for batch in batches[3:] for index, _ in batch) == list(range(12))
    assert batches == list(FixedEpochBatchSampler(12, 4, 42, 0, 2)) + list(
        FixedEpochBatchSampler(12, 4, 42, 2, 6))


def test_partial_temporal_masks_hide_half_selected_times_only():
    selected = torch.tensor([[True, False, True], [False, True, False]])
    torch.manual_seed(61)
    first = partial_temporal_masks([selected, selected], 4)
    torch.manual_seed(61)
    repeat = partial_temporal_masks([selected, selected], 4)
    assert all(torch.equal(a, b) for a, b in zip(first, repeat))
    for mask in first:
        assert mask.shape == (4, 2, 3)
        torch.testing.assert_close(mask.sum(0), selected.long() * 2)
        assert torch.equal(mask.any(0), selected)
