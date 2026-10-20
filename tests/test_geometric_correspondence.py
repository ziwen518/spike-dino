import torch

from scripts.diagnose_v3_geometric_correspondence import (geometric_targets,
                                                          patch_activity)


def test_shifted_crop_maps_patch_centers_in_sensor_coordinates():
    y, x, nearest, inside = geometric_targets(
        (0, 0, 16, 16), (0, 4, 16, 16), grid=4, device='cpu')
    assert not inside[0]
    assert inside[1]
    assert x[1] == 0
    assert y[1] == 0
    assert nearest[1] == 0
    assert inside[15]


def test_patch_activity_reads_nonempty_event_regions():
    frames = torch.zeros(4, 2, 8, 8)
    frames[1, 0, 5, 6] = 1
    active = patch_activity(frames, patch_size=4)
    torch.testing.assert_close(active, torch.tensor([False, False, False, True]))
