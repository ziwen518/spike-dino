import json
import sys

from scripts.merge_v3_resumed_metrics import main


def test_merge_discards_steps_after_last_safe_checkpoint(tmp_path, monkeypatch):
    first = tmp_path / 'first.jsonl'
    resumed = tmp_path / 'resumed.jsonl'
    output = tmp_path / 'merged.jsonl'
    first.write_text(''.join(json.dumps({'step': step, 'branch': 'old'}) + '\n'
                             for step in (1, 2, 3, 4)), encoding='utf-8')
    resumed.write_text(''.join(json.dumps({'step': step, 'branch': 'new'}) + '\n'
                               for step in (3, 4, 5)), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', [
        'merge_v3_resumed_metrics', '--initial', str(first),
        '--resumed', str(resumed), '--resume-step', '2',
        '--expected-final', '5', '--output', str(output)])
    main()
    rows = [json.loads(line) for line in output.read_text(
        encoding='utf-8').splitlines()]
    assert [row['step'] for row in rows] == [1, 2, 3, 4, 5]
    assert [row['branch'] for row in rows] == ['old', 'old', 'new', 'new', 'new']
