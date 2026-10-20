"""Finish dataset preparation when verified archive transfers arrive."""
import json
from pathlib import Path
import subprocess
import sys
import time

project=Path(__file__).resolve().parents[1]
status_path=project/'dataset/preparation_status.json'
deadline=time.monotonic()+4*3600
completed=[]


def write_status(state,**kwargs):
    status={'state':state,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'completed':completed,**kwargs}
    temp=status_path.with_suffix('.tmp')
    temp.write_text(json.dumps(status,indent=2),encoding='utf-8')
    temp.replace(status_path)


try:
    for name in ['CIFAR10DVS','NCALTECH101']:
        if not (project/'dataset'/name/'read_check.json').exists():
            raise RuntimeError(f'Missing initial read verification: {name}')
        completed.append(name)
    pending={'NMNIST':['Test.zip','Train.zip'],
             'DVSGesture':['ibmGestureTest.tar.gz','ibmGestureTrain.tar.gz']}
    while pending:
        write_status('waiting_for_archives',pending=pending)
        if time.monotonic()>deadline:
            raise TimeoutError('Archive transfer has not completed within four hours')
        for dataset,names in list(pending.items()):
            if (project/'dataset'/dataset/'read_check.json').is_file():
                completed.append(dataset)
                del pending[dataset]
                continue
            if all((project/'dataset'/dataset/'archives'/name).is_file() for name in names):
                write_status('validating',dataset=dataset,pending=pending)
                subprocess.run([sys.executable,str(project/'scripts/prepare_extra_data.py'),
                                '--dataset',dataset,'--root',str(project/'dataset')],
                               cwd=project,check=True)
                completed.append(dataset)
                del pending[dataset]
        if pending:time.sleep(15)
    write_status('complete',pending={})
    print('ALL FOUR DATASETS VERIFIED AND SAMPLE-READ',flush=True)
except Exception as error:
    write_status('failed',error=f'{type(error).__name__}: {error}')
    raise
