"""Bounded parallel HTTP range downloads with final source MD5 validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request

from download_cifar10dvs import digest


def fetch_file(item, root, workers=8):
    root.mkdir(parents=True,exist_ok=True)
    target=root/item['name']
    if target.exists():
        if target.stat().st_size!=item['size'] or digest(target)!=item['md5']:
            raise ValueError(f'Existing file has incorrect checksum: {target}')
        print('verified existing',target,flush=True)
        return target
    chunk_size=16*1024**2
    parts=root/(item['name']+'.chunks')
    parts.mkdir(exist_ok=True)
    spans=[(i,start,min(start+chunk_size,item['size'])-1)
           for i,start in enumerate(range(0,item['size'],chunk_size))]
    def fetch(span):
        index,start,end=span
        path=parts/f'{index:05d}'
        if path.exists() and path.stat().st_size==end-start+1:
            return path
        temporary=parts/f'{index:05d}.partial'
        for attempt in range(5):
            try:
                req=urllib.request.Request(item['url'],headers={'Range':f'bytes={start}-{end}',
                                                               'Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=60) as response:
                    expected=f'bytes {start}-{end}/{item["size"]}'
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError(f'Incorrect range for {target.name}, chunk {index}')
                    with temporary.open('wb') as stream:
                        while chunk:=response.read(1024**2):
                            stream.write(chunk)
                if temporary.stat().st_size!=end-start+1:
                    raise IOError('Truncated range')
                temporary.replace(path)
                print(f'{target.name}: chunk {index+1}/{len(spans)}',flush=True)
                return path
            except OSError:
                if attempt==4:raise
                time.sleep(min(2**attempt,16))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths=list(pool.map(fetch,spans))
    combined=root/(item['name']+'.assembling')
    with combined.open('wb') as out:
        for path in paths:
            with path.open('rb') as source:
                for chunk in iter(lambda:source.read(8*1024**2),b''):out.write(chunk)
    if digest(combined)!=item['md5']:
        raise ValueError(f'MD5 mismatch: {combined}; preserved for inspection')
    combined.rename(target)
    print('MD5 VERIFIED',target,flush=True)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=16:parser.error('workers must be 1..16')
    for item in json.loads(args.manifest.read_text(encoding='utf-8'))['files']:
        fetch_file(item,args.root,args.workers)
