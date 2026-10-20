"""Download original CIFAR10-DVS with publisher checksums; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile


def digest(path):
    md5 = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            md5.update(chunk)
    return md5.hexdigest()


def download(item, directory):
    target = directory/item['name']
    if target.exists():
        if target.stat().st_size == item['size'] and digest(target) == item['md5']:
            print('verified existing', target.name, flush=True)
            return target
        raise RuntimeError(f'Existing file fails checksum; preserve and inspect: {target}')
    part = target.with_suffix(target.suffix+'.part')
    for attempt in range(5):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > item['size']:
                raise ValueError(f'Oversized partial file: {part}')
            if offset < item['size']:
                headers = {'User-Agent':'SpikeDINO-dataset-fetch/1.0', 'Accept-Encoding':'identity'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(item['url'],headers=headers),timeout=60) as response:
                    if response.status == 206:
                        if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('Server returned incorrect byte range')
                        mode = 'ab'
                    elif response.status == 200:
                        mode = 'wb'
                        offset = 0
                    else:
                        raise ValueError(f'Unexpected HTTP status {response.status}')
                    last = time.monotonic()
                    with part.open(mode) as stream:
                        for chunk in iter(lambda:response.read(4*1024*1024), b''):
                            stream.write(chunk)
                            offset += len(chunk)
                            if time.monotonic()-last > 15:
                                print(f'{target.name}: {offset}/{item["size"]} bytes',flush=True)
                                last=time.monotonic()
            if part.stat().st_size != item['size']:
                raise IOError('Incomplete download')
            if digest(part) != item['md5']:
                raise ValueError(f'Checksum mismatch; partial file preserved: {part}')
            part.rename(target)
            print('downloaded and verified',target.name,flush=True)
            return target
        except (OSError, TimeoutError) as error:
            print(f'attempt {attempt+1}: {type(error).__name__}: {error}',flush=True)
            if attempt==4:
                raise
            time.sleep(min(2**attempt,16))


def extract(archive, root):
    root = root.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            dest = (root/info.filename).resolve()
            if not dest.is_relative_to(root) or ':' in info.filename or '\\' in info.filename:
                raise ValueError(f'Unsafe archive path: {info.filename}')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not supported')
        z.extractall(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('dataset'))
    parser.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[1]/'configs/cifar10dvs_download.json')
    parser.add_argument('--archives-only',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    root=args.root.resolve()/'CIFAR10DVS'
    archives=root/'archives'
    archives.mkdir(parents=True,exist_ok=True)
    remaining=sum(max(0,f['size']-(archives/f['name']).stat().st_size) if (archives/f['name']).exists() else f['size'] for f in manifest['files'])
    required=remaining+(0 if args.archives_only else 25*1024**3)
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f'Insufficient free space; require approximately {required/1024**3:.1f} GiB')
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for item in manifest['files']:
        archive=download(item,archives)
        if archive.suffix=='.zip' and not args.archives_only:
            extract(archive,root/'raw')
    counts={name:len(list((root/'raw'/name).rglob('*.aedat'))) for name in
            ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']}
    if not args.archives_only and any(n!=1000 for n in counts.values()):
        raise RuntimeError(f'Expected 1000 recordings per class, found {counts}')
    status={'archives_verified':True,'extracted':not args.archives_only,'class_counts':counts}
    (root/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    main()
