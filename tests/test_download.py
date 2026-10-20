import hashlib
import io
import zipfile
import pytest
from scripts.download_cifar10dvs import download, extract


def test_resume_validates_range_and_checksum(tmp_path,monkeypatch):
    content=b'example event archive'
    (tmp_path/'sample.zip.part').write_bytes(content[:5])
    class Response(io.BytesIO):
        status=206
        headers={'Content-Range':f'bytes 5-{len(content)-1}/{len(content)}'}
    def open_response(request,timeout):
        assert request.headers['Range']=='bytes=5-'
        return Response(content[5:])
    monkeypatch.setattr('urllib.request.urlopen',open_response)
    result=download({'name':'sample.zip','size':len(content),'url':'https://example.com/data',
                     'md5':hashlib.md5(content).hexdigest()},tmp_path)
    assert result.read_bytes()==content


def test_archive_rejects_path_escape(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('../escape.txt','bad')
    with pytest.raises(ValueError):
        extract(path,tmp_path/'raw')
    assert not (tmp_path/'escape.txt').exists()
