import os
import urllib
import torch
from pathlib import Path

DEFAULT_WEIGHTS = {
    'yolo': {
        'url': f'https://github.com/ultralytics/yolov5/releases/download/v1.0/',
        'assets': [
            'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov51.pt', 'yolov5x.pt', 'yolov5n6.pt',
            'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
    }
}

def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")

def attempt_download(model, weights):

    # If URL Specified
    if str(weights).startswith(('http:/', 'https://')):
        url = str(weights).replace(':/','://')
        file = Path(urllib.parse.unquote(str(weights))).name.split('?')[0]
        safe_download(file=file, url=url)
        return True
    else:
        if weights in DEFAULT_WEIGHTS[model]['assets']:
            url = DEFAULT_WEIGHTS[model][url] + weights
            safe_download(file=weights, url=url)
            return True
        else:
            print(f"ERROR: Weight {weights} not found in DEFAULT_WEIGHTS")
            return False
        
