"""Download datasets (Penn-Fudan and Oxford-IIIT Pets).

Functions return the path to the downloaded dataset directory.
"""
from pathlib import Path
import zipfile
import shutil
import requests
from tqdm import tqdm


def _download_url(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(out_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=out_path.name) as pbar:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_pennfudan(dest: Path) -> Path:
    """Downloads and extracts the Penn-Fudan Pedestrian dataset."""
    dest = Path(dest)
    zip_path = dest.with_suffix('.zip')
    url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
    if not dest.exists():
        dest.mkdir(parents=True, exist_ok=True)
        _download_url(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest.parent)
        # The zip extracts to PennFudanPed
        extracted = dest.parent / 'PennFudanPed'
        if extracted.exists():
            # move to dest
            if dest.exists():
                shutil.rmtree(dest)
            extracted.rename(dest)
        zip_path.unlink(missing_ok=True)
    return dest


def download_oxford_pets(dest: Path) -> Path:
    """Downloads Oxford-IIIT Pet dataset (images + annotations).

    This function downloads the image and annotation zips and extracts them.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    base = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data'
    files = ['images.tar.gz', 'annotations.tar.gz']
    import tarfile
    for f in files:
        url = f"{base}/{f}"
        out = dest / f
        if not (dest / f.replace('.tar.gz', '')).exists():
            _download_url(url, out)
            with tarfile.open(out, 'r:gz') as tar:
                tar.extractall(dest)
            out.unlink(missing_ok=True)
    # images are in dest/images
    return dest


if __name__ == '__main__':
    download_pennfudan(Path('./datasets/penn_fudan'))
    download_oxford_pets(Path('./datasets/oxford_pets'))
