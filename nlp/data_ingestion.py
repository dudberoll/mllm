from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_PDF_URLS = (
    "https://publichealth.hsc.wvu.edu/media/5553/russian-history-part-i.pdf",
    "https://publichealth.hsc.wvu.edu/media/5589/history-of-russia-part-2.pdf",
)


def download_pdfs(
    pdf_urls: tuple[str, ...] | list[str],
    data_dir: Path,
    timeout: int = 60,
) -> list[Path]:
    """Download the PDFs into the local data directory."""
    # TODO: YOUR CODE HERE
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_paths = []
    for url in pdf_urls:
        if url is not None:
            pdf_url = urlparse(url)
        else:
            raise ValueError("пустой url")
        name = Path(pdf_url.path).name
        path = data_dir / name
        
        if not path.exists():
            response = requests.get(url, timeout=timeout)
        
            with open(path, "wb") as f:
                f.write(response.content)
    
        downloaded_paths.append(path)
    # raise NotImplementedError
    return downloaded_paths


if __name__ == "__main__":
    download_pdfs(DEFAULT_PDF_URLS, Path("."))
