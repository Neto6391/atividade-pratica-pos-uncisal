from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


SOURCES = {
    "fraud-ecommerce.zip": "https://www.kaggle.com/api/v1/datasets/download/vbinh002/fraud-ecommerce",
    "malicious-urls-dataset.zip": "https://www.kaggle.com/api/v1/datasets/download/sid321axn/malicious-urls-dataset",
    "creditcardfraud.zip": "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud",
}


def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        destination = output_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"ok {destination} ({destination.stat().st_size} bytes)")
            continue
        print(f"downloading {url}")
        urlretrieve(url, destination)
        print(f"saved {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
