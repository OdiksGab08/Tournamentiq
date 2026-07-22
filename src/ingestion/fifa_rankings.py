"""Download a raw snapshot of the FIFA Men's Rankings page.

This executable ingestion helper requests FIFA's public men's rankings page and
writes the unparsed HTML response to ``data/raw/fifa_rankings/fifa_page.html``.
The saved snapshot is an input for later ranking extraction or auditing and is
kept separate from the prediction and training services.
"""

from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "fifa_rankings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fifa_rankings.csv"


def download_rankings() -> None:
    """Fetch the FIFA Men's Rankings page and persist its raw HTML response.

    Args:
        None.

    Returns:
        None. The downloaded response body is written to
        ``data/raw/fifa_rankings/fifa_page.html``.

    Notes:
        Persisting the source page makes downstream parsing reproducible and
        gives data-maintenance workflows an inspectable record of the response.
    """

    url = "https://inside.fifa.com/fifa-world-ranking/men"

    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)

    print("Status:", response.status_code)

    with open(OUTPUT_DIR / "fifa_page.html", "w", encoding="utf-8") as f:
        f.write(response.text)

    print("HTML downloaded successfully.")


if __name__ == "__main__":
    download_rankings()
