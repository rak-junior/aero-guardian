
"""
=============================================================================
NASA DASHlink ALFA Dataset Downloader
=============================================================================

Purpose:
    Download curated NASA ALFA UAV flight telemetry archives from DASHlink
    for offline research and safety analysis.

=============================================================================
"""

from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from typing import List, Optional

import requests

# =============================================================================
# Configuration
# =============================================================================

BASE_URL = "https://c3.ndc.nasa.gov/dashlink/static/media/dataset"

# Explicit allow-list (MUST be manually verified)
FILES: List[str] = [
    "Tail_666_9.zip",
]

OUTPUT_DIR = Path("C:/VIRAK/Python Code/uav-safety-ai-platform/data/telemetry/raw/nasa/raw_zips")
LOG_FILE = Path("C:/VIRAK/Python Code/uav-safety-ai-platform/logs/download.log")

REQUEST_TIMEOUT = 60  # seconds
RETRY_COUNT = 3
RETRY_DELAY = 5  # seconds

# Optional: path to cookies file exported from browser (Netscape format)
COOKIES_FILE: Optional[Path] = None
# Example:
# COOKIES_FILE = Path("configs/dashlink_cookies.txt")

# =============================================================================
# Logging Setup
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def load_cookies(session: requests.Session, cookies_file: Path) -> None:
    """
    Load cookies from a Netscape-format cookies file into a session.
    """
    try:
        import http.cookiejar as cookiejar
    except ImportError as exc:
        raise RuntimeError("Failed to import cookiejar") from exc

    cj = cookiejar.MozillaCookieJar()
    cj.load(cookies_file, ignore_discard=True, ignore_expires=True)
    session.cookies.update(cj)
    logger.info("Loaded authentication cookies from %s", cookies_file)


def download_file(
    session: requests.Session,
    filename: str,
    output_dir: Path,
) -> None:
    """
    Download a single file with retries and integrity checks.
    """
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        logger.info("Skipping existing file: %s", filename)
        return

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", filename, attempt, RETRY_COUNT)
            response = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info("Successfully downloaded %s", filename)
            return

        except requests.RequestException as exc:
            logger.warning(
                "Download failed for %s (attempt %d): %s",
                filename,
                attempt,
                exc,
            )
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Failed to download {filename}") from exc


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    logger.info("Starting NASA ALFA dataset download")
    logger.info("Files to download: %d", len(FILES))

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Academic-Research-Downloader/1.0"
        })

        if COOKIES_FILE:
            load_cookies(session, COOKIES_FILE)
        else:
            logger.warning(
                "No cookies file provided. "
                "If downloads fail, authentication may be required."
            )

        for filename in FILES:
            download_file(session, filename, OUTPUT_DIR)

    logger.info("All downloads completed successfully")


if __name__ == "__main__":
    main()
