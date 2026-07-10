"""
Download PTB-XL (100 Hz records + metadata) from PhysioNet.

Handles two known issues:

1. wfdb.dl_database() crashes on PTB-XL because the RECORDS file is missing a
   newline between the last records100 entry and the first records500 entry,
   producing an invalid URL. We parse RECORDS with a regex instead.

2. PhysioNet's /files/ endpoint intermittently returns 500 for the metadata
   CSVs. We try wfdb.dl_files() first, fall back to a retrying requests
   session, and treat metadata failure as non-fatal so the signal download
   can still proceed.
"""

import os
import re
import sys
import time

import requests
import wfdb

DB_NAME = "ptb-xl"
DB_VERSION = "1.0.3"
BASE_URL = f"https://physionet.org/files/{DB_NAME}/{DB_VERSION}"
DL_DIR = os.path.join("data", "ptbxl")

# Metadata CSVs that dl_database does not fetch but the project needs.
METADATA_FILES = [
    "ptbxl_database.csv",  # record index, demographics, SCP diagnostic codes
    "scp_statements.csv",  # SCP code -> diagnostic superclass mapping
]

# Matches e.g. records100/00000/00001_lr  or  records500/21000/21837_hr
RECORD_PATTERN = re.compile(r"records\d+/\d+/\d+_(?:lr|hr)")

# PhysioNet can reject the default python-requests User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ecg-agent/1.0)"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get_with_retry(session: requests.Session, url: str, attempts: int = 4):
    """GET with exponential backoff. Returns response or raises."""
    last_exc = None
    for i in range(attempts):
        try:
            resp = session.get(url, timeout=180, stream=True)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = 2**i
            print(f"    attempt {i + 1}/{attempts} failed ({exc.__class__.__name__}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise last_exc


def fetch_record_list(session: requests.Session, sampling_rate: int = 100) -> list:
    """Return record paths for the requested sampling rate (100 or 500 Hz)."""
    suffix = "_lr" if sampling_rate == 100 else "_hr"

    resp = _get_with_retry(session, f"{BASE_URL}/RECORDS")
    text = resp.text

    # Regex findall is immune to the missing-newline concatenation bug.
    all_records = RECORD_PATTERN.findall(text)
    records = [r for r in all_records if r.endswith(suffix)]

    if not records:
        sys.exit(f"No {sampling_rate} Hz records found in RECORDS file.")

    return records


def download_metadata(session: requests.Session) -> bool:
    """Fetch metadata CSVs. Returns True if all present, False otherwise."""
    missing = [f for f in METADATA_FILES
               if not os.path.exists(os.path.join(DL_DIR, f))]

    for name in METADATA_FILES:
        if name not in missing:
            print(f"  [skip] {name} already present")

    if not missing:
        return True

    # Primary path: let wfdb handle the transfer.
    try:
        print(f"  [wfdb] fetching {', '.join(missing)}")
        wfdb.dl_files(DB_NAME, DL_DIR, missing, overwrite=False)
    except Exception as exc:
        print(f"  [wfdb] failed: {exc}")
        print("  [http] falling back to direct download")

        for name in missing:
            dest = os.path.join(DL_DIR, name)
            try:
                resp = _get_with_retry(session, f"{BASE_URL}/{name}")
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                print(f"  [http] got {name}")
            except Exception as exc2:
                print(f"  [http] could not fetch {name}: {exc2}")

    still_missing = [f for f in METADATA_FILES
                     if not os.path.exists(os.path.join(DL_DIR, f))]
    if still_missing:
        print(f"\n  WARNING: metadata not downloaded: {', '.join(still_missing)}")
        print("  Signal download will continue. Re-run this script later to retry,")
        print(f"  or download manually from {BASE_URL}/")
        return False

    return True


def main() -> None:
    os.makedirs(DL_DIR, exist_ok=True)
    session = _session()

    print("Fetching metadata CSVs...")
    have_metadata = download_metadata(session)

    print("\nResolving 100 Hz record list...")
    records = fetch_record_list(session, sampling_rate=100)
    print(f"  Found {len(records)} records (expected ~21,799)")

    print("\nDownloading signal files. This will take a while.")
    print("Safe to interrupt and re-run: existing files are skipped.\n")

    wfdb.dl_database(
        DB_NAME,
        dl_dir=DL_DIR,
        records=records,
        keep_subdirs=True,
        overwrite=False,
    )

    print("\nPTB-XL signal download complete.")
    print(f"Data location: {os.path.abspath(DL_DIR)}")

    if not have_metadata:
        print("\nNOTE: metadata CSVs are still missing. Re-run to retry.")


if __name__ == "__main__":
    main()