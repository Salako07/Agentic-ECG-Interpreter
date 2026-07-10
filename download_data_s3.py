"""
Download PTB-XL from the PhysioNet S3 open-data mirror.

Why S3 instead of physionet.org?

The https://physionet.org/files/... endpoints intermittently return 500/502
errors. PhysioNet mirrors every open-access project to a public, unauthenticated
S3 bucket (s3://physionet-open/), which is served by AWS and is unaffected by
those outages. PhysioNet documents this as an official download route.

We skip records500/ (the 500 Hz waveforms). The 100 Hz records in records100/
are what virtually every published PTB-XL benchmark uses, and they are more
than sufficient for feature-based classification with neurokit2. Skipping the
500 Hz set takes the download from ~3.0 GB to ~1.3 GB.

Requires: pip install boto3
"""

import os
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET = "physionet-open"
PREFIX = "ptb-xl/1.0.3/"
DL_DIR = os.path.join("data", "ptbxl")

# Prefixes to skip. records500 is the 500 Hz waveform set.
SKIP_PREFIXES = ("records500/",)


def make_client():
    """Anonymous S3 client. PhysioNet's open bucket needs no credentials."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_objects(client):
    """Yield (key, size) for every object under PREFIX, minus skipped dirs."""
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(PREFIX):]

            if not rel or key.endswith("/"):
                continue
            if rel.startswith(SKIP_PREFIXES):
                continue

            yield key, obj["Size"]


def main() -> None:
    os.makedirs(DL_DIR, exist_ok=True)
    client = make_client()

    print(f"Listing s3://{BUCKET}/{PREFIX} ...")
    try:
        objects = list(list_objects(client))
    except ClientError as exc:
        sys.exit(f"Could not list bucket: {exc}")

    total_bytes = sum(size for _, size in objects)
    print(f"  {len(objects):,} files, {total_bytes / 1e9:.2f} GB "
          f"(records500/ excluded)\n")

    downloaded = skipped = failed = 0

    for i, (key, size) in enumerate(objects, start=1):
        rel = key[len(PREFIX):]
        dest = os.path.join(DL_DIR, *rel.split("/"))

        # Skip if already present at the correct size (resumable).
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            skipped += 1
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)

        try:
            client.download_file(BUCKET, key, dest)
            downloaded += 1
        except ClientError as exc:
            print(f"  FAILED {rel}: {exc}")
            failed += 1
            continue

        if downloaded % 250 == 0:
            print(f"  {i:,}/{len(objects):,} processed "
                  f"({downloaded:,} downloaded, {skipped:,} skipped)")

    print("\nDone.")
    print(f"  downloaded: {downloaded:,}")
    print(f"  skipped (already present): {skipped:,}")
    print(f"  failed: {failed:,}")
    print(f"\nData location: {os.path.abspath(DL_DIR)}")

    # Sanity check the two files that carry all the labels.
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        path = os.path.join(DL_DIR, name)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()