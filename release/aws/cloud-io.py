#!/usr/bin/env python3
"""S3-compatible worker archive transport, including private Cloudflare R2.

Shares the download/upload CLI contract with Azure and GCP transports. Boto3's
managed transfers stream archives, retry requests and use multipart uploads;
no Azure implementation or stored object is changed by this adapter.
"""

import argparse
import os
from pathlib import Path
import sys


def credentials(env):
    """Explicit R2 credentials must never fall through to an AWS/VM profile."""
    r2 = env.get("DL4J_CACHE_BACKEND") == "r2"
    names = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if r2 else (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
    )
    values = []
    for name in names:
        value = env.get(name, "").strip()
        if not value or any(char.isspace() for char in value):
            raise ValueError(f"{name} must contain a nonempty credential without embedded whitespace")
        if env.get("GITHUB_ACTIONS") == "true":
            print("::add-mask::" + value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"), flush=True)
        values.append(value)
    return values


def client(env=None):
    env = os.environ if env is None else env
    access_key, secret_key = credentials(env)
    import boto3
    from botocore.config import Config
    endpoint = env.get("DL4J_S3_ENDPOINT", "").strip()
    if env.get("DL4J_CACHE_BACKEND") == "r2" and not endpoint.startswith("https://"):
        raise ValueError("R2 requires an explicit HTTPS DL4J_S3_ENDPOINT")
    return boto3.client(
        "s3", endpoint_url=endpoint or None,
        region_name=env.get("DL4J_S3_REGION", "us-east-1"),
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
        # R2 uses account API credentials, never an inherited AWS session token.
        aws_session_token=None if env.get("DL4J_CACHE_BACKEND") == "r2" else env.get("AWS_SESSION_TOKEN"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      retries={"mode": "standard", "max_attempts": 6},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"),
    )


def transfer(action, bucket, object_name, file, missing_exit_code=None):
    from boto3.s3.transfer import TransferConfig
    from botocore.exceptions import ClientError
    storage = client()
    options = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                             multipart_chunksize=64 * 1024 * 1024,
                             max_concurrency=4)
    try:
        if action == "download":
            file.parent.mkdir(parents=True, exist_ok=True)
            storage.download_file(bucket, object_name, str(file), Config=options)
        else:
            storage.upload_file(str(file), bucket, object_name, Config=options)
    except ClientError as exc:
        # A missing key is a cache miss; forbidden/bad signatures/network errors
        # are failures. Do not print SDK errors which may contain request data.
        code = str(exc.response.get("Error", {}).get("Code", "unknown"))
        if action == "download":
            file.unlink(missing_ok=True)
            if code in {"404", "NoSuchKey", "NotFound"} and missing_exit_code is not None:
                return missing_exit_code
        raise RuntimeError(f"S3 cache {action} failed (code={code})") from None
    except Exception:
        if action == "download":
            file.unlink(missing_ok=True)
        raise RuntimeError(f"S3 cache {action} failed") from None
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "upload"))
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--client-id", default="")  # shared transport CLI
    parser.add_argument("--missing-exit-code", type=int)
    args = parser.parse_args()
    return transfer(args.action, args.bucket, args.object, args.file, args.missing_exit_code)


if __name__ == "__main__":
    sys.exit(main())
