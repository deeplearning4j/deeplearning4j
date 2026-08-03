#!/usr/bin/env python3
"""Dependency-free Azure worker transport for Blob Storage and the kill switch.

The helper authenticates with the VM's managed identity through Azure IMDS, so
workers never receive storage keys or release/publication credentials.
"""

import argparse
import datetime as dt
import json
import mimetypes
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

METADATA_TOKEN_URL = "http://169.254.169.254/metadata/identity/oauth2/token"
STORAGE_RESOURCE = "https://storage.azure.com/"
STORAGE_API_VERSION = "2023-11-03"
_TOKEN = {"value": None, "expires": 0.0, "client_id": None}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def request(url, *, method="GET", data=None, headers=None, authenticated=False,
            client_id=None, retries=6, retry_statuses=(), accepted_statuses=()):
    headers = dict(headers or {})
    payload = data.encode("utf-8") if isinstance(data, str) else data
    delay = 1.0
    attempts = max(retries, 36) if authenticated else retries
    retry_statuses = set(retry_statuses)
    accepted_statuses = set(accepted_statuses)
    for attempt in range(attempts):
        if authenticated:
            headers["Authorization"] = "Bearer " + access_token(client_id)
            headers["x-ms-version"] = STORAGE_API_VERSION
            headers["x-ms-date"] = dt.datetime.now(dt.timezone.utc).strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read()
            body = body_bytes.decode("utf-8", "replace")
            if exc.code in accepted_statuses:
                return body_bytes
            if exc.code == 401 and authenticated:
                _TOKEN["expires"] = 0
            retryable = exc.code in (408, 409, 429, 500, 502, 503, 504)
            retryable = retryable or exc.code in retry_statuses
            retryable = retryable or (authenticated and exc.code in (401, 403))
            extended = exc.code in retry_statuses or (
                authenticated and exc.code in (401, 403)
            )
            retry_limit = attempts if extended else retries
            if retryable and attempt + 1 < retry_limit:
                time.sleep(delay)
                delay = min(delay * 2, 20)
                continue
            if exc.code in (404, 410):
                raise FileNotFoundError(url) from exc
            raise RuntimeError(
                f"Azure request failed ({exc.code}) {url}: {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"Azure request failed {url}: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 20)
    raise AssertionError("unreachable")


def access_token(client_id=None):
    client_id = client_id or os.environ.get("AZURE_CLIENT_ID")
    now = time.time()
    if (
        _TOKEN["value"]
        and _TOKEN["expires"] > now + 120
        and _TOKEN["client_id"] == client_id
    ):
        return _TOKEN["value"]
    query = {
        "api-version": "2018-02-01",
        "resource": STORAGE_RESOURCE,
    }
    if client_id:
        query["client_id"] = client_id
    raw = request(
        METADATA_TOKEN_URL + "?" + urllib.parse.urlencode(query),
        headers={"Metadata": "true"},
        retries=36,
        retry_statuses=(400, 404, 410, 429, 500, 502, 503, 504),
    )
    value = json.loads(raw.decode("utf-8"))
    _TOKEN["value"] = value["access_token"]
    _TOKEN["expires"] = float(value.get("expires_on", now + 3000))
    _TOKEN["client_id"] = client_id
    return _TOKEN["value"]


def parse_bucket(value):
    """Return (storage account, container) from account/container or a blob URL."""
    if "://" in value:
        parsed = urllib.parse.urlparse(value)
        account = parsed.netloc.split(".", 1)[0]
        container = parsed.path.strip("/").split("/", 1)[0]
    else:
        parts = value.strip("/").split("/", 1)
        if len(parts) != 2:
            raise ValueError("--bucket must be STORAGE_ACCOUNT/CONTAINER")
        account, container = parts
    if not re.fullmatch(r"[a-z0-9]{3,24}", account):
        raise ValueError(f"invalid Azure storage account in --bucket: {account!r}")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])?", container):
        raise ValueError(f"invalid Azure container in --bucket: {container!r}")
    return account, container


def blob_url(bucket, name):
    account, container = parse_bucket(bucket)
    quoted_name = urllib.parse.quote(name.lstrip("/"), safe="/~")
    return f"https://{account}.blob.core.windows.net/{container}/{quoted_name}"


def upload_bytes(bucket, name, data, content_type="application/octet-stream", client_id=None):
    return request(
        blob_url(bucket, name),
        method="PUT",
        data=data,
        headers={
            "Content-Type": content_type,
            "x-ms-blob-type": "BlockBlob",
        },
        authenticated=True,
        client_id=client_id,
    )


def download_bytes(bucket, name, client_id=None):
    return request(
        blob_url(bucket, name),
        authenticated=True,
        client_id=client_id,
    )


def create_append_blob(bucket, name, client_id=None):
    return request(
        blob_url(bucket, name),
        method="PUT",
        data=b"",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "If-None-Match": "*",
            "x-ms-blob-type": "AppendBlob",
        },
        authenticated=True,
        client_id=client_id,
        accepted_statuses=(409, 412),
    )


def append_bytes(bucket, name, data, position, client_id=None):
    if not data:
        return b""
    return request(
        blob_url(bucket, name) + "?comp=appendblock",
        method="PUT",
        data=data,
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "x-ms-blob-condition-appendpos": str(position),
        },
        authenticated=True,
        client_id=client_id,
        retries=1,
    )


def command_upload(args):
    path = pathlib.Path(args.file)
    content_type = args.content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    upload_bytes(args.bucket, args.object, path.read_bytes(), content_type, args.client_id)


def command_upload_json(args):
    payload = pathlib.Path(args.file).read_bytes() if args.file else sys.stdin.buffer.read()
    json.loads(payload.decode("utf-8"))
    upload_bytes(args.bucket, args.object, payload, "application/json", args.client_id)


def command_download(args):
    try:
        payload = download_bytes(args.bucket, args.object, args.client_id)
    except FileNotFoundError:
        return 1
    pathlib.Path(args.file).write_bytes(payload)
    return 0


def command_kill_enabled(args):
    try:
        value = json.loads(download_bytes(args.bucket, args.object, args.client_id).decode("utf-8"))
    except FileNotFoundError:
        print("kill switch object is missing", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"kill switch is unreadable: {exc}", file=sys.stderr, flush=True)
        return 2
    expected_epoch = getattr(args, "controller_epoch", None)
    actual_epoch = value.get("controllerEpoch")
    if value.get("enabled") is True:
        if (
            not expected_epoch
            or actual_epoch == expected_epoch
            or value.get("force") is True
        ):
            return 0
        print(
            "ignoring kill switch written by a stale controller epoch",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if value.get("enabled") is False:
        if expected_epoch and actual_epoch != expected_epoch:
            print(
                "kill switch controller epoch does not match this worker",
                file=sys.stderr,
                flush=True,
            )
            return 2
        return 1
    print("kill switch JSON does not contain a boolean enabled field", file=sys.stderr, flush=True)
    return 2


def command_log(args):
    name = (
        f"{args.prefix.strip('/')}/{args.run_id}/{args.shard}/events/"
        f"{time.time_ns()}.json"
    )
    payload = json.dumps(
        {
            "timestamp": utc_now(),
            "runId": args.run_id,
            "shard": args.shard,
            "message": args.message,
        },
        sort_keys=True,
    ).encode("utf-8")
    upload_bytes(args.bucket, name, payload, "application/json", args.client_id)


def command_forward(args):
    path = pathlib.Path(args.file)
    stop_path = pathlib.Path(args.stop_file) if args.stop_file else None
    idle_after_stop = 0
    create_append_blob(args.bucket, args.object, args.client_id)
    remote = download_bytes(args.bucket, args.object, args.client_id)
    payload = path.read_bytes() if path.exists() else b""
    if not payload.startswith(remote):
        raise RuntimeError(
            "existing Azure live log is not a prefix of the local log; refusing to duplicate or overwrite it"
        )
    offset = len(remote)
    while True:
        payload = path.read_bytes() if path.exists() else b""
        if len(payload) < offset:
            raise RuntimeError("local live log was truncated while forwarding")
        chunk = payload[offset:]
        if chunk:
            try:
                append_bytes(
                    args.bucket, args.object, chunk, offset, args.client_id
                )
                offset = len(payload)
            except Exception as exc:
                try:
                    remote = download_bytes(args.bucket, args.object, args.client_id)
                except Exception as reconcile_exc:
                    print(
                        f"[dl4j-cloud-io] live log append failed: {exc}; "
                        f"reconciliation failed: {reconcile_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    if not payload.startswith(remote):
                        raise RuntimeError(
                            "Azure live log diverged after an ambiguous append; refusing to duplicate it"
                        ) from exc
                    if len(remote) > offset:
                        offset = len(remote)
                    else:
                        print(
                            f"[dl4j-cloud-io] live log append failed without committing: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
            idle_after_stop = 0
        elif stop_path and stop_path.exists():
            idle_after_stop += 1
            if idle_after_stop >= 2:
                return 0
        time.sleep(args.interval)


def add_identity_option(command):
    command.add_argument(
        "--client-id",
        help="user-assigned managed identity client ID; defaults to AZURE_CLIENT_ID/IMDS default",
    )


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload")
    upload.add_argument("--bucket", required=True, help="STORAGE_ACCOUNT/CONTAINER")
    upload.add_argument("--object", required=True)
    upload.add_argument("--file", required=True)
    upload.add_argument("--content-type")
    add_identity_option(upload)
    upload.set_defaults(func=command_upload)

    upload_json = commands.add_parser("upload-json")
    upload_json.add_argument("--bucket", required=True)
    upload_json.add_argument("--object", required=True)
    upload_json.add_argument("--file")
    add_identity_option(upload_json)
    upload_json.set_defaults(func=command_upload_json)

    download = commands.add_parser("download")
    download.add_argument("--bucket", required=True)
    download.add_argument("--object", required=True)
    download.add_argument("--file", required=True)
    add_identity_option(download)
    download.set_defaults(func=command_download)

    kill = commands.add_parser("kill-enabled")
    kill.add_argument("--bucket", required=True)
    kill.add_argument("--object", required=True)
    kill.add_argument("--controller-epoch")
    add_identity_option(kill)
    kill.set_defaults(func=command_kill_enabled)

    log = commands.add_parser("log")
    log.add_argument("--bucket", required=True)
    log.add_argument("--prefix", default="deeplearning4j/releases")
    log.add_argument("--run-id", required=True)
    log.add_argument("--shard", required=True)
    log.add_argument("--message", required=True)
    add_identity_option(log)
    log.set_defaults(func=command_log)

    forward = commands.add_parser("forward")
    forward.add_argument("--bucket", required=True)
    forward.add_argument("--object", required=True)
    forward.add_argument("--file", required=True)
    forward.add_argument("--stop-file")
    forward.add_argument("--interval", type=float, default=3.0)
    add_identity_option(forward)
    forward.set_defaults(func=command_forward)
    return root


def main():
    args = parser().parse_args()
    result = args.func(args)
    raise SystemExit(0 if result is None else result)


if __name__ == "__main__":
    main()
