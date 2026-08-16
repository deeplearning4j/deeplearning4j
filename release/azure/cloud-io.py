#!/usr/bin/env python3
"""Dependency-free Azure worker transport for Blob Storage and the kill switch.

Azure VMs authenticate through managed identity. Non-VM workers, such as GitHub
Actions, use the controller-provided SAS or account-key connection string from
their secret environment; credentials are never written to worker files/logs.
"""

import argparse
import base64
import binascii
import datetime as dt
import hashlib
import hmac
import http.client
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
# Keep ordinary metadata uploads simple, but stream release archives through the
# block-list API well below Azure's per-request limit (currently 5,000 MiB).
SINGLE_PUT_LIMIT = 256 * 1024 * 1024
BLOCK_UPLOAD_SIZE = 128 * 1024 * 1024
MAX_BLOCKS = 50_000
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
            request_url, request_headers = storage_authentication(
                url, method, payload, headers, client_id
            )
        else:
            request_url, request_headers = url, headers
        req = urllib.request.Request(
            request_url, data=payload, headers=request_headers, method=method
        )
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


def storage_sas():
    """Return a configured Azure SAS query for non-VM workers.

    Azure VMs continue to use managed identity. GitHub Actions has no IMDS
    endpoint, so the shared release action exposes the controller-issued SAS
    through the same secret used by sccache; it is never written to a worker
    config or log.
    """
    connection = (
        os.environ.get("DL4J_AZURE_CONNECTION_STRING")
        or os.environ.get("SCCACHE_AZURE_CONNECTION_STRING")
        or ""
    )
    for part in connection.split(";"):
        if part.startswith("SharedAccessSignature="):
            return part.split("=", 1)[1].lstrip("?")
    return ""


def storage_connection_parts():
    """Return the Azure connection-string fields exposed to CI workers."""
    connection = (
        os.environ.get("DL4J_AZURE_CONNECTION_STRING")
        or os.environ.get("SCCACHE_AZURE_CONNECTION_STRING")
        or ""
    )
    parts = {}
    for item in connection.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip().lower()] = value.strip()
    return parts


def storage_shared_key():
    """Return (account, decoded key) for an Azure account-key connection string."""
    parts = storage_connection_parts()
    account = parts.get("accountname")
    encoded_key = parts.get("accountkey")
    if not account or not encoded_key:
        return None
    try:
        return account, base64.b64decode(encoded_key)
    except (ValueError, binascii.Error) as exc:
        raise RuntimeError("invalid Azure AccountKey in storage connection string") from exc


def _header_value(headers, name):
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return str(value)
    return ""


def _canonicalized_headers(headers):
    values = []
    for key, value in headers.items():
        key = key.lower()
        if not key.startswith("x-ms-"):
            continue
        value = re.sub(r"\s+", " ", str(value).strip())
        values.append((key, value))
    return "".join(f"{key}:{value}\n" for key, value in sorted(values))


def _canonicalized_resource(url, account):
    parsed = urllib.parse.urlsplit(url)
    resource = f"/{account}{urllib.parse.unquote(parsed.path or '/')}"
    query = {}
    for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        query.setdefault(key.lower(), []).append(urllib.parse.unquote(value))
    for key in sorted(query):
        resource += f"\n{key}:{','.join(sorted(query[key]))}"
    return resource


def _shared_key_authorization(url, method, payload, headers, account, key):
    content_length = "" if payload is None or len(payload) == 0 else str(len(payload))
    standard_fields = (
        method.upper(),
        _header_value(headers, "Content-Encoding"),
        _header_value(headers, "Content-Language"),
        content_length,
        _header_value(headers, "Content-MD5"),
        _header_value(headers, "Content-Type"),
        _header_value(headers, "Date"),
        _header_value(headers, "If-Modified-Since"),
        _header_value(headers, "If-Match"),
        _header_value(headers, "If-None-Match"),
        _header_value(headers, "If-Unmodified-Since"),
        _header_value(headers, "Range"),
    )
    # Canonicalized x-ms headers already end in a newline. Do not add a
    # second separator before the canonicalized resource.
    string_to_sign = (
        "\n".join(standard_fields)
        + "\n"
        + _canonicalized_headers(headers)
        + _canonicalized_resource(url, account)
    )
    digest = hmac.new(key, string_to_sign.encode("utf-8"), hashlib.sha256).digest()
    return f"SharedKey {account}:{base64.b64encode(digest).decode('ascii')}"


def storage_authentication(url, method, payload, headers, client_id=None):
    """Build SAS, account-key, or managed-identity authentication for a request."""
    request_headers = dict(headers)
    sas = storage_sas()
    if sas:
        separator = "&" if "?" in url else "?"
        return url + separator + sas, request_headers

    shared_key = storage_shared_key()
    if shared_key:
        account, key = shared_key
        request_headers["x-ms-version"] = STORAGE_API_VERSION
        request_headers["x-ms-date"] = dt.datetime.now(dt.timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        request_headers["Authorization"] = _shared_key_authorization(
            url, method, payload, request_headers, account, key
        )
        return url, request_headers

    request_headers["Authorization"] = "Bearer " + access_token(client_id)
    request_headers["x-ms-version"] = STORAGE_API_VERSION
    request_headers["x-ms-date"] = dt.datetime.now(dt.timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    return url, request_headers


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


def _block_id(index):
    return base64.b64encode(f"{index:08d}".encode("ascii")).decode("ascii")


CHECKSUM_LENGTHS = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}


def checksum_metadata_headers(values=None, metadata_sha256=None):
    checksums = dict(values or {})
    if metadata_sha256 is not None:
        metadata_sha256 = str(metadata_sha256).lower()
        if (
            "sha256" in checksums
            and str(checksums["sha256"]).lower() != metadata_sha256
        ):
            raise ValueError("conflicting Azure Blob SHA-256 metadata values")
        checksums["sha256"] = metadata_sha256
    headers = {}
    for algorithm, value in checksums.items():
        if algorithm not in CHECKSUM_LENGTHS:
            raise ValueError(f"unsupported Azure Blob checksum metadata: {algorithm}")
        value = str(value).lower()
        length = CHECKSUM_LENGTHS[algorithm]
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
            raise ValueError(
                f"Azure Blob {algorithm.upper()} metadata must be "
                f"{length} hexadecimal characters"
            )
        headers[f"x-ms-meta-dl4j_{algorithm}"] = value
    return headers


def sha256_metadata_header(value):
    return checksum_metadata_headers(metadata_sha256=value)


def upload_block_blob(
    bucket,
    name,
    chunks,
    content_type,
    client_id=None,
    metadata_sha256=None,
    metadata_digests=None,
):
    """Upload chunks as uncommitted blocks, then atomically publish their list."""
    url = blob_url(bucket, name)
    block_ids = []
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        if index >= MAX_BLOCKS:
            raise RuntimeError(
                f"Azure block blob upload exceeds the {MAX_BLOCKS} block limit"
            )
        block_id = _block_id(index)
        query = urllib.parse.urlencode(
            (("comp", "block"), ("blockid", block_id))
        )
        request(
            url + "?" + query,
            method="PUT",
            data=chunk,
            headers={"Content-Type": "application/octet-stream"},
            authenticated=True,
            client_id=client_id,
        )
        block_ids.append(block_id)

    block_list = (
        '<?xml version="1.0" encoding="utf-8"?>\n<BlockList>'
        + "".join(f"<Latest>{block_id}</Latest>" for block_id in block_ids)
        + "</BlockList>"
    ).encode("utf-8")
    return request(
        url + "?comp=blocklist",
        method="PUT",
        data=block_list,
        headers={
            "Content-Type": "application/xml; charset=utf-8",
            "x-ms-blob-content-type": content_type,
            **checksum_metadata_headers(metadata_digests, metadata_sha256),
        },
        authenticated=True,
        client_id=client_id,
    )


def upload_bytes(
    bucket,
    name,
    data,
    content_type="application/octet-stream",
    client_id=None,
    metadata_sha256=None,
    metadata_digests=None,
):
    if len(data) > SINGLE_PUT_LIMIT:
        chunks = (
            data[offset:offset + BLOCK_UPLOAD_SIZE]
            for offset in range(0, len(data), BLOCK_UPLOAD_SIZE)
        )
        return upload_block_blob(
            bucket,
            name,
            chunks,
            content_type,
            client_id,
            metadata_sha256,
            metadata_digests,
        )
    return request(
        blob_url(bucket, name),
        method="PUT",
        data=data,
        headers={
            "Content-Type": content_type,
            "x-ms-blob-type": "BlockBlob",
            **checksum_metadata_headers(metadata_digests, metadata_sha256),
        },
        authenticated=True,
        client_id=client_id,
    )


def upload_file(
    bucket,
    name,
    path,
    content_type="application/octet-stream",
    client_id=None,
    metadata_sha256=None,
    metadata_digests=None,
):
    path = pathlib.Path(path)
    if path.stat().st_size <= SINGLE_PUT_LIMIT:
        return upload_bytes(
            bucket,
            name,
            path.read_bytes(),
            content_type,
            client_id,
            metadata_sha256,
            metadata_digests,
        )

    def chunks():
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(BLOCK_UPLOAD_SIZE)
                if not chunk:
                    return
                yield chunk

    return upload_block_blob(
        bucket,
        name,
        chunks(),
        content_type,
        client_id,
        metadata_sha256,
        metadata_digests,
    )


def download_bytes(bucket, name, client_id=None):
    return request(
        blob_url(bucket, name),
        authenticated=True,
        client_id=client_id,
    )


def download_file(bucket, name, path, client_id=None):
    """Stream a Blob to disk with authenticated ranged retries."""
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".partial")
    staged.unlink(missing_ok=True)
    url = blob_url(bucket, name)
    delay = 1.0
    etag = None
    attempts = 12
    try:
        for attempt in range(attempts):
            offset = staged.stat().st_size if staged.exists() else 0
            headers = {}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            if etag:
                headers["If-Match"] = etag
            request_url, headers = storage_authentication(
                url, "GET", None, headers, client_id
            )
            req = urllib.request.Request(request_url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    status = getattr(response, "status", response.getcode())
                    response_etag = response.headers.get("ETag")
                    if etag and response_etag and response_etag != etag:
                        raise RuntimeError(
                            f"Azure Blob changed during download: {name}"
                        )
                    etag = etag or response_etag
                    if offset and status != 206:
                        offset = 0
                    content_range = response.headers.get("Content-Range", "")
                    if offset and not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(
                            f"Azure Blob returned an invalid resume range for {name}: "
                            f"{content_range!r}"
                        )
                    mode = "ab" if offset else "wb"
                    with staged.open(mode) as output:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                    expected_size = None
                    if "/" in content_range:
                        total = content_range.rsplit("/", 1)[1]
                        if total.isdigit():
                            expected_size = int(total)
                    elif response.headers.get("Content-Length", "").isdigit():
                        expected_size = int(response.headers["Content-Length"])
                    actual_size = staged.stat().st_size
                    if expected_size is not None and actual_size != expected_size:
                        raise OSError(
                            f"incomplete Azure Blob download for {name}: "
                            f"expected {expected_size} bytes, got {actual_size}"
                        )
                os.replace(staged, destination)
                return
            except urllib.error.HTTPError as exc:
                body = exc.read(65536).decode("utf-8", "replace")
                if exc.code in (404, 410):
                    raise FileNotFoundError(url) from exc
                if exc.code in (401, 403):
                    _TOKEN["expires"] = 0
                retryable = exc.code in (401, 403, 408, 409, 429, 500, 502, 503, 504)
                if not retryable or attempt + 1 >= attempts:
                    raise RuntimeError(
                        f"Azure download failed ({exc.code}) {url}: {body}"
                    ) from exc
            except (
                http.client.HTTPException,
                urllib.error.URLError,
                TimeoutError,
                OSError,
            ) as exc:
                if attempt + 1 >= attempts:
                    raise RuntimeError(f"Azure download failed {url}: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 20)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


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
    upload_file(
        args.bucket,
        args.object,
        path,
        content_type,
        args.client_id,
        getattr(args, "metadata_sha256", None),
    )


def command_upload_json(args):
    payload = pathlib.Path(args.file).read_bytes() if args.file else sys.stdin.buffer.read()
    json.loads(payload.decode("utf-8"))
    upload_bytes(args.bucket, args.object, payload, "application/json", args.client_id)


def command_download(args):
    try:
        download_file(args.bucket, args.object, args.file, args.client_id)
    except FileNotFoundError:
        return getattr(args, "missing_exit_code", 1)
    return 0


def kill_switch_state(
    bucket,
    object_name,
    client_id,
    *,
    expected_epoch=None,
    force_only=False,
):
    try:
        value = json.loads(
            download_bytes(bucket, object_name, client_id).decode("utf-8")
        )
    except FileNotFoundError:
        print("kill switch object is missing", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"kill switch is unreadable: {exc}", file=sys.stderr, flush=True)
        return 2
    if not isinstance(value, dict):
        print(
            "kill switch JSON must be an object",
            file=sys.stderr,
            flush=True,
        )
        return 2
    enabled = value.get("enabled")
    if not isinstance(enabled, bool):
        print(
            "kill switch JSON does not contain a boolean enabled field",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if force_only:
        return 0 if enabled and value.get("force") is True else 1
    actual_epoch = value.get("controllerEpoch")
    if enabled:
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
    if expected_epoch and actual_epoch != expected_epoch:
        print(
            "kill switch controller epoch does not match this worker",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 1


def command_kill_enabled(args):
    state = kill_switch_state(
        args.bucket,
        args.object,
        args.client_id,
        expected_epoch=getattr(args, "controller_epoch", None),
    )
    if state != 1:
        return state
    emergency_object = getattr(args, "emergency_object", None)
    if emergency_object:
        return kill_switch_state(
            args.bucket,
            emergency_object,
            args.client_id,
            force_only=True,
        )
    return 1


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
    upload.add_argument("--metadata-sha256")
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
    download.add_argument("--missing-exit-code", type=int, default=1)
    add_identity_option(download)
    download.set_defaults(func=command_download)

    kill = commands.add_parser("kill-enabled")
    kill.add_argument("--bucket", required=True)
    kill.add_argument("--object", required=True)
    kill.add_argument("--emergency-object")
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
