#!/usr/bin/env python3
"""Small, dependency-free GCP worker transport.

It uses the VM's attached service-account token from the metadata server.  Keeping
this separate from the controller means Linux, Windows, and TPU startup scripts
can publish logs/status and observe the kill switch before pip/gcloud exist.
"""

import argparse
import datetime as dt
import json
import mimetypes
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

METADATA = "http://metadata.google.internal/computeMetadata/v1"
SCOPES = "https://www.googleapis.com/auth/cloud-platform"
_TOKEN = {"value": None, "expires": 0.0}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def request(url, *, method="GET", data=None, headers=None, authenticated=False, retries=6):
    headers = dict(headers or {})
    if authenticated:
        headers["Authorization"] = "Bearer " + access_token()
    payload = data
    if isinstance(data, str):
        payload = data.encode("utf-8")
    delay = 1.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code in (404, 410):
                raise FileNotFoundError(url) from exc
            if exc.code == 401 and authenticated:
                _TOKEN["expires"] = 0
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == retries:
                raise RuntimeError(f"GCP request failed ({exc.code}) {url}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"GCP request failed {url}: {exc}") from exc
        time.sleep(delay)
        delay = min(delay * 2, 20)
    raise AssertionError("unreachable")


def metadata(path):
    raw = request(f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
    return raw.decode("utf-8")


def access_token():
    now = time.time()
    if _TOKEN["value"] and _TOKEN["expires"] > now + 120:
        return _TOKEN["value"]
    raw = metadata("instance/service-accounts/default/token")
    value = json.loads(raw)
    _TOKEN["value"] = value["access_token"]
    _TOKEN["expires"] = now + int(value.get("expires_in", 3000))
    return _TOKEN["value"]


def object_url(bucket, name, *, upload=False):
    quoted_bucket = urllib.parse.quote(bucket, safe="")
    quoted_name = urllib.parse.quote(name, safe="")
    if upload:
        return f"https://storage.googleapis.com/upload/storage/v1/b/{quoted_bucket}/o?uploadType=media&name={quoted_name}"
    return f"https://storage.googleapis.com/storage/v1/b/{quoted_bucket}/o/{quoted_name}?alt=media"


def upload_bytes(bucket, name, data, content_type="application/octet-stream"):
    return request(
        object_url(bucket, name, upload=True),
        method="POST",
        data=data,
        headers={"Content-Type": content_type},
        authenticated=True,
    )


def download_bytes(bucket, name):
    return request(object_url(bucket, name), authenticated=True)


def write_log_entries(project, log_id, run_id, shard, lines):
    if not lines:
        return
    encoded_log = urllib.parse.quote(log_id, safe="")
    body = {
        "logName": f"projects/{project}/logs/{encoded_log}",
        "resource": {"type": "global", "labels": {"project_id": project}},
        "labels": {"dl4j_run_id": run_id, "dl4j_shard": shard},
        "entries": [
            {"timestamp": utc_now(), "textPayload": line.rstrip("\r\n")[:240000]}
            for line in lines
        ],
    }
    request(
        "https://logging.googleapis.com/v2/entries:write",
        method="POST",
        data=json.dumps(body),
        headers={"Content-Type": "application/json"},
        authenticated=True,
    )


def command_upload(args):
    path = pathlib.Path(args.file)
    content_type = args.content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    upload_bytes(args.bucket, args.object, path.read_bytes(), content_type)


def command_upload_json(args):
    payload = pathlib.Path(args.file).read_bytes() if args.file else sys.stdin.buffer.read()
    json.loads(payload.decode("utf-8"))
    upload_bytes(args.bucket, args.object, payload, "application/json")


def command_download(args):
    pathlib.Path(args.file).write_bytes(download_bytes(args.bucket, args.object))


def command_kill_enabled(args):
    try:
        value = json.loads(download_bytes(args.bucket, args.object).decode("utf-8"))
    except FileNotFoundError:
        print("kill switch object is missing", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(f"kill switch is unreadable: {exc}", file=sys.stderr, flush=True)
        return 2
    if value.get("enabled") is True:
        return 0
    if value.get("enabled") is False:
        return 1
    print("kill switch JSON does not contain a boolean enabled field", file=sys.stderr, flush=True)
    return 2


def command_log(args):
    write_log_entries(args.project or metadata("project/project-id"), args.log_id, args.run_id, args.shard, [args.message])


def command_forward(args):
    project = args.project or metadata("project/project-id")
    path = pathlib.Path(args.file)
    stop_path = pathlib.Path(args.stop_file) if args.stop_file else None
    position = 0
    idle_after_stop = 0
    while True:
        lines = []
        if path.exists():
            size = path.stat().st_size
            if size < position:
                position = 0
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(position)
                for _ in range(100):
                    line = stream.readline()
                    if not line:
                        break
                    lines.append(line)
                position = stream.tell()
        if lines:
            try:
                write_log_entries(project, args.log_id, args.run_id, args.shard, lines)
            except Exception as exc:  # logging must not kill the build
                print(f"[dl4j-cloud-io] log forward failed: {exc}", file=sys.stderr, flush=True)
            idle_after_stop = 0
        elif stop_path and stop_path.exists():
            idle_after_stop += 1
            if idle_after_stop >= 2:
                return 0
        time.sleep(args.interval)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload")
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--object", required=True)
    upload.add_argument("--file", required=True)
    upload.add_argument("--content-type")
    upload.set_defaults(func=command_upload)

    upload_json = commands.add_parser("upload-json")
    upload_json.add_argument("--bucket", required=True)
    upload_json.add_argument("--object", required=True)
    upload_json.add_argument("--file")
    upload_json.set_defaults(func=command_upload_json)

    download = commands.add_parser("download")
    download.add_argument("--bucket", required=True)
    download.add_argument("--object", required=True)
    download.add_argument("--file", required=True)
    download.set_defaults(func=command_download)

    kill = commands.add_parser("kill-enabled")
    kill.add_argument("--bucket", required=True)
    kill.add_argument("--object", required=True)
    kill.set_defaults(func=command_kill_enabled)

    log = commands.add_parser("log")
    log.add_argument("--project")
    log.add_argument("--log-id", required=True)
    log.add_argument("--run-id", required=True)
    log.add_argument("--shard", required=True)
    log.add_argument("--message", required=True)
    log.set_defaults(func=command_log)

    forward = commands.add_parser("forward")
    forward.add_argument("--project")
    forward.add_argument("--log-id", required=True)
    forward.add_argument("--run-id", required=True)
    forward.add_argument("--shard", required=True)
    forward.add_argument("--file", required=True)
    forward.add_argument("--stop-file")
    forward.add_argument("--interval", type=float, default=2.0)
    forward.set_defaults(func=command_forward)
    return root


def main():
    args = parser().parse_args()
    result = args.func(args)
    raise SystemExit(0 if result is None else result)


if __name__ == "__main__":
    main()
