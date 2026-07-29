#!/usr/bin/env python3
"""Tail a build log into CloudWatch Logs using only Python and AWS CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

MAX_MESSAGE_BYTES = 240 * 1024
MAX_BATCH_BYTES = 900 * 1024
MAX_BATCH_EVENTS = 9000


def chunks(message: str):
    data = message.encode("utf-8", errors="replace")
    while data:
        part = data[:MAX_MESSAGE_BYTES]
        while part:
            try:
                yield part.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                part = part[:exc.start]
        if not part:
            part = data[:1]
            yield part.decode("utf-8", errors="replace")
        data = data[len(part):]


def aws(region: str, *arguments: str, input_path: Path | None = None, check: bool = True, quiet: bool = False):
    # Ubuntu Jammy installs AWS CLI v1, which does not consistently support the
    # v2-only --no-cli-pager global option. AWS_PAGER works with both versions.
    command = ["aws", "--region", region, *arguments]
    if input_path is not None:
        command.extend(["--log-events", f"file://{input_path}"])
    environment = os.environ.copy()
    environment["AWS_PAGER"] = ""
    return subprocess.run(
        command, check=check, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL if quiet else None, env=environment,
    )


def create_stream(region: str, group: str, stream: str) -> None:
    aws(region, "logs", "create-log-stream", "--log-group-name", group, "--log-stream-name", stream, check=False, quiet=True)


def send(region: str, group: str, stream: str, messages: list[str]) -> None:
    if not messages:
        return
    events = []
    size = 0
    now = int(time.time() * 1000)
    for message in messages:
        for part in chunks(message):
            event_size = len(part.encode("utf-8")) + 26
            if events and (len(events) >= MAX_BATCH_EVENTS or size + event_size > MAX_BATCH_BYTES):
                put(region, group, stream, events)
                events, size = [], 0
            events.append({"timestamp": now, "message": part})
            size += event_size
            now += 1
    put(region, group, stream, events)


def put(region: str, group: str, stream: str, events: list[dict]) -> None:
    if not events:
        return
    with tempfile.TemporaryDirectory(prefix="dl4j-log-events-") as temporary:
        payload = Path(temporary) / "events.json"
        payload.write_text(json.dumps(events, separators=(",", ":")), encoding="utf-8")
        delay = 1
        for attempt in range(6):
            result = aws(
                region, "logs", "put-log-events", "--log-group-name", group,
                "--log-stream-name", stream, input_path=payload, check=False,
            )
            if result.returncode == 0:
                return
            if attempt == 5:
                raise RuntimeError("CloudWatch put-log-events failed after retries")
            time.sleep(delay)
            delay = min(delay * 2, 15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args()

    create_stream(args.region, args.group, args.stream)
    offset = 0
    partial = ""
    last_heartbeat = 0.0
    stopping_since = None
    while True:
        messages = []
        data = b""
        if args.file.exists():
            size = args.file.stat().st_size
            if size < offset:
                offset, partial = 0, ""
            with args.file.open("rb") as source:
                source.seek(offset)
                data = source.read()
                offset = source.tell()
            if data:
                text = partial + data.decode("utf-8", errors="replace")
                lines = text.splitlines(keepends=True)
                partial = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    partial = lines.pop()
                messages.extend(line.rstrip("\r\n") for line in lines)
        now = time.monotonic()
        if now - last_heartbeat >= args.heartbeat_seconds:
            messages.append(f"[dl4j-heartbeat] forwardedBytes={offset}")
            last_heartbeat = now
        send(args.region, args.group, args.stream, messages)
        if args.stop_file.exists():
            if stopping_since is None:
                stopping_since = now
            if not data:
                if partial:
                    send(args.region, args.group, args.stream, [partial])
                send(args.region, args.group, args.stream, ["[dl4j-log-forwarder] stream complete"])
                return
            if now - stopping_since > 30:
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
