#!/usr/bin/env python3
"""Content-addressed cloud cache for immutable release-worker dependencies.

Each logical dependency has an independently versioned identity. A publisher
uploads the archive first and its small identity index last. Restores verify the
index, compressed archive, member safety, expanded size, and member count before
atomically exposing the destination directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


SCHEMA_VERSION = 1
MAX_MEMBERS = 1_000_000
MAX_EXPANDED_BYTES = 64 * 1024 * 1024 * 1024
MISS_EXIT_CODE = 3


def canonical_identity(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def validate_name(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-._" for character in value):
        raise ValueError(f"unsafe dependency cache name: {value!r}")
    return value


def validate_identity(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("dependency cache identity must be a lowercase SHA-256 digest")
    return value


def cloud_command(
    cloud_io: Path,
    action: str,
    *,
    bucket: str,
    object_name: str,
    file: Path,
    client_id: str | None,
    missing_exit_code: int | None = None,
) -> list[str]:
    command = [
        os.environ.get("PYTHON") or sys.executable,
        str(cloud_io),
        action,
        "--bucket",
        bucket,
        "--object",
        object_name,
        "--file",
        str(file),
    ]
    if client_id:
        command.extend(["--client-id", client_id])
    if missing_exit_code is not None:
        command.extend(["--missing-exit-code", str(missing_exit_code)])
    return command


def cloud_download(
    cloud_io: Path,
    *,
    bucket: str,
    object_name: str,
    file: Path,
    client_id: str | None,
    optional: bool,
) -> bool:
    file.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cloud_command(
            cloud_io,
            "download",
            bucket=bucket,
            object_name=object_name,
            file=file,
            client_id=client_id,
            missing_exit_code=MISS_EXIT_CODE if optional else None,
        ),
        check=False,
    )
    if result.returncode == 0:
        return True
    file.unlink(missing_ok=True)
    if optional and result.returncode == MISS_EXIT_CODE:
        return False
    result.check_returncode()
    raise AssertionError("unreachable")


def cloud_upload(
    cloud_io: Path,
    *,
    bucket: str,
    object_name: str,
    file: Path,
    client_id: str | None,
) -> None:
    subprocess.run(
        cloud_command(
            cloud_io,
            "upload",
            bucket=bucket,
            object_name=object_name,
            file=file,
            client_id=client_id,
        ),
        check=True,
    )


def index_object(prefix: str, name: str, identity: str) -> str:
    return f"{prefix.strip('/')}/indexes/{name}/{identity}.json"


def archive_object(prefix: str, name: str, digest: str) -> str:
    return f"{prefix.strip('/')}/archives/{name}/sha256/{digest}.tar.gz"


def normalized_link_path(member: tarfile.TarInfo) -> PurePosixPath | None:
    if not (member.issym() or member.islnk()):
        return PurePosixPath(member.name)
    link = PurePosixPath(member.linkname)
    if link.is_absolute():
        return None
    parent = PurePosixPath(member.name).parent if member.issym() else PurePosixPath()
    resolved: list[str] = []
    for part in (*parent.parts, *link.parts):
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                return False
            resolved.pop()
        else:
            resolved.append(part)
    if not resolved or resolved[0] != "payload":
        return None
    return PurePosixPath(*resolved)


def safe_link(member: tarfile.TarInfo) -> bool:
    return normalized_link_path(member) is not None


def validate_archive(bundle: tarfile.TarFile) -> tuple[int, int]:
    member_count = 0
    expanded_bytes = 0
    names: set[str] = set()
    for member in bundle.getmembers():
        member_count += 1
        if member_count > MAX_MEMBERS:
            raise RuntimeError("dependency cache archive exceeds the member limit")
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "payload"
            or ".." in path.parts
            or member.name in names
        ):
            raise RuntimeError(f"unsafe dependency cache archive path: {member.name!r}")
        if not (
            member.isdir()
            or member.isfile()
            or member.issym()
            or member.islnk()
        ):
            raise RuntimeError(f"unsafe dependency cache archive member: {member.name!r}")
        if not safe_link(member):
            raise RuntimeError(f"unsafe dependency cache archive link: {member.name!r}")
        names.add(member.name)
        if member.isfile():
            expanded_bytes += member.size
            if expanded_bytes > MAX_EXPANDED_BYTES:
                raise RuntimeError("dependency cache archive exceeds the expanded-size limit")
    return member_count, expanded_bytes


def archive_source(source: Path, archive: Path) -> tuple[int, int]:
    if not source.is_dir():
        raise RuntimeError(f"dependency cache source directory is missing: {source}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    source_root = source.resolve()

    def normalize_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
        if member.issym() and PurePosixPath(member.linkname).is_absolute():
            target = Path(member.linkname).resolve()
            try:
                relative_target = target.relative_to(source_root)
            except ValueError:
                return member
            archive_target = PurePosixPath("payload", *relative_target.parts)
            member.linkname = posixpath.relpath(
                archive_target.as_posix(),
                PurePosixPath(member.name).parent.as_posix(),
            )
        return member

    with tarfile.open(archive, mode="w:gz", compresslevel=1) as bundle:
        bundle.add(source_root, arcname="payload", recursive=True, filter=normalize_member)
    with tarfile.open(archive, mode="r:gz") as bundle:
        return validate_archive(bundle)


def load_index(path: Path, *, prefix: str, name: str, identity: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schemaVersion",
        "name",
        "identity",
        "archiveObject",
        "archiveSize",
        "archiveSha256",
        "memberCount",
        "expandedBytes",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["schemaVersion"] != SCHEMA_VERSION
        or value["name"] != name
        or value["identity"] != identity
        or not isinstance(value["archiveSize"], int)
        or value["archiveSize"] < 0
        or not isinstance(value["memberCount"], int)
        or value["memberCount"] < 1
        or not isinstance(value["expandedBytes"], int)
        or value["expandedBytes"] < 0
        or not isinstance(value["archiveSha256"], str)
        or len(value["archiveSha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["archiveSha256"])
        or value["archiveObject"] != archive_object(prefix, name, value["archiveSha256"])
    ):
        raise RuntimeError("dependency cache index has an invalid contract")
    return value


def restore(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    identity = validate_identity(args.identity)
    cloud_io = args.cloud_io.resolve()
    destination = Path(os.path.abspath(args.destination))
    marker = destination / ".dl4j-dependency-cache.json"
    if marker.is_file():
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        if value.get("schemaVersion") == SCHEMA_VERSION and value.get("identity") == identity:
            print(f"[dl4j-toolchain-cache] hit-local name={name} identity={identity}", flush=True)
            return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.dl4j-restore-",
        dir=destination.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        index_path = temporary_root / "index.json"
        if not cloud_download(
            cloud_io,
            bucket=args.bucket,
            object_name=index_object(args.prefix, name, identity),
            file=index_path,
            client_id=args.client_id,
            optional=True,
        ):
            print(f"[dl4j-toolchain-cache] miss name={name} identity={identity}", flush=True)
            return MISS_EXIT_CODE
        index = load_index(index_path, prefix=args.prefix, name=name, identity=identity)
        archive = temporary_root / "archive.tar.gz"
        if not cloud_download(
            cloud_io,
            bucket=args.bucket,
            object_name=str(index["archiveObject"]),
            file=archive,
            client_id=args.client_id,
            optional=False,
        ):
            raise RuntimeError(f"dependency cache archive is missing for {name}/{identity}")
        size, digest = file_digest(archive)
        if size != index["archiveSize"] or digest != index["archiveSha256"]:
            raise RuntimeError(f"dependency cache archive attestation mismatch for {name}/{identity}")

        extracted = temporary_root / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, mode="r:gz") as bundle:
            member_count, expanded_bytes = validate_archive(bundle)
            if (
                member_count != index["memberCount"]
                or expanded_bytes != index["expandedBytes"]
            ):
                raise RuntimeError(
                    f"dependency cache expanded attestation mismatch for {name}/{identity}"
                )
            bundle.extractall(extracted)
        payload = extracted / "payload"
        marker_payload = {
            "schemaVersion": SCHEMA_VERSION,
            "name": name,
            "identity": identity,
            "archiveSha256": digest,
        }
        (payload / ".dl4j-dependency-cache.json").write_text(
            json.dumps(marker_payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged = destination.with_name(destination.name + ".dl4j-new")
        previous = destination.with_name(destination.name + ".dl4j-old")
        shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(previous, ignore_errors=True)
        os.replace(payload, staged)
        if destination.exists() or destination.is_symlink():
            os.replace(destination, previous)
        os.replace(staged, destination)
        shutil.rmtree(previous, ignore_errors=True)
    print(f"[dl4j-toolchain-cache] restored name={name} identity={identity}", flush=True)
    return 0


def publish(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    identity = validate_identity(args.identity)
    source = args.source.resolve()
    cloud_io = args.cloud_io.resolve()
    with tempfile.TemporaryDirectory(prefix=f"dl4j-{name}-publish-") as temporary:
        root = Path(temporary)
        archive = root / "archive.tar.gz"
        member_count, expanded_bytes = archive_source(source, archive)
        size, digest = file_digest(archive)
        object_name = archive_object(args.prefix, name, digest)
        cloud_upload(
            cloud_io,
            bucket=args.bucket,
            object_name=object_name,
            file=archive,
            client_id=args.client_id,
        )
        index = {
            "schemaVersion": SCHEMA_VERSION,
            "name": name,
            "identity": identity,
            "archiveObject": object_name,
            "archiveSize": size,
            "archiveSha256": digest,
            "memberCount": member_count,
            "expandedBytes": expanded_bytes,
        }
        index_path = root / "index.json"
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cloud_upload(
            cloud_io,
            bucket=args.bucket,
            object_name=index_object(args.prefix, name, identity),
            file=index_path,
            client_id=args.client_id,
        )
    print(
        f"[dl4j-toolchain-cache] published name={name} identity={identity} "
        f"archiveSha256={digest} archiveBytes={size} expandedBytes={expanded_bytes}",
        flush=True,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command_name in ("restore", "publish"):
        command = commands.add_parser(command_name)
        command.add_argument("--cloud-io", type=Path, required=True)
        command.add_argument("--bucket", required=True)
        command.add_argument("--prefix", required=True)
        command.add_argument("--name", required=True)
        command.add_argument("--identity", required=True)
        command.add_argument("--client-id")
    restore_command = commands.choices["restore"]
    restore_command.add_argument("--destination", type=Path, required=True)
    restore_command.set_defaults(func=restore)
    publish_command = commands.choices["publish"]
    publish_command.add_argument("--source", type=Path, required=True)
    publish_command.set_defaults(func=publish)
    return root


def main() -> None:
    arguments = parser().parse_args()
    raise SystemExit(arguments.func(arguments))


if __name__ == "__main__":
    main()
