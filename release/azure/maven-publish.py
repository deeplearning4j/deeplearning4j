#!/usr/bin/env python3
"""Publish one Azure worker Maven shard directly to the stable Blob repository.

Large Maven artifacts never transit through the controller.  The worker validates
its staged repository, creates primary-file checksums, uploads those files to the
stable prefix, and returns the generated Maven metadata as small accounting data.
The controller merges that metadata with the accumulated repository while holding
its publication lease.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import mimetypes
from pathlib import Path
from types import ModuleType
from typing import Any


CHECKSUM_ALGORITHMS = ("md5", "sha1", "sha256", "sha512")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(repository: Path, path: Path) -> str:
    relative = path.relative_to(repository)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe Maven repository path: {path}")
    value = relative.as_posix()
    if not (
        value.startswith("org/eclipse/deeplearning4j/")
        or value.startswith("org/nd4j/")
    ):
        raise RuntimeError(f"unexpected Maven namespace: {value}")
    return value


def prepare_repository(
    repository: Path,
    central_repository: ModuleType,
    release_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    central_repository.restore_embedded_component_poms(repository)
    central_repository.verify(repository, None, release_version, None)
    primary_files = list(central_repository.primary_files(repository))
    metadata_paths = list(
        central_repository.write_maven_metadata(repository, release_version)
    )
    central_repository.write_checksums(primary_files)

    published_files: list[dict[str, Any]] = []
    for primary in primary_files:
        candidates = [primary]
        candidates.extend(
            Path(str(primary) + f".{algorithm}")
            for algorithm in CHECKSUM_ALGORITHMS
        )
        for path in candidates:
            published_files.append(
                {
                    "path": relative_path(repository, path),
                    "sha256": sha256(path),
                    "size": path.stat().st_size,
                }
            )

    metadata_files = []
    for path in metadata_paths:
        payload = path.read_bytes()
        metadata_files.append(
            {
                "path": relative_path(repository, path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "contentBase64": base64.b64encode(payload).decode("ascii"),
            }
        )
    return published_files, metadata_files


def publish_files(
    repository: Path,
    cloud_io: ModuleType,
    bucket: str,
    repository_prefix: str,
    client_id: str,
    files: list[dict[str, Any]],
) -> None:
    prefix = repository_prefix.strip("/")
    if not prefix:
        raise RuntimeError("stable Maven repository prefix is empty")

    def upload(item: dict[str, Any]) -> str:
        relative = item["path"]
        path = repository / relative
        content_type = (
            mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        )
        cloud_io.upload_file(
            bucket,
            f"{prefix}/{relative}",
            path,
            content_type,
            client_id,
            metadata_sha256=item["sha256"],
        )
        return relative

    workers = max(1, min(4, len(files)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(upload, item) for item in files]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--repository", type=Path, required=True)
    root.add_argument("--central-repository", type=Path, required=True)
    root.add_argument("--cloud-io", type=Path, required=True)
    root.add_argument("--bucket", required=True)
    root.add_argument("--repository-prefix", required=True)
    root.add_argument("--client-id", required=True)
    root.add_argument("--run-id", required=True)
    root.add_argument("--shard", required=True)
    root.add_argument("--release-version", required=True)
    root.add_argument("--commit", required=True)
    root.add_argument("--accounting", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    repository = args.repository.resolve()
    if not repository.is_dir():
        raise RuntimeError(f"Maven repository directory is missing: {repository}")

    central_repository = load_module(
        args.central_repository.resolve(), "dl4j_central_repository"
    )
    cloud_io = load_module(args.cloud_io.resolve(), "dl4j_azure_cloud_io")
    published_files, metadata_files = prepare_repository(
        repository, central_repository, args.release_version
    )
    if not published_files:
        raise RuntimeError("Maven repository contains no publishable files")

    publish_files(
        repository,
        cloud_io,
        args.bucket,
        args.repository_prefix,
        args.client_id,
        published_files,
    )

    accounting = {
        "schemaVersion": 2,
        "mode": "stable-maven-upsert",
        "repositoryPrefix": args.repository_prefix.strip("/"),
        "runId": args.run_id,
        "shard": args.shard,
        "releaseVersion": args.release_version,
        "commit": args.commit,
        "publishedBlobs": sorted(item["path"] for item in published_files),
        "publishedFiles": sorted(published_files, key=lambda item: item["path"]),
        "metadataFiles": sorted(metadata_files, key=lambda item: item["path"]),
    }
    args.accounting.parent.mkdir(parents=True, exist_ok=True)
    args.accounting.write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
