#!/usr/bin/env python3
"""Merge, validate, sign, bundle, and upload prebuilt Maven repository shards."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Iterable

PRIMARY_SUFFIXES = (".pom", ".jar", ".aar", ".war", ".module")
CHECKSUMS = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}  # nosec: Central requires MD5/SHA1 metadata
MAX_BUNDLE_BYTES = 1_000_000_000


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = CHECKSUMS[algorithm]()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_relative(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {path}")
    return relative


def extract_input(source: Path, destination: Path) -> Path:
    if source.is_dir():
        return source
    target = destination / source.name.replace(".", "-")
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                relative = safe_relative(member.filename)
                if member.is_dir():
                    continue
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as incoming, output.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                relative = safe_relative(member.name)
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                incoming = archive.extractfile(member)
                if incoming is None:
                    raise ValueError(f"unable to read archive member: {member.name}")
                with incoming, output.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    else:
        raise ValueError(f"unsupported shard input: {source}")
    return target


def repository_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.name in {"shard-manifest.json", "release-build-manifest.json"}:
            continue
        if relative.parts[:3] not in (("org", "eclipse", "deeplearning4j"), ("org", "nd4j")):
            continue
        yield path


def merge(inputs: list[Path], output: Path, manifest_path: Path, release_version: str, commit: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ownership: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory(prefix="dl4j-central-merge-") as temporary:
        scratch = Path(temporary)
        for source in inputs:
            root = extract_input(source, scratch)
            candidates = list(repository_files(root))
            if not candidates:
                raise ValueError(f"Maven shard has no DL4J repository files: {source}")
            for path in candidates:
                relative = path.relative_to(root)
                destination = output / relative
                key = relative.as_posix()
                ownership.setdefault(key, []).append(source.name)
                if destination.exists():
                    if digest(destination) != digest(path):
                        raise ValueError(f"conflicting duplicate Maven path {key} from {source}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    files = [{"path": path.relative_to(output).as_posix(), "sha256": digest(path), "size": path.stat().st_size, "shards": ownership[path.relative_to(output).as_posix()]} for path in repository_files(output)]
    manifest = {"schemaVersion": 1, "releaseVersion": release_version, "commit": commit, "workloads": ["maven", "sdk"], "files": files}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(f"{digest(manifest_path)}  {manifest_path.name}\n", encoding="ascii")
    return manifest


def verify_release_assets(directory: Path, manifest_path: Path, version: str, commit: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("releaseVersion") != version:
        raise ValueError(f"release version mismatch: {manifest.get('releaseVersion')} != {version}")
    if manifest.get("commit") != commit:
        raise ValueError(f"commit mismatch: {manifest.get('commit')} != {commit}")
    if set(manifest.get("workloads", [])) != {"maven", "sdk"}:
        raise ValueError("release manifest must attest both maven and sdk workloads")
    assets = manifest.get("assets", [])
    expected: dict[str, dict] = {}
    for item in assets:
        name = item.get("fileName", "")
        if not name or Path(name).name != name or name in expected:
            raise ValueError(f"invalid or duplicate release asset name: {name!r}")
        expected[name] = item
    maven_assets = [name for name in expected if name.startswith("maven-repository-") and name.endswith(".tar.gz")]
    sdk_assets = [name for name in expected if name.startswith("sdk-assets-") and name.endswith(".tar.gz")]
    if not maven_assets or not sdk_assets:
        raise ValueError("release manifest must contain both Maven repository and SDK asset archives")
    for name, item in expected.items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"release asset is missing: {name}")
        if digest(path) != item.get("sha256"):
            raise ValueError(f"release asset checksum mismatch: {name}")
        if path.stat().st_size != item.get("size"):
            raise ValueError(f"release asset size mismatch: {name}")


def verify(repository: Path, manifest_path: Path | None, version: str | None, commit: str | None) -> None:
    if manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if version and manifest.get("releaseVersion") != version:
            raise ValueError(f"release version mismatch: {manifest.get('releaseVersion')} != {version}")
        if commit and manifest.get("commit") != commit:
            raise ValueError(f"commit mismatch: {manifest.get('commit')} != {commit}")
        expected = {item["path"]: item["sha256"] for item in manifest.get("files", [])}
        actual = {path.relative_to(repository).as_posix(): digest(path) for path in repository_files(repository)}
        if expected != actual:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
            raise ValueError(f"manifest mismatch; missing={missing}, extra={extra}, changed={changed}")
    version_dirs: dict[Path, list[Path]] = {}
    for path in repository_files(repository):
        if path.name.endswith((".asc", ".md5", ".sha1", ".sha256", ".sha512")):
            continue
        if version and version not in path.parts:
            raise ValueError(f"unexpected version path in repository: {path.relative_to(repository)}")
        version_dirs.setdefault(path.parent, []).append(path)
    if not version_dirs:
        raise ValueError("repository contains no publishable components")
    for directory, files in version_dirs.items():
        suffixes = {path.suffix for path in files}
        if ".pom" not in suffixes:
            raise ValueError(f"component is missing its POM: {directory.relative_to(repository)}")
        jars = [path for path in files if path.suffix == ".jar"]
        non_metadata_jars = [path for path in jars if not path.name.endswith(("-sources.jar", "-javadoc.jar"))]
        if jars and not non_metadata_jars:
            raise ValueError(f"component has no main/classifier JAR: {directory.relative_to(repository)}")


def primary_files(repository: Path) -> list[Path]:
    return [path for path in repository_files(repository) if path.suffix in PRIMARY_SUFFIXES and not path.name.endswith(".asc")]


def materialize_test_repository(
    inputs: list[Path], output: Path, manifest_path: Path, release_version: str, commit: str,
) -> dict:
    """Merge shards into a checksum-complete Maven layout suitable for local testing."""
    scratch_manifest = manifest_path.with_name(manifest_path.name + ".merge")
    merge(inputs, output, scratch_manifest, release_version, commit)
    verify(output, scratch_manifest, release_version, commit)
    for path in primary_files(output):
        for algorithm in ("md5", "sha1", "sha256", "sha512"):
            Path(str(path) + f".{algorithm}").write_text(digest(path, algorithm) + "\n", encoding="ascii")
    files = [
        {"path": path.relative_to(output).as_posix(), "sha256": digest(path), "size": path.stat().st_size}
        for path in repository_files(output)
    ]
    manifest = {
        "schemaVersion": 1,
        "layout": "maven2",
        "releaseVersion": release_version,
        "commit": commit,
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(f"{digest(manifest_path)}  {manifest_path.name}\n", encoding="ascii")
    scratch_manifest.unlink(missing_ok=True)
    Path(str(scratch_manifest) + ".sha256").unlink(missing_ok=True)
    return manifest


def sign_bundle(repository: Path, output: Path, gpg_executable: str = "gpg") -> None:
    for path in primary_files(repository):
        signature = Path(str(path) + ".asc")
        command = [gpg_executable, "--batch", "--yes", "--armor", "--detach-sign", "--output", str(signature)]
        passphrase = os.environ.get("MAVEN_GPG_PASSPHRASE")
        input_bytes = None
        if passphrase:
            command.extend(["--pinentry-mode", "loopback", "--passphrase-fd", "0"])
            input_bytes = (passphrase + "\n").encode()
        subprocess.run([*command, str(path)], input=input_bytes, check=True)
        for algorithm in CHECKSUMS:
            Path(str(path) + f".{algorithm}").write_text(digest(path, algorithm) + "\n", encoding="ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(repository.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(repository).as_posix())
    if output.stat().st_size >= MAX_BUNDLE_BYTES:
        output.unlink()
        raise ValueError("Central bundle is 1 GB or larger; it cannot be uploaded as one deployment")


def snapshot_deploy_commands(
    repository: Path,
    version: str,
    repository_id: str,
    url: str,
    maven_executable: str = "mvn",
) -> list[list[str]]:
    """Create deploy-file commands for a prebuilt snapshot repository without rebuilding it."""
    if not version.endswith("-SNAPSHOT"):
        raise ValueError(f"snapshot publication requires a -SNAPSHOT version: {version}")
    commands: list[list[str]] = []
    supported = {".jar", ".aar", ".war"}
    for pom in sorted(repository.rglob(f"*-{version}.pom")):
        if pom.parent.name != version:
            continue
        artifact_id = pom.parent.parent.name
        base = f"{artifact_id}-{version}"
        main_files = [pom.parent / f"{base}{suffix}" for suffix in sorted(supported)]
        main = next((path for path in main_files if path.is_file()), pom)
        packaging = main.suffix.lstrip(".") if main != pom else "pom"
        attachments: list[tuple[Path, str, str]] = []
        unsupported = []
        for path in sorted(pom.parent.iterdir()):
            if not path.is_file() or path in {pom, main}:
                continue
            if path.name.endswith((".asc", ".md5", ".sha1", ".sha256", ".sha512")):
                continue
            suffix = path.suffix
            if suffix not in supported or not path.name.startswith(base + "-"):
                if suffix in PRIMARY_SUFFIXES:
                    unsupported.append(path.name)
                continue
            classifier = path.name[len(base) + 1:-len(suffix)]
            if not classifier:
                unsupported.append(path.name)
                continue
            attachments.append((path, classifier, suffix.lstrip(".")))
        if unsupported:
            raise ValueError(f"snapshot component {artifact_id} has unsupported prebuilt files: {unsupported}")
        command = [
            maven_executable, "--batch-mode",
            "org.apache.maven.plugins:maven-deploy-plugin:3.1.4:deploy-file",
            f"-DrepositoryId={repository_id}", f"-Durl={url}",
            f"-Dfile={main}", f"-DpomFile={pom}", f"-Dpackaging={packaging}",
            "-DgeneratePom=false", "-DretryFailedDeploymentCount=3",
        ]
        if attachments:
            command.extend((
                "-Dfiles=" + ",".join(str(item[0]) for item in attachments),
                "-Dclassifiers=" + ",".join(item[1] for item in attachments),
                "-Dtypes=" + ",".join(item[2] for item in attachments),
            ))
        commands.append(command)
    if not commands:
        raise ValueError(f"repository contains no deployable {version} snapshot components")
    return commands


def deploy_snapshot(repository: Path, version: str, repository_id: str, url: str, maven_executable: str = "mvn") -> None:
    for index, command in enumerate(snapshot_deploy_commands(repository, version, repository_id, url, maven_executable), start=1):
        print(json.dumps({"snapshotComponent": index, "command": command}), flush=True)
        subprocess.run(command, check=True)


def authorization(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Bearer {token}"


def request(url: str, auth: str, data: bytes | None = None, method: str = "POST", content_type: str | None = None) -> bytes:
    headers = {"Authorization": auth, "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method), timeout=120) as response:
        return response.read()


def upload(bundle: Path, username: str, password: str, automatic: bool, wait_seconds: int) -> str:
    boundary = f"----dl4j-{uuid.uuid4().hex}"
    payload = b"".join((
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"bundle\"; filename=\"{bundle.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode(),
        bundle.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ))
    publishing_type = "AUTOMATIC" if automatic else "USER_MANAGED"
    url = "https://central.sonatype.com/api/v1/publisher/upload?" + urllib.parse.urlencode({"name": bundle.stem, "publishingType": publishing_type})
    auth = authorization(username, password)
    deployment_id = request(url, auth, payload, content_type=f"multipart/form-data; boundary={boundary}").decode().strip()
    deadline = time.time() + wait_seconds
    while True:
        status_raw = request("https://central.sonatype.com/api/v1/publisher/status?" + urllib.parse.urlencode({"id": deployment_id}), auth)
        status = json.loads(status_raw)
        state = status.get("deploymentState")
        print(json.dumps(status, sort_keys=True))
        if state in {"PUBLISHED", "VALIDATED"}:
            return deployment_id
        if state == "FAILED":
            raise RuntimeError(f"Central deployment failed: {status}")
        if time.time() >= deadline:
            raise TimeoutError(f"Central deployment {deployment_id} remained in {state}")
        time.sleep(10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge_cmd = sub.add_parser("merge")
    merge_cmd.add_argument("--input", type=Path, action="append", required=True)
    merge_cmd.add_argument("--output", type=Path, required=True)
    merge_cmd.add_argument("--manifest", type=Path, required=True)
    merge_cmd.add_argument("--release-version", required=True)
    merge_cmd.add_argument("--commit", required=True)
    materialize_cmd = sub.add_parser("materialize-test-repository")
    materialize_cmd.add_argument("--input", type=Path, action="append", required=True)
    materialize_cmd.add_argument("--output", type=Path, required=True)
    materialize_cmd.add_argument("--manifest", type=Path, required=True)
    materialize_cmd.add_argument("--release-version", required=True)
    materialize_cmd.add_argument("--commit", required=True)
    assets_cmd = sub.add_parser("verify-release-assets")
    assets_cmd.add_argument("--directory", type=Path, required=True)
    assets_cmd.add_argument("--manifest", type=Path, required=True)
    assets_cmd.add_argument("--release-version", required=True)
    assets_cmd.add_argument("--commit", required=True)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--repository", type=Path, required=True)
    verify_cmd.add_argument("--manifest", type=Path)
    verify_cmd.add_argument("--release-version")
    verify_cmd.add_argument("--commit")
    sign_cmd = sub.add_parser("sign-bundle")
    sign_cmd.add_argument("--repository", type=Path, required=True)
    sign_cmd.add_argument("--output", type=Path, required=True)
    sign_cmd.add_argument("--gpg-executable", default="gpg")
    upload_cmd = sub.add_parser("upload")
    upload_cmd.add_argument("--bundle", type=Path, required=True)
    upload_cmd.add_argument("--automatic", action="store_true")
    upload_cmd.add_argument("--wait-seconds", type=int, default=3600)
    snapshot_cmd = sub.add_parser("deploy-snapshot")
    snapshot_cmd.add_argument("--repository", type=Path, required=True)
    snapshot_cmd.add_argument("--release-version", required=True)
    snapshot_cmd.add_argument("--repository-id", default="central-portal-snapshots")
    snapshot_cmd.add_argument("--url", default="https://central.sonatype.com/repository/maven-snapshots/")
    snapshot_cmd.add_argument("--maven-executable", default="mvn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "merge":
        merge(args.input, args.output, args.manifest, args.release_version, args.commit)
    elif args.command == "materialize-test-repository":
        materialize_test_repository(args.input, args.output, args.manifest, args.release_version, args.commit)
    elif args.command == "verify-release-assets":
        verify_release_assets(args.directory, args.manifest, args.release_version, args.commit)
    elif args.command == "verify":
        verify(args.repository, args.manifest, args.release_version, args.commit)
    elif args.command == "sign-bundle":
        sign_bundle(args.repository, args.output, args.gpg_executable)
    elif args.command == "upload":
        username = os.environ.get("CENTRAL_SONATYPE_TOKEN_USERNAME")
        password = os.environ.get("CENTRAL_SONATYPE_TOKEN_PASSWORD")
        if not username or not password:
            raise SystemExit("CENTRAL_SONATYPE_TOKEN_USERNAME and CENTRAL_SONATYPE_TOKEN_PASSWORD are required")
        deployment_id = upload(args.bundle, username, password, args.automatic, args.wait_seconds)
        print(deployment_id)
    elif args.command == "deploy-snapshot":
        if not os.environ.get("CENTRAL_SONATYPE_TOKEN_USERNAME") or not os.environ.get("CENTRAL_SONATYPE_TOKEN_PASSWORD"):
            raise SystemExit("CENTRAL_SONATYPE_TOKEN_USERNAME and CENTRAL_SONATYPE_TOKEN_PASSWORD are required")
        deploy_snapshot(args.repository, args.release_version, args.repository_id, args.url, args.maven_executable)


if __name__ == "__main__":
    main()
