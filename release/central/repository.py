#!/usr/bin/env python3
"""Merge, validate, sign, bundle, and upload prebuilt Maven repository shards."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import html
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
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable

PRIMARY_SUFFIXES = (".pom", ".jar", ".aar", ".war", ".module")
CHECKSUMS = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}  # nosec: Central requires MD5/SHA1 metadata
MAX_BUNDLE_BYTES = 1_000_000_000
MAVEN_METADATA_NAMESPACE = "http://maven.apache.org/METADATA/1.1.0"
MAVEN_METADATA_SCHEMA = "https://maven.apache.org/xsd/repository-metadata-1.1.0.xsd"
XML_SCHEMA_INSTANCE_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"


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
        if relative.name == "index.html":
            continue
        if relative.name in {"shard-manifest.json", "release-build-manifest.json"}:
            continue
        is_eclipse_dl4j = relative.parts[:3] == ("org", "eclipse", "deeplearning4j")
        is_legacy_nd4j = relative.parts[:2] == ("org", "nd4j")
        if not is_eclipse_dl4j and not is_legacy_nd4j:
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
        expected = {
            item["path"]: (item["sha256"], item["size"])
            for item in manifest.get("files", [])
        }
        actual = {
            path.relative_to(repository).as_posix(): (
                digest(path),
                path.stat().st_size,
            )
            for path in repository_files(repository)
        }
        if expected != actual:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            changed = sorted(
                path
                for path in expected.keys() & actual.keys()
                if expected[path] != actual[path]
            )
            raise ValueError(
                f"manifest mismatch; missing={missing}, extra={extra}, changed={changed}"
            )
    version_dirs: dict[Path, list[Path]] = {}
    for path in repository_files(repository):
        if path.name.endswith((".asc", ".md5", ".sha1", ".sha256", ".sha512")):
            continue
        if path.name == "maven-metadata.xml":
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
    return [
        path
        for path in repository_files(repository)
        if path.suffix in PRIMARY_SUFFIXES and not path.name.endswith(".asc")
    ]


def write_checksums(paths: Iterable[Path]) -> None:
    for path in paths:
        for algorithm in CHECKSUMS:
            Path(str(path) + f".{algorithm}").write_text(
                digest(path, algorithm) + "\n", encoding="ascii"
            )


def render_browse_index(directory: str, children: dict[str, bool]) -> str:
    """Render a small static directory index for object-store browsing."""
    title = "DL4J Maven Repository"
    if directory:
        title += f" / {directory}"
    rows = []
    if directory:
        rows.append('<li><a href="../">../</a></li>')
    for name, is_directory in sorted(children.items(), key=lambda item: (not item[1], item[0].lower())):
        label = name + ("/" if is_directory else "")
        href = urllib.parse.quote(label, safe="/-_.~")
        rows.append(
            f'<li><a href="{html.escape(href, quote=True)}">'
            f"{html.escape(label)}</a></li>"
        )
    listing = "\n".join(rows) or "<li><em>(empty)</em></li>"
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title></head><body>\n"
        f"<h1>{html.escape(title)}</h1>\n<ul>\n{listing}\n</ul>\n"
        "</body></html>\n"
    )


def write_browse_indexes(repository: Path) -> list[Path]:
    """Write index.html in the root and every directory of a Maven tree."""
    files = [
        path
        for path in repository.rglob("*")
        if path.is_file() and path.name != "index.html"
    ]
    directories: set[Path] = {Path(".")}
    children: dict[Path, dict[str, bool]] = {}
    for path in files:
        relative = path.relative_to(repository)
        parent = relative.parent
        directories.add(parent)
        for ancestor in parent.parents:
            directories.add(ancestor)
        children.setdefault(parent, {})[relative.name] = False
        current = parent
        while current != Path("."):
            child = current.name
            parent_of_current = current.parent
            children.setdefault(parent_of_current, {})[child] = True
            current = parent_of_current

    written: list[Path] = []
    for directory in sorted(directories, key=lambda value: value.as_posix()):
        target_directory = repository / directory
        target_directory.mkdir(parents=True, exist_ok=True)
        target = target_directory / "index.html"
        target.write_text(
            render_browse_index(
                "" if directory == Path(".") else directory.as_posix(),
                children.get(directory, {}),
            ),
            encoding="utf-8",
        )
        written.append(target)
    return written


def metadata_element() -> ET.Element:
    ET.register_namespace("", MAVEN_METADATA_NAMESPACE)
    ET.register_namespace("xsi", XML_SCHEMA_INSTANCE_NAMESPACE)
    return ET.Element(
        f"{{{MAVEN_METADATA_NAMESPACE}}}metadata",
        {
            "modelVersion": "1.1.0",
            f"{{{XML_SCHEMA_INSTANCE_NAMESPACE}}}schemaLocation": (
                f"{MAVEN_METADATA_NAMESPACE} {MAVEN_METADATA_SCHEMA}"
            ),
        },
    )


def metadata_child(parent: ET.Element, name: str, value: str | None = None) -> ET.Element:
    child = ET.SubElement(parent, f"{{{MAVEN_METADATA_NAMESPACE}}}{name}")
    child.text = value
    return child


def write_metadata_xml(path: Path, root: ET.Element) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        path, encoding="utf-8", xml_declaration=True, short_empty_elements=True
    )


def snapshot_file_identity(
    path: Path, artifact_id: str, release_version: str
) -> tuple[str, str | None]:
    extension = path.suffix.removeprefix(".")
    stem = path.name[: -(len(extension) + 1)]
    base_name = f"{artifact_id}-{release_version}"
    if stem == base_name:
        return extension, None
    classifier_prefix = base_name + "-"
    if not stem.startswith(classifier_prefix):
        raise ValueError(
            f"Maven artifact name does not match its coordinates: {path.name}"
        )
    classifier = stem[len(classifier_prefix) :]
    if not classifier:
        raise ValueError(f"Maven artifact has an empty classifier: {path.name}")
    return extension, classifier


def write_maven_metadata(
    repository: Path, release_version: str, updated: str | None = None
) -> list[Path]:
    """Write Maven A-level and snapshot V-level metadata with stable filenames."""
    updated = updated or time.strftime("%Y%m%d%H%M%S", time.gmtime())
    try:
        if len(updated) != 14 or not updated.isdigit():
            raise ValueError
        datetime.strptime(updated, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(
            "Maven metadata timestamp must use UTC yyyyMMddHHmmss"
        ) from exc

    components: dict[tuple[str, str, Path], list[Path]] = {}
    for path in primary_files(repository):
        version_dir = path.parent
        if version_dir.name != release_version:
            raise ValueError(
                f"unexpected Maven version directory: {version_dir.relative_to(repository)}"
            )
        artifact_dir = version_dir.parent
        relative_artifact = artifact_dir.relative_to(repository)
        if len(relative_artifact.parts) < 2:
            raise ValueError(
                f"invalid Maven coordinate path: {relative_artifact.as_posix()}"
            )
        group_id = ".".join(relative_artifact.parts[:-1])
        artifact_id = relative_artifact.parts[-1]
        components.setdefault((group_id, artifact_id, version_dir), []).append(path)
    if not components:
        raise ValueError("repository contains no components for Maven metadata")

    metadata_paths: list[Path] = []
    for (group_id, artifact_id, version_dir), artifacts in sorted(
        components.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        artifact_metadata = metadata_element()
        metadata_child(artifact_metadata, "groupId", group_id)
        metadata_child(artifact_metadata, "artifactId", artifact_id)
        artifact_versioning = metadata_child(artifact_metadata, "versioning")
        metadata_child(artifact_versioning, "latest", release_version)
        if not release_version.endswith("-SNAPSHOT"):
            metadata_child(artifact_versioning, "release", release_version)
        versions = metadata_child(artifact_versioning, "versions")
        metadata_child(versions, "version", release_version)
        metadata_child(artifact_versioning, "lastUpdated", updated)
        artifact_metadata_path = version_dir.parent / "maven-metadata.xml"
        write_metadata_xml(artifact_metadata_path, artifact_metadata)
        metadata_paths.append(artifact_metadata_path)

        if release_version.endswith("-SNAPSHOT"):
            version_metadata = metadata_element()
            metadata_child(version_metadata, "groupId", group_id)
            metadata_child(version_metadata, "artifactId", artifact_id)
            metadata_child(version_metadata, "version", release_version)
            versioning = metadata_child(version_metadata, "versioning")
            snapshot = metadata_child(versioning, "snapshot")
            metadata_child(snapshot, "localCopy", "true")
            metadata_child(versioning, "lastUpdated", updated)
            snapshot_versions = metadata_child(versioning, "snapshotVersions")
            identities: set[tuple[str, str | None]] = set()
            for path in sorted(artifacts):
                extension, classifier = snapshot_file_identity(
                    path, artifact_id, release_version
                )
                identity = (extension, classifier)
                if identity in identities:
                    raise ValueError(
                        "duplicate Maven snapshot extension/classifier for "
                        f"{group_id}:{artifact_id}:{release_version}: {identity}"
                    )
                identities.add(identity)
                snapshot_version = metadata_child(
                    snapshot_versions, "snapshotVersion"
                )
                if classifier is not None:
                    metadata_child(snapshot_version, "classifier", classifier)
                metadata_child(snapshot_version, "extension", extension)
                metadata_child(snapshot_version, "value", release_version)
                metadata_child(snapshot_version, "updated", updated)
            version_metadata_path = version_dir / "maven-metadata.xml"
            write_metadata_xml(version_metadata_path, version_metadata)
            metadata_paths.append(version_metadata_path)
    return metadata_paths


def materialize_test_repository(
    inputs: list[Path],
    output: Path,
    manifest_path: Path,
    release_version: str,
    commit: str,
    *,
    metadata_updated: str | None = None,
) -> dict:
    """Merge shards into a checksum-complete, remote-consumable Maven 2 layout."""
    scratch_manifest = manifest_path.with_name(manifest_path.name + ".merge")
    merge(inputs, output, scratch_manifest, release_version, commit)
    verify(output, scratch_manifest, release_version, commit)
    metadata_paths = write_maven_metadata(
        output, release_version, updated=metadata_updated
    )
    write_checksums([*primary_files(output), *metadata_paths])
    files = [
        {
            "path": path.relative_to(output).as_posix(),
            "sha256": digest(path),
            "size": path.stat().st_size,
        }
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(str(manifest_path) + ".sha256").write_text(
        f"{digest(manifest_path)}  {manifest_path.name}\n", encoding="ascii"
    )
    verify(output, manifest_path, release_version, commit)
    # The Maven tree is also served from object storage, where directory
    # listings are not generated automatically.  Keep a browsable index at
    # the root and at every coordinate directory alongside the normal Maven
    # metadata/checksum files.
    write_browse_indexes(output)
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
