#!/usr/bin/env python3
"""Scoped, prebuilt tokenizer snapshot publication (no reactor/native builds).

collect writes a Maven tree and tokenizer-receipt.json. merge consumes four such
host directories (GitHub artifact extraction layout), never arbitrary ZIP shards.
The workflow must retain the merged tree before calling deploy, serialize deploys,
and call verify --remote afterwards. Receipts attest workflow-supplied provenance;
they do not independently prove that an installed artifact came from that commit.
No JAR is repacked. linux-x86_64 explicitly owns all main JARs; libtokenizers may
vary only in its own native subtree, whose headers/API must remain identical.
Credentials belong in Maven settings, never command arguments or receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

GROUP = "org.eclipse.deeplearning4j"
GROUP_PATH = Path("org/eclipse/deeplearning4j")
ARTIFACTS = ("libtokenizers", "tokenizers-native-preset", "tokenizers-native")
PARENTS = ("deeplearning4j", "nd4j", "nd4j-tokenizers")  # deployment order
HOSTS = ("linux-x86_64", "linux-arm64", "windows-x86_64", "macosx-arm64")
CANONICAL = "linux-x86_64"
VERSION = "1.0.0-SNAPSHOT"
RECEIPT = "tokenizer-receipt.json"
URL = "https://central.sonatype.com/repository/maven-snapshots/"
REPOSITORY_ID = "central-portal-snapshots"
# Maven coordinates changed independently of Java/resource packages. Do not rewrite either.
RESOURCE_ROOTS = ("org/eclipse/deeplearning4j/tokenizers/", "org/nd4j/tokenizers/")
MAX_XML = 8 * 1024 * 1024
MAX_JAR = 512 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(commit, run_id, run_attempt, version):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit), "commit must be a full SHA")
    require(re.fullmatch(r"[1-9][0-9]*", str(run_id)), "invalid run ID")
    require(re.fullmatch(r"[1-9][0-9]*", str(run_attempt)), "invalid run attempt")
    require(version == VERSION, "only 1.0.0-SNAPSHOT is in scope")
    return {"commit": commit.lower(), "runId": str(run_id),
            "runAttempt": int(run_attempt), "version": version}


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "unsafe member/path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in name.split("/")),
            "unsafe member/path")
    return Path(*path.parts)


def regular(root, name):
    path = root / relative(name)
    require(not root.is_symlink(), "symlink repository")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink payload")
    require(path.is_file(), "missing payload: " + name)
    return path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path, root):
    return {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
            "sha256": sha(path)}


def xml(data):
    require(len(data) <= MAX_XML and b"\0" not in data and b"<!DOCTYPE" not in data.upper()
            and b"<!ENTITY" not in data.upper(), "unsafe XML")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError("invalid XML") from exc
    namespace = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    expected_namespace = {"project": "{http://maven.apache.org/POM/4.0.0}",
                          "metadata": "{http://maven.apache.org/METADATA/1.1.0}"}.get(root.tag.removeprefix(namespace))
    require(namespace == "" or namespace == expected_namespace, "unknown XML namespace")
    for node in root.iter():
        require(node.tag.startswith(namespace) if namespace else not node.tag.startswith("{"),
                "mixed XML namespaces")
        node.tag = node.tag.removeprefix(namespace)
    return root


def field(node, name, optional=False):
    fields = node.findall(name)
    if optional and not fields:
        return ""
    require(len(fields) == 1 and not len(fields[0]), "invalid XML field: " + name)
    result = (fields[0].text or "").strip()
    require(bool(result), "empty XML field: " + name)
    return result


def pom_info(data):
    root = xml(data)
    require(root.tag == "project", "not a POM")
    parents = root.findall("parent")
    require(len(parents) <= 1, "duplicate POM parent")
    parent = None
    if parents:
        parent = tuple(field(parents[0], n) for n in ("groupId", "artifactId", "version"))
    group = field(root, "groupId", True) or (parent[0] if parent else "")
    version = field(root, "version", True) or (parent[2] if parent else "")
    artifact = field(root, "artifactId")
    packaging = field(root, "packaging", True) or "jar"
    return (group, artifact, version), parent, packaging


def xml_signature(data):
    """Compare POM models without ZIP or XML formatting/timestamp noise."""
    def signature(node):
        return (node.tag, tuple(sorted(node.attrib.items())), (node.text or "").strip(),
                tuple(signature(child) for child in node))
    return signature(xml(data))


def pom_path(artifact, version=VERSION):
    return GROUP_PATH / artifact / version / f"{artifact}-{version}.pom"


def artifact_path(artifact, classifier="", version=VERSION):
    name = f"{artifact}-{version}" + ("-" + classifier if classifier else "") + ".jar"
    return GROUP_PATH / artifact / version / name


def closure(root, version):
    """Installed parent POMs only, restricted to the actual reactor ancestry."""
    needed = set(ARTIFACTS)
    visited = set()
    while needed - visited:
        artifact = sorted(needed - visited)[0]
        data = regular(root, pom_path(artifact, version).as_posix()).read_bytes()
        coords, parent, packaging = pom_info(data)
        require(coords == (GROUP, artifact, version), "wrong POM coordinates: " + artifact)
        require(packaging == ("jar" if artifact in ARTIFACTS else "pom"), "wrong packaging")
        expected_parent = "nd4j-tokenizers" if artifact in ARTIFACTS else {
            "nd4j-tokenizers": "nd4j", "nd4j": "deeplearning4j", "deeplearning4j": None,
        }[artifact]
        require(parent == ((GROUP, expected_parent, version) if expected_parent else None),
                "unexpected reactor parent: " + artifact)
        if parent:
            needed.add(parent[1])
        visited.add(artifact)
    return visited


def native_header(data, host):
    """Check object family and architecture, not just a misleading filename."""
    if host.startswith("linux-"):
        machine = 62 if host == "linux-x86_64" else 183
        return (len(data) >= 20 and data[:6] == b"\x7fELF\x02\x01"
                and int.from_bytes(data[18:20], "little") == machine)
    if host == "windows-x86_64":
        if len(data) < 64 or data[:2] != b"MZ":
            return False
        offset = int.from_bytes(data[60:64], "little")
        return (offset + 6 <= len(data) and data[offset:offset+4] == b"PE\0\0"
                and data[offset+4:offset+6] == b"\x64\x86")
    return (len(data) >= 8 and data[:4] == b"\xcf\xfa\xed\xfe"
            and int.from_bytes(data[4:8], "little") == 0x0100000c)


def native_name(name):
    return bool(re.search(r"\.(so(?:\.[0-9.]+)?|dll|dylib|jnilib|vso)$", name))


def jar_entries(path, artifact, host=None, *, main=False):
    """Validate ZIP members/CRC and native host payloads; return normalized content hashes."""
    entries = {}
    native = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            require(sum(i.file_size for i in infos) <= MAX_JAR, "oversized JAR")
            names = set()
            for item in infos:
                name = item.filename.rstrip("/") if item.is_dir() else item.filename
                relative(name)
                require(name.casefold() not in names, "duplicate ZIP member")
                names.add(name.casefold())
                mode = item.external_attr >> 16
                require(not stat.S_ISLNK(mode) and not item.flag_bits & 1, "unsafe ZIP member")
                if item.is_dir():
                    continue
                data = archive.read(item)  # also checks CRC
                entries[name] = hashlib.sha256(data).hexdigest()
                if host:
                    # Reject other host/accelerator subtrees, including native-image metadata.
                    for root in RESOURCE_ROOTS:
                        if name.startswith(root):
                            tail = name[len(root):].removeprefix("bindings/")
                            segment = tail.split("/")[0]
                            if segment.startswith(("linux-", "windows-", "macosx-", "android-", "ios-")):
                                require(segment == host, "foreign native platform in JAR")
                    if name.startswith("META-INF/native-image/"):
                        require(name.split("/")[2] == host, "foreign native-image platform")
                if native_name(name):
                    require(host is not None, "native in platform-independent JAR")
                    require(native_header(data, host), "wrong native architecture: " + name)
                    native.append(name)
                if name.startswith("META-INF/maven/") and name.endswith("/pom.xml"):
                    coords, _, _ = pom_info(data)
                    require(coords == (GROUP, artifact, VERSION), "wrong embedded POM coordinates")
                    external = path.parent / f"{artifact}-{VERSION}.pom"
                    require(external.is_file() and xml_signature(data) == xml_signature(external.read_bytes()),
                            "embedded/external POM content conflict")
            require(entries, "empty JAR")
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("invalid JAR") from exc
    entries.pop("META-INF/MANIFEST.MF", None)
    if host and artifact == "tokenizers-native-preset":
        # Presets supply JavaCPP Java metadata, not the wrapper or JNI objects.
        # Maven still attaches a classifier JAR containing its own POM metadata.
        require(not native, "native payload in metadata-only preset classifier")
        require(f"META-INF/maven/{GROUP}/{artifact}/pom.xml" in entries,
                "missing preset classifier POM metadata")
    elif host:
        prefix = tuple(root + ("" if artifact == "libtokenizers" else "bindings/")
                       + host + "/" for root in RESOURCE_ROOTS)
        extension = ".so" if host.startswith("linux-") else (
            ".dll" if host.startswith("windows-") else ".dylib")
        wrapper = "libtokenizers_wrapper" + extension
        jni = ("" if host.startswith("windows-") else "lib") + "jnitokenizers" + extension
        required = (wrapper,) if artifact == "libtokenizers" else (wrapper, jni)
        for library in required:
            require(any(n.startswith(prefix) and PurePosixPath(n).name == library for n in native),
                    "missing native payload: " + artifact + ":" + host + ":" + library)
        if artifact == "libtokenizers":
            require(any(n.startswith(tuple(r + "include/" for r in RESOURCE_ROOTS))
                        and n.endswith((".h", ".hpp")) for n in entries), "missing public headers")
        if main and artifact == "libtokenizers":
            # Explicitly ignore only native objects and the producer's per-host build manifest.
            # Classes/POM/API/header/unknown entries are NEVER exempted from comparison.
            for name in list(entries):
                if name.startswith(prefix) and (native_name(name) or name.endswith("/manifest.properties")):
                    del entries[name]
    return entries


def write_receipt(root, provenance, kind, files, **extra):
    receipt = {"schemaVersion": 1, "kind": kind, **provenance,
               "files": [file_record(root / p, root) for p in sorted(files)], **extra}
    (root / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def empty_output(output):
    require(not output.exists(), "output must not already exist")
    require(not any(p.is_symlink() for p in (output, *output.parents)), "symlink output")


def collect(repository, output, classifier, commit, run_id, run_attempt, version=VERSION):
    provenance = identity(commit, run_id, run_attempt, version)
    require(classifier in HOSTS, "unsupported host")
    empty_output(output)
    parents = closure(repository, version)
    files = {pom_path(a, version).as_posix() for a in parents}
    for artifact in ARTIFACTS:
        files.update(artifact_path(artifact, c, version).as_posix() for c in ("", classifier))
        # Retain installed own-coordinate documentation; never pull stale remote binaries.
        for c in ("sources", "javadoc"):
            p = artifact_path(artifact, c, version)
            if (repository / p).exists():
                files.add(p.as_posix())
    validate_files(repository, files, (classifier,), version)
    output.mkdir(parents=True)
    for name in sorted(files):
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(regular(repository, name), dest)
    return write_receipt(output, provenance, "host", files, classifier=classifier)


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_receipt(root, provenance=None, kind=None):
    receipt = json.loads(regular(root, RECEIPT).read_text(encoding="utf-8"),
                         object_pairs_hook=no_duplicate_keys)
    require(isinstance(receipt, dict) and receipt.get("schemaVersion") == 1, "invalid receipt schema")
    actual = identity(receipt.get("commit"), receipt.get("runId"), receipt.get("runAttempt"),
                      receipt.get("version"))
    require(provenance is None or provenance == actual, "receipt source/run/attempt/version mismatch")
    require(receipt.get("kind") in ("host", "merged") and (kind is None or receipt["kind"] == kind),
            "wrong receipt kind")
    files = receipt.get("files")
    require(isinstance(files, list) and files, "empty receipt")
    seen = set()
    for record in files:
        require(isinstance(record, dict) and set(record) == {"path", "size", "sha256"}, "invalid file record")
        name = record["path"]
        relative(name)
        require(name not in seen, "duplicate receipt path")
        seen.add(name)
        require(type(record["size"]) is int and record["size"] > 0
                and isinstance(record["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "invalid digest/size")
        require(file_record(regular(root, name), root) == record, "checksum/size mismatch: " + name)
    actual_files = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "symlink repository member")
        if path.is_file() and path != root / RECEIPT:
            actual_files.add(path.relative_to(root).as_posix())
    require(actual_files == seen, "unreceipted/missing files")
    require(closure(root, actual["version"]) == set((*ARTIFACTS, *PARENTS)), "invalid closure")
    return receipt, seen


def validate_files(root, files, hosts, version):
    allowed = {pom_path(a, version).as_posix() for a in (*ARTIFACTS, *PARENTS)}
    required = set(allowed)
    for artifact in ARTIFACTS:
        allowed.update(artifact_path(artifact, c, version).as_posix()
                       for c in ("", *hosts, "sources", "javadoc"))
        required.update(artifact_path(artifact, c, version).as_posix() for c in ("", *hosts))
    require(required <= files <= allowed, "unexpected/missing scoped artifacts")
    for name in files:
        path = root / name
        if path.suffix != ".jar":
            continue
        artifact = path.parent.parent.name
        is_main = name == artifact_path(artifact, "", version).as_posix()
        host = next((h for h in hosts if name == artifact_path(artifact, h, version).as_posix()), None)
        if is_main and artifact == "libtokenizers":
            host = CANONICAL if len(hosts) > 1 else hosts[0]
        jar_entries(path, artifact, host, main=is_main)
    # libtokenizers' main currently embeds this host's native payload and public headers.
    # Match its classifier to that exact installed producer before canonical main selection.
    main_host = CANONICAL if len(hosts) > 1 else hosts[0]
    lib_main = jar_entries(root / artifact_path("libtokenizers", "", version), "libtokenizers", main_host)
    lib_classifier = jar_entries(root / artifact_path("libtokenizers", main_host, version), "libtokenizers", main_host)
    headers = lambda entries: {n: h for n, h in entries.items()
                               if n.startswith(tuple(r + "include/" for r in RESOURCE_ROOTS))}
    require(headers(lib_main) == headers(lib_classifier), "main/classifier public header conflict")
    for name, digest in lib_classifier.items():
        if native_name(name) or name.endswith(".class"):
            require(lib_main.get(name) == digest, "main/classifier native/API content conflict")
    # Every classifier's optional JVM API entries must match the corresponding shared main.
    for artifact in ARTIFACTS:
        main_entries = jar_entries(root / artifact_path(artifact, "", version), artifact,
                                  main_host if artifact == "libtokenizers" else None)
        for host in hosts:
            classified = jar_entries(root / artifact_path(artifact, host, version), artifact, host)
            if artifact == "libtokenizers":
                require(headers(classified) == headers(main_entries), "classifier public header conflict")
            for name, digest in classified.items():
                if name.endswith(".class"):
                    require(main_entries.get(name) == digest, "classifier JVM API content conflict")


def merge(inputs, output, commit, run_id, run_attempt, version=VERSION):
    provenance = identity(commit, run_id, run_attempt, version)
    empty_output(output)
    roots = {}
    for path in sorted(inputs.rglob(RECEIPT)):
        receipt, files = load_receipt(path.parent, provenance, "host")
        host = receipt.get("classifier")
        require(host in HOSTS and host not in roots, "duplicate/unknown host receipt")
        validate_files(path.parent, files, (host,), version)
        roots[host] = (path.parent, files)
    require(set(roots) == set(HOSTS), "exactly four host receipts required")
    canonical, files = roots[CANONICAL]
    selected = {name: canonical / name for name in files}
    for host in HOSTS:
        root, host_files = roots[host]
        # All shared files, including documentation and POMs, must be present on all hosts.
        shared = {n for n in host_files if n != artifact_path(Path(n).parent.parent.name, host, version).as_posix()}
        canonical_shared = {n for n in files if n != artifact_path(Path(n).parent.parent.name, CANONICAL, version).as_posix()}
        require(shared == canonical_shared, "shared artifact set conflict")
        for name in shared:
            if name.endswith(".jar"):
                artifact = Path(name).parent.parent.name
                is_main = name == artifact_path(artifact, "", version).as_posix()
                incoming = jar_entries(root / name, artifact, host if artifact == "libtokenizers" and is_main else None,
                                       main=is_main)
                expected = jar_entries(canonical / name, artifact, CANONICAL if artifact == "libtokenizers" and is_main else None,
                                       main=is_main)
                require(incoming == expected, "shared JAR content conflict: " + name)
            else:
                require((root / name).read_bytes() == (canonical / name).read_bytes(), "POM content conflict: " + name)
        for artifact in ARTIFACTS:
            name = artifact_path(artifact, host, version).as_posix()
            selected[name] = root / name
    # No output is created until every receipt, host payload and shared artifact passes.
    output.mkdir(parents=True)
    for name, source in sorted(selected.items()):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return write_receipt(output, provenance, "merged", selected,
                         classifiers=list(HOSTS), canonicalMainOwner=CANONICAL,
                         hostReceipts={h: sha(roots[h][0] / RECEIPT) for h in HOSTS})


def verify(repository, commit, run_id, run_attempt, version=VERSION, *, remote=False):
    provenance = identity(commit, run_id, run_attempt, version)
    receipt, files = load_receipt(repository, provenance, "merged")
    require(receipt.get("classifiers") == list(HOSTS) and receipt.get("canonicalMainOwner") == CANONICAL,
            "invalid merged ownership")
    hosts = receipt.get("hostReceipts", {})
    require(isinstance(hosts, dict) and set(hosts) == set(HOSTS)
            and all(isinstance(d, str) and re.fullmatch(r"[0-9a-f]{64}", d) for d in hosts.values()),
            "invalid host receipt attestations")
    validate_files(repository, files, HOSTS, version)
    if remote:
        metadata = preflight(repository, version)
        for record in receipt["files"]:
            path = Path(record["path"])
            artifact = path.parent.parent.name
            classifier, extension = local_identity(path, artifact, version)
            value = metadata[artifact].get((classifier, extension))
            require(value is not None, "remote snapshot artifact missing")
            filename = f"{artifact}-{value}" + ("-" + classifier if classifier else "") + "." + extension
            data = fetch(URL + (path.parent / filename).as_posix())
            require(len(data) == record["size"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                    "remote artifact hash/size mismatch: " + artifact + ":" + classifier)
    return receipt


def local_identity(path, artifact, version):
    base = f"{artifact}-{version}"
    stem = path.name[:-len(path.suffix)]
    require(stem == base or stem.startswith(base + "-"), "wrong artifact filename")
    return ("" if stem == base else stem[len(base)+1:], path.suffix[1:])


def snapshot_entries(data, artifact, version):
    """Fail closed: invalid, duplicate, ambiguous or mismatched XML is never a 404."""
    root = xml(data)
    require(root.tag == "metadata" and field(root, "groupId") == GROUP
            and field(root, "artifactId") == artifact and field(root, "version") == version,
            "snapshot metadata coordinate mismatch")
    versioning = root.findall("versioning")
    require(len(versioning) == 1, "invalid snapshot versioning")
    containers = versioning[0].findall("snapshotVersions")
    require(len(containers) == 1 and len(containers[0]) > 0, "missing snapshotVersions")
    result = {}
    for entry in containers[0]:
        require(entry.tag == "snapshotVersion"
                and all(e.tag in ("classifier", "extension", "value", "updated") for e in entry),
                "invalid snapshotVersion fields")
        classifier = field(entry, "classifier", True)
        extension = field(entry, "extension")
        value = field(entry, "value")
        updated = field(entry, "updated")
        require(not classifier or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", classifier), "invalid classifier")
        require(re.fullmatch(r"[a-z][a-z0-9]*", extension), "invalid extension")
        require(re.fullmatch(re.escape(version[:-9]) + r"-[0-9]{8}\.[0-9]{6}-[1-9][0-9]*", value),
                "invalid snapshot value")
        require(re.fullmatch(r"[0-9]{14}", updated), "invalid snapshot timestamp")
        try:
            datetime.strptime(updated, "%Y%m%d%H%M%S")
            datetime.strptime(value[len(version[:-9])+1:].rsplit("-", 1)[0], "%Y%m%d.%H%M%S")
        except ValueError as exc:
            raise ValueError("invalid snapshot calendar timestamp") from exc
        key = (classifier, extension)
        require(key not in result, "duplicate snapshot identity")
        result[key] = value
    require(("", "pom") in result, "metadata missing main POM")
    return result


def fetch(url):
    # Do not echo network exception strings, redirect destinations or credential material.
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(MAX_JAR + 1)
            require(len(data) <= MAX_JAR, "oversized remote snapshot payload")
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError("snapshot metadata/artifact not found") from None
        raise RuntimeError("snapshot request failed (HTTP " + str(exc.code) + ")") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("snapshot request failed") from None


def preflight(repository, version):
    """Check required AND all remote-advertised identities for every touched GAV first."""
    result = {}
    for artifact in (*PARENTS, *ARTIFACTS):
        directory = repository / pom_path(artifact, version).parent
        local = {local_identity(p, artifact, version) for p in directory.iterdir() if p.is_file()}
        required = {(h, "jar") for h in HOSTS} if artifact in ARTIFACTS else set()
        required.add(("", "pom"))
        if artifact in ARTIFACTS:
            required.add(("", "jar"))
        try:
            data = fetch(URL + pom_path(artifact, version).parent.as_posix() + "/maven-metadata.xml")
        except FileNotFoundError:
            advertised = {}
        else:
            advertised = snapshot_entries(data, artifact, version)
        require(required | set(advertised) <= local,
                "incomplete snapshot classifiers for " + artifact + "; retain every advertised attachment")
        result[artifact] = advertised
    return result


def deploy_commands(repository, version, maven="mvn"):
    """One deploy-file per GAV, parents first; attach the complete classifier set."""
    commands = []
    for artifact in (*PARENTS, *ARTIFACTS):
        pom = repository / pom_path(artifact, version)
        main = repository / artifact_path(artifact, "", version) if artifact in ARTIFACTS else pom
        attachments = []
        for path in sorted(pom.parent.glob("*.jar")):
            if path == main:
                continue
            classifier, extension = local_identity(path, artifact, version)
            require(classifier and "," not in str(path), "invalid deployment attachment")
            attachments.append((path, classifier, extension))
        command = [maven, "--batch-mode", "org.apache.maven.plugins:maven-deploy-plugin:3.1.4:deploy-file",
                   f"-DrepositoryId={REPOSITORY_ID}", f"-Durl={URL}", f"-Dfile={main}",
                   f"-DpomFile={pom}", "-Dpackaging=" + ("jar" if artifact in ARTIFACTS else "pom"),
                   "-DgeneratePom=false", "-DretryFailedDeploymentCount=3"]
        if attachments:
            command += ["-Dfiles=" + ",".join(str(a[0]) for a in attachments),
                        "-Dclassifiers=" + ",".join(a[1] for a in attachments),
                        "-Dtypes=" + ",".join(a[2] for a in attachments)]
        commands.append(command)
    return commands


def deploy(repository, version, *, maven="mvn"):
    receipt, _ = load_receipt(repository, kind="merged")
    verify(repository, receipt["commit"], receipt["runId"], receipt["runAttempt"], version)
    # Freeze the verified payload before network checks/upload; keep Maven metadata writes
    # away from the retained tree. All GAVs must pass before the first subprocess starts.
    with tempfile.TemporaryDirectory(prefix="tokenizer-deploy-") as temporary:
        frozen = Path(temporary) / "repository"
        shutil.copytree(repository, frozen)
        verify(frozen, receipt["commit"], receipt["runId"], receipt["runAttempt"], version)
        preflight(frozen, version)
        for command in deploy_commands(frozen, version, maven):
            # Use external Maven settings. No credentials or full command are logged here.
            subprocess.run(command, check=True, cwd=temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("collect", "merge", "verify", "deploy"):
        p = sub.add_parser(action)
        p.add_argument("--version", required=True, choices=(VERSION,))
        if action == "merge":
            p.add_argument("--inputs", type=Path, required=True)
        else:
            p.add_argument("--repository", type=Path, required=True)
        if action in ("collect", "merge"):
            p.add_argument("--output", type=Path, required=True)
        if action == "collect":
            p.add_argument("--classifier", required=True, choices=HOSTS)
        if action != "deploy":
            p.add_argument("--commit", required=True)
            p.add_argument("--run-id", required=True)
            p.add_argument("--run-attempt", required=True, type=int)
        if action == "verify":
            p.add_argument("--remote", action="store_true", help="also compare published snapshot bytes")
        if action == "deploy":
            p.add_argument("--maven", default="mvn", help="Maven executable; settings supply authentication")
    args = vars(parser.parse_args(argv))
    action = args.pop("action")
    try:
        globals()[action](**args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        # These errors contain only our validation diagnostics; fetch sanitizes network errors.
        parser.exit(1, "tokenizer publication failed: " + str(exc) + "\n")
    except subprocess.CalledProcessError:
        # Never emit a provider exception's command, environment or captured output.
        parser.exit(1, "tokenizer Maven deployment failed\n")
    print("tokenizer " + action + " completed")


if __name__ == "__main__":
    main()
