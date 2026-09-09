"""Explicit source-preserving recovery of missing libtokenizers metadata only."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

MODULE = Path("nd4j/nd4j-tokenizers/libtokenizers")
NS = {"m": "http://maven.apache.org/POM/4.0.0"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def model(element):
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(model(child) for child in element))


def validate_pom(original, fixed):
    before, after = ET.fromstring(original), ET.fromstring(fixed)
    profiles = after.find("m:profiles", NS)
    additions = [p for p in profiles if p.findtext("m:id", namespaces=NS) == "central-release"]
    if len(additions) != 1:
        raise ValueError("metadata fix must add exactly one central-release profile")
    profiles.remove(additions[0])
    if model(before) != model(after):
        raise ValueError("metadata fix changed more than the central-release packaging profile")


def tree(source):
    rows = subprocess.check_output(["git", "ls-tree", "-r", "HEAD", "--", str(MODULE)],
                                   cwd=source, text=True).splitlines()
    return {row.split("\t", 1)[1]: row.split("\t", 1)[0] for row in rows
            if row.split("\t", 1)[1] != str(MODULE / "pom.xml")}


def prepare(source, fix_source, fix_commit, commit, version, output):
    if not re.fullmatch(r"[0-9a-f]{40}", fix_commit):
        raise ValueError("metadata fix requires an immutable 40-character SHA")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fix_source, text=True).strip()
    if actual != fix_commit or not tree(source) or tree(source) != tree(fix_source):
        raise ValueError("metadata fix checkout differs in native source content or commit")
    pom = source / MODULE / "pom.xml"
    original, fixed = pom.read_bytes(), (fix_source / MODULE / "pom.xml").read_bytes()
    validate_pom(original, fixed)
    target = source / MODULE / "target"
    if target.exists():
        raise ValueError("metadata recovery requires an untouched source checkout")
    # Direct goals do not execute any lifecycle phase, Rust/C++, or JavaCPP.
    # Restore the original POM before the Java reactor runs. Native worker
    # repositories and success receipts are never changed by metadata recovery.
    command = ["mvn", "--batch-mode", "--no-transfer-progress", "-f", str(MODULE / "pom.xml"),
               "-Pcentral-release", "jar:jar@central-native-sources", "jar:jar@central-native-javadoc"]
    try:
        pom.write_bytes(fixed)
        subprocess.run(command, cwd=source, check=True)
        supplements, evidence = {}, []
        for kind in ("sources", "javadoc"):
            archives = list(target.glob(f"libtokenizers-*-{kind}.jar"))
            if len(archives) != 1:
                raise ValueError(f"metadata producer did not emit exactly one {kind} archive")
            entries = {}
            with ZipFile(archives[0]) as archive:
                for name in archive.namelist():
                    if name.endswith("/") or name.startswith("META-INF/"):
                        continue
                    path = Path(name)
                    if path.is_absolute() or ".." in path.parts or any(p in ("target", "build") for p in path.parts):
                        raise ValueError(f"invalid metadata archive path: {name}")
                    if path.suffix in (".jar", ".class", ".so", ".dll", ".dylib", ".a", ".o"):
                        raise ValueError(f"binary in native metadata: {name}")
                    data = archive.read(name)
                    if not data or data != (source / MODULE / path).read_bytes():
                        raise ValueError(f"metadata entry is not the original source: {name}")
                    entries[name] = digest(data)
            required = {"include/tokenizers_c.h", "include/tokenizers_ffi.h"}
            if kind == "sources":
                required |= {"src/tokenizers_c.cpp", "tokenizers-ffi/src/lib.rs", "pom.xml"}
            if not required <= entries.keys():
                raise ValueError(f"missing native metadata entries: {required - entries.keys()}")
            relative = Path("org/eclipse/deeplearning4j/libtokenizers") / version / f"libtokenizers-{version}-{kind}.jar"
            destination = output / "metadata-supplements" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archives[0], destination)
            supplements[relative.as_posix()] = destination
            evidence.append({"path": relative.as_posix(), "sha256": digest(destination.read_bytes()), "entries": entries})
    finally:
        pom.write_bytes(original)
    provenance = {"schemaVersion": 1, "sourceCommit": commit, "packagingFixCommit": fix_commit,
                  "sourcePomSha256": digest(original), "packagingPomSha256": digest(fixed),
                  "nativeSourceTreeSha256": digest(json.dumps(tree(source), sort_keys=True).encode()),
                  "command": command, "files": evidence}
    (output / "metadata-recovery-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return supplements, provenance
