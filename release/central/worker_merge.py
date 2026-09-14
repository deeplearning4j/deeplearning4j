"""Select shared artifacts using the existing full-release ownership contract.

Never select a non-owner just because it arrived first. Platform classifiers
remain byte-checked by the ordinary strict merge. Input images are not mutated.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _jar_content_digest(path):
    """Digest of ZIP entry contents, ignoring metadata timestamps."""
    import zipfile
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            digest.update(info.filename.encode())
            digest.update(archive.read(info.filename))
    return digest.hexdigest()


def _jars_match(left, right):
    """Content match for real ZIP archives; strict bytes otherwise."""
    import zipfile
    try:
        return _jar_content_digest(left) == _jar_content_digest(right)
    except zipfile.BadZipFile:
        return left.read_bytes() == right.read_bytes()


def select(inputs, version, commit):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "canonical_release_assembly", root / "release/github/full-repository.py")
    assembly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembly)
    plan = assembly.load_json(root / "release/aws/release-plan.json")
    owners = assembly.shared_owners(plan)
    selected = {}
    by_identity = {}
    for repository in inputs:
        if not repository.is_dir():
            raise ValueError("Canonical worker merge requires extracted worker images")
        worker = repository.parent
        config = json.loads((worker / "worker-config.json").read_text())
        result = json.loads((worker / "build-result.json").read_text())
        if not (worker / "worker-success").is_file():
            raise ValueError(f"Worker did not complete: {worker}")
        try:
            from .source_identity import check
        except ImportError:
            from source_identity import check
        try:
            check(commit, config.get('commit'))
        except ValueError as error:
            raise ValueError(f"Worker source/version mismatch: {worker}") from error
        if config.get("releaseVersion") != version:
            raise ValueError(f"Worker source/version mismatch: {worker}")
        variants = config["shard"]["build"]["variants"]
        if len(variants) != 1 or variants[0]["name"] not in result["completedVariants"]:
            raise ValueError(f"Worker variant receipt mismatch: {worker}")
        identity = (config["shard"]["id"], variants[0]["name"])
        if identity in by_identity:
            raise ValueError(f"Duplicate worker identity: {identity}")
        by_identity[identity] = repository
        selected[repository] = identity
        # Match full-repository assembly: libnd4j is a build aggregator, not
        # a published component. Fail if any consumer actually needs its POM.
        if (repository / 'org/eclipse/deeplearning4j/libnd4j').exists():
            for pom in repository.rglob('*.pom'):
                if pom.parent.parent.name == 'libnd4j':
                    continue
                project = ET.parse(pom).getroot()
                refs = project.findall('{*}parent')
                for model in [project, *project.findall('{*}profiles/{*}profile')]:
                    # This explicitly opted-in source assembly profile downloads
                    # a native ZIP only when rebuilding from source. It is not
                    # activated by a consumer resolving the published JAR.
                    if (model is not project and
                            model.findtext('{*}id') == 'libnd4j-assembly' and
                            model.findtext('{*}activation/{*}property/{*}name') == 'libnd4j-assembly' and
                            model.findtext('{*}activation/{*}activeByDefault', 'false') == 'false'):
                        dependencies = model.findall('{*}dependencies/{*}dependency')
                        if (len(dependencies) == 1 and
                                dependencies[0].findtext('{*}artifactId') == 'libnd4j' and
                                dependencies[0].findtext('{*}type') == 'zip' and
                                dependencies[0].findtext('{*}classifier')):
                            continue
                    refs += model.findall('{*}dependencies/{*}dependency')
                    refs += model.findall('{*}dependencyManagement/{*}dependencies/{*}dependency')
                # Build plugin dependencies order header generation, but are not
                # part of a consumer's dependency graph.
                if any(ref.findtext('{*}artifactId') == 'libnd4j' and
                       ref.findtext('{*}groupId') == 'org.eclipse.deeplearning4j'
                       for ref in refs):
                    raise ValueError(f'Published POM requires build-only libnd4j: {pom}')

    def include(repository, relative):
        artifact, artifact_version = relative.parent.parent.name, relative.parent.name
        if relative.parts[:3] == ('org', 'eclipse', 'deeplearning4j') and artifact == 'libnd4j':
            return False
        # Unowned shared components (for example nd4j-presets-common) are built
        # by several native lanes. Main artifacts are accepted when their ZIP
        # entry contents match (reproducible-build timestamps differ); real
        # content divergence fails. Attachments come from the first shard in
        # canonical order, recorded in the manifest.
        if owners.get(artifact) is None:
            filename = relative.name
            is_main = filename in (f"{artifact}-{artifact_version}.jar",
                                   f"{artifact}-{artifact_version}.pom")
            probe = repository / relative
            for other in inputs:
                candidate = other / relative
                if candidate == probe or not candidate.is_file():
                    continue
                if candidate.read_bytes() == probe.read_bytes():
                    continue
                if (candidate.suffix == '.jar' and probe.suffix == '.jar'
                        and _jars_match(candidate, probe)):
                    continue
                if is_main:
                    raise ValueError(
                        f"Conflicting component main artifact {relative}: "
                        f"{selected[repository]} vs {selected[other]}")
            if is_main:
                return True
            ordered = sorted(inputs, key=lambda item: selected[item], reverse=True)
            producer = next(item for item in ordered if (item / relative).is_file())
            return repository == producer
        owner = owners[artifact]
        shared_names = {f"{artifact}-{artifact_version}{suffix}" for suffix in
                        (".jar", "-sources.jar", "-javadoc.jar")}
        # Also select checksums/signatures with their owning artifact.
        filename = relative.name
        for suffix in (".sha512", ".sha256", ".sha1", ".md5", ".asc"):
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
        if filename not in shared_names:
            return True
        owner_repository = by_identity.get(owner)
        if owner_repository is None or not (owner_repository / relative).is_file():
            raise ValueError(f"Missing canonical owner {owner} for {relative}")
        return selected[repository] == owner
    return include
