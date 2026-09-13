"""Select shared artifacts using the existing full-release ownership contract.

Never select a non-owner just because it arrived first. Platform classifiers
remain byte-checked by the ordinary strict merge. Input images are not mutated.
"""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET


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
        if config.get("commit") != commit or config.get("releaseVersion") != version:
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
        owner = owners.get(artifact)
        if owner is None:
            return True
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
