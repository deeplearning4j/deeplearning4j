#!/usr/bin/env python3
"""Assemble the canonical native matrix and Java reactor, without publishing.

Shared native coordinates have declared owners, not a first-writer-wins merge.
Java and parent POMs are installed once from the same immutable source checkout.
The isolated Maven repository is never restored from a dependency cache.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from release.central import repository as central


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_module("release_preparer", ROOT / "release/github/prepare-worker.py")
driver = load_module("release_driver", ROOT / "release/aws/build-platform.py")
NS = {"m": "http://maven.apache.org/POM/4.0.0"}
GROUP = "org.eclipse.deeplearning4j"
GROUP_PATH = Path(*GROUP.split("."))
JAVA_OWNER = "java-reactor"
CPU_OWNER = ("linux-x86_64-cpu", "base")
BACKENDS = Path("nd4j/nd4j-backends/nd4j-backend-impls")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_digest(shard: dict) -> str:
    return hashlib.sha256(json.dumps(shard, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selected_contract(shard: dict, variant: dict) -> tuple[dict, dict]:
    build, rules = copy.deepcopy(shard["build"]), copy.deepcopy(shard["artifactRules"])
    build["variants"] = [copy.deepcopy(variant)]
    if build["backend"] == "cpu":
        build["modules"].append(":nd4j-cpu-backend-common")
        for key in ("artifactIds", "unclassifiedArtifactIds"):
            rules.setdefault(key, []).append("nd4j-cpu-backend-common")
    driver.enable_sdx_release_component(build, rules)
    return build, rules


def shared_owners(plan: dict) -> dict[str, tuple[str, str]]:
    owners = {artifact: CPU_OWNER for artifact in (
        "nd4j-native", "nd4j-native-preset", "nd4j-cpu-backend-common",
        *driver.TOKENIZER_ARTIFACT_IDS, *driver.SDX_ARTIFACT_IDS,
    )}
    for shard in worker.plan_shards(plan).values():
        build = shard["build"]
        if build["backend"] == "cuda" and not build.get("zludaVersion"):
            cuda = build["cudaVersion"]
            owner = (f"linux-x86_64-cuda-{cuda.replace('.', '-')}", "base")
            for artifact in (f"nd4j-cuda-{cuda}", f"nd4j-cuda-{cuda}-preset",
                             f"nd4j-cuda-{cuda}-backend-common"):
                owners[artifact] = owner
    for backend in ("vulkan", "hexagon", "tpu"):
        for artifact in (f"nd4j-{backend}", f"nd4j-{backend}-preset"):
            owners[artifact] = (f"linux-x86_64-{backend}", "base")
    owners["nd4j-zluda-12.9"] = ("linux-x86_64-zluda", "cuda-12.9")
    expected = {(s["id"], v["name"]) for s in worker.plan_shards(plan).values()
                for v in s["build"]["variants"]}
    if set(owners.values()) - expected:
        raise ValueError("shared component owner is absent from the canonical plan")
    return owners


def classifier_components(build: dict, rules: dict) -> tuple[str, ...]:
    required = list(driver.required_classifier_artifact_ids(build, rules))
    if build["backend"] == "tpu":
        required.append("nd4j-tpu")
    # Hexagon currently embeds its runtime in the unclassified backend JAR.
    if build.get("buildCrossPlatform") and driver.has_base_platform_variant(build):
        required.extend(driver.TOKENIZER_ARTIFACT_IDS)
    return tuple(dict.fromkeys(required))


def same_workflow_run(actual: str, expected: str) -> bool:
    # Failed-job reruns reuse successful artifacts from earlier attempts, but
    # artifacts from another workflow run must never join the repository.
    current = re.fullmatch(r"github-([0-9]+)-([1-9][0-9]*)", expected)
    previous = re.fullmatch(r"github-([0-9]+)-([1-9][0-9]*)", actual)
    return bool(current and previous and current[1] == previous[1]
                and int(previous[2]) <= int(current[2]))


def inspect_workers(root: Path, plan: dict, version: str, commit: str,
                    run_id: str) -> dict[tuple[str, str], Path]:
    shards = worker.plan_shards(plan)
    expected = {(s["id"], v["name"]) for s in shards.values() for v in s["build"]["variants"]}
    found = {}
    for path in sorted(root.rglob("worker-config.json")):
        config = load_json(path)
        shard = config["shard"]
        variants = shard["build"]["variants"]
        if len(variants) != 1:
            raise ValueError(f"expected one worker variant: {path}")
        key = (shard["id"], variants[0]["name"])
        if key not in expected or key in found:
            raise ValueError(f"unexpected or duplicate worker: {key}")
        if ((config.get("commit"), config.get("releaseVersion")) != (commit, version)
                or not same_workflow_run(config.get("runId", ""), run_id)):
            raise ValueError(f"worker source/version/run mismatch: {key}")
        if shard.get("contractDigest") != contract_digest(shards[key[0]]):
            raise ValueError(f"worker plan digest mismatch: {key}")
        variant = next(v for v in shards[key[0]]["build"]["variants"] if v["name"] == key[1])
        if variants[0] != variant:
            raise ValueError(f"worker variant contract mismatch: {key}")
        if not version.endswith("-SNAPSHOT") and not shard["build"].get("releaseMetadata"):
            raise ValueError(f"worker did not enable release metadata: {key}")
        worker_root = path.parent
        if not (worker_root / "worker-success").is_file():
            raise ValueError(f"worker has no success receipt: {key}")
        if load_json(worker_root / "build-result.json").get("completedVariants") != [key[1]]:
            raise ValueError(f"worker did not attest completion: {key}")
        if not (worker_root / "maven-repository").is_dir():
            raise ValueError(f"worker has no Maven repository: {key}")
        found[key] = worker_root / "maven-repository"
    if set(found) != expected:
        raise ValueError(f"missing canonical workers: {sorted(expected - set(found))}")
    return found


def component_path(artifact: str, version: str) -> Path:
    return GROUP_PATH / artifact / version


def put(source: Path, local: Path, relative: Path, owner: str, ownership: dict,
        *, replace_parent: bool = False, consume: bool = False) -> None:
    destination = local / relative
    previous = ownership.get(relative.as_posix())
    if previous and previous != owner and not replace_parent:
        raise ValueError(f"multiple owners for {relative}: {previous}, {owner}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        if consume:
            shutil.move(str(source), destination)
        else:
            shutil.copy2(source, destination)
    ownership[relative.as_posix()] = owner


def seed_native(found: dict, plan: dict, local: Path, version: str,
                supplements: dict | None = None) -> dict[str, str]:
    ownership = {}
    supplements = dict(supplements or {})
    owners = shared_owners(plan)
    shards = worker.plan_shards(plan)
    for key, repository in sorted(found.items()):
        shard = shards[key[0]]
        variant = next(v for v in shard["build"]["variants"] if v["name"] == key[1])
        build, rules = selected_contract(shard, variant)
        label = "/".join(key)
        driver.attest_variant_classifier_artifacts(repository, build, rules, variant, version, "full-repository")
        for artifact in classifier_components(build, rules):
            classifier = driver.variant_artifact_classifier_for(build, variant, artifact)
            relative = component_path(artifact, version) / f"{artifact}-{version}-{classifier}.jar"
            source = repository / relative
            if not source.is_file():
                raise ValueError(f"missing canonical classifier: {label}: {relative}")
            driver.attest_classifier_archive_contract(source, rules, artifact, classifier, "full-repository")
            put(source, local, relative, label, ownership, consume=True)
        for artifact, owner in owners.items():
            if key != owner:
                continue
            directory = component_path(artifact, version)
            # Sources/javadoc are component metadata, not native variant outputs.
            suffixes = (".pom", ".jar")
            if not version.endswith("-SNAPSHOT"):
                suffixes += ("-sources.jar", "-javadoc.jar")
            for suffix in suffixes:
                relative = directory / f"{artifact}-{version}{suffix}"
                source = repository / relative
                metadata_source = supplements.pop(relative.as_posix(), None)
                if metadata_source is not None:
                    if artifact != "libtokenizers" or suffix not in ("-sources.jar", "-javadoc.jar"):
                        raise ValueError(f"recovery cannot replace native artifacts: {relative}")
                    if source.exists():
                        raise ValueError(f"recovery cannot replace existing worker metadata: {relative}")
                    put(metadata_source, local, relative, "metadata-recovery", ownership)
                else:
                    if not source.is_file():
                        raise ValueError(f"missing canonical shared artifact: {label}: {relative}")
                    put(source, local, relative, label, ownership, consume=True)
    if supplements:
        raise ValueError(f"unused metadata recovery artifacts: {sorted(supplements)}")
    # Bootstrap only the actual ancestor chain, not every build-only POM (for
    # example libnd4j) installed by --also-make. Java must replace every ancestor.
    pending = [local / path for path in ownership if path.endswith(".pom")]
    visited = set()
    while pending:
        pom = pending.pop()
        if pom in visited:
            continue
        visited.add(pom)
        parent = ET.parse(pom).getroot().find("m:parent", NS)
        if parent is None:
            continue
        group = parent.findtext("m:groupId", namespaces=NS)
        if group != GROUP:
            raise ValueError(f"native component has a parent outside the release namespace: {pom}")
        artifact = parent.findtext("m:artifactId", namespaces=NS)
        if parent.findtext("m:version", namespaces=NS) != version:
            raise ValueError(f"native component parent version mismatch: {pom}")
        relative = component_path(artifact, version) / f"{artifact}-{version}.pom"
        if relative.as_posix() not in ownership:
            source = found[CPU_OWNER] / relative
            if not source.is_file():
                raise ValueError(f"missing canonical parent POM: {relative}")
            put(source, local, relative, "parent-bootstrap", ownership)
        pending.append(local / relative)
    return ownership


def run(command: list[str], source: Path, env: dict) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=source, env=env, check=True)


def effective_components(path: Path, version: str) -> list[tuple[Path, str]]:
    root = ET.parse(path).getroot()
    projects = [root] if root.tag.endswith("}project") else root.findall("m:project", NS)
    if not projects:
        raise ValueError(f"no reactor models in {path}")
    result = []
    for project in projects:
        group = project.findtext("m:groupId", namespaces=NS)
        artifact = project.findtext("m:artifactId", namespaces=NS)
        current = project.findtext("m:version", namespaces=NS)
        if not group or not (group == GROUP or group.startswith(GROUP + ".")) or current != version:
            raise ValueError(f"reactor coordinate outside release: {group}:{artifact}:{current}")
        packaging = project.findtext("m:packaging", "jar", NS)
        result.append((Path(*group.split(".")) / artifact / version, packaging))
    return result


def install_java(command: list[str], source: Path, env: dict, model: Path,
                 local: Path, version: str, ownership: dict) -> None:
    # Obtain Maven's actual activated reactor, not a filesystem scan that could
    # silently count inactive modules or old artifacts as freshly built Java.
    model_command = [arg for arg in command if arg != "install"]
    run([*model_command, "org.apache.maven.plugins:maven-help-plugin:3.5.1:effective-pom",
         f"-Doutput={model}"], source, env)
    components = effective_components(model, version)
    for directory, _ in components:
        if any(Path(path).parent == directory and owner not in ("parent-bootstrap", JAVA_OWNER)
               for path, owner in ownership.items()):
            raise ValueError(f"Java reactor includes native-owned component {directory}")
    run(command, source, env)
    for directory, packaging in components:
        artifact = directory.parent.name
        required = [f"{artifact}-{version}.pom"]
        if packaging != "pom":
            extension = "jar" if packaging in {"maven-plugin", "bundle"} else packaging
            required.append(f"{artifact}-{version}.{extension}")
            if not version.endswith("-SNAPSHOT"):
                required.extend(f"{artifact}-{version}-{kind}.jar" for kind in ("sources", "javadoc"))
        for name in required:
            if not (local / directory / name).is_file():
                raise ValueError(f"Java install missing required output: {directory / name}")
        for path in (local / directory).iterdir():
            if path.suffix not in central.PRIMARY_SUFFIXES:
                continue
            relative = path.relative_to(local)
            previous = ownership.get(relative.as_posix())
            if previous not in (None, "parent-bootstrap", JAVA_OWNER):
                raise ValueError(f"Java reactor includes native-owned component {relative}")
            put(path, local, relative, JAVA_OWNER, ownership, replace_parent=packaging == "pom")


def build_java(source: Path, local: Path, output: Path, version: str, snapshot: str,
               ownership: dict, plan: dict) -> None:
    # The CUDA updater also rewrites release-plan.json. Read the version set
    # from the immutable contract, never from that mutated working checkout.
    cuda_versions = sorted({s["build"]["cudaVersion"] for s in worker.plan_shards(plan).values()
                            if s["build"].get("cudaVersion")})
    env = dict(os.environ, DL4J_PLATFORM="linux-x86_64", DL4J_OS="linux",
               DL4J_MAVEN_GOAL="install", DL4J_MAVEN_REPOSITORY=str(local),
               DL4J_BUILD_SDX="0", DL4J_RELEASE_METADATA="0" if version.endswith("-SNAPSHOT") else "1")
    # Version tooling must use the same isolated repository too.
    env["MAVEN_OPTS"] = env.get("MAVEN_OPTS", "") + f" -Dmaven.repo.local={local}"
    run(["bash", "./update-versions.sh", snapshot, version], source, env)
    # 12.9 is the canonical parent model; version-specific CUDA components own
    # their own dependency overrides and distinct common-module coordinates.
    run(["bash", "./change-cuda-versions.sh", "12.9"], source, env)
    script = "build-scripts/release/cross-platform.sh"
    printed = subprocess.check_output(["bash", script, "--print-java"], cwd=source, env=env, text=True)
    command = shlex.split(printed)
    install_java(command, source, env, output / "java-effective-pom.xml", local, version, ownership)
    base = ["mvn", "--batch-mode", "--no-transfer-progress", f"-Dmaven.repo.local={local}",
            "-Dmaven.test.skip=true", "install"]
    if not version.endswith("-SNAPSHOT"):
        base += ["-Pcentral-release", "-Dmaven.javadoc.failOnError=true"]
    # Aggregators are built standalone, without --also-make or native profiles.
    # Do not set javacpp.platform: that would collapse all platform dependencies
    # onto this Linux assembly host through the custom-platform parent profile.
    for module in ("nd4j-native-platform", "nd4j-vulkan-platform", "nd4j-zluda-platform"):
        install_java([*base, "-f", str(BACKENDS / module / "pom.xml")], source, env,
                     output / f"{module}-effective-pom.xml", local, version, ownership)
    for cuda in cuda_versions:
        run(["bash", "./change-cuda-versions.sh", cuda], source, env)
        install_java([*base, "-f", str(BACKENDS / "nd4j-cuda-platform/pom.xml")], source, env,
                     output / f"cuda-{cuda}-effective-pom.xml", local, version, ownership)
    if "parent-bootstrap" in ownership.values():
        missing = [p for p, owner in ownership.items() if owner == "parent-bootstrap"]
        raise ValueError(f"parents not rebuilt by the Java reactor: {missing}")


def verify_consumer(local: Path, output: Path, ownership: dict, source: Path) -> None:
    """Resolve every shipped binary, using installed POMs as a real consumer.

    Any DL4J artifact Maven obtains that was not produced by this run is a
    closure failure, even if a remote repository happened to satisfy it.
    """
    def consumer_project(name: str, packaging: str = "pom"):
        project = ET.Element("project", xmlns=NS["m"])
        for key, value in (("modelVersion", "4.0.0"), ("groupId", "release.verification"),
                           ("artifactId", name), ("version", "1"), ("packaging", packaging)):
            ET.SubElement(project, key).text = value
        return project

    project = consumer_project("repository-consumer")
    modules = ET.SubElement(project, "modules")
    # Independent consumer modules prevent Maven nearest-wins mediation from
    # hiding an unsatisfied dependency behind another binary's dependency graph.
    consumer_root = output / "consumers"
    consumer_root.mkdir()
    count = 0
    for relative in sorted(ownership):
        path = Path(relative)
        if path.suffix not in (".jar", ".aar", ".war") or path.name.endswith(("-sources.jar", "-javadoc.jar", "-tests.jar")):
            continue
        artifact, version = path.parts[-3:-1]
        prefix = f"{artifact}-{version}"
        classifier = path.stem[len(prefix):].removeprefix("-")
        name = f"consumer-{count:04d}"
        count += 1
        module = consumer_project(name)
        dependencies = ET.SubElement(module, "dependencies")
        dep = ET.SubElement(dependencies, "dependency")
        for key, value in (("groupId", ".".join(path.parts[:-3])), ("artifactId", artifact),
                           ("version", version), ("type", path.suffix[1:]), ("classifier", classifier)):
            if value:
                ET.SubElement(dep, key).text = value
        directory = consumer_root / name
        directory.mkdir()
        ET.ElementTree(module).write(directory / "pom.xml", encoding="utf-8", xml_declaration=True)
        ET.SubElement(modules, "module").text = name
    if not count:
        raise ValueError("repository contains no consumer binaries")
    consumer = consumer_root / "pom.xml"
    ET.ElementTree(project).write(consumer, encoding="utf-8", xml_declaration=True)
    run(["mvn", "--batch-mode", "--no-transfer-progress", "-f", str(consumer),
         f"-Dmaven.repo.local={local}", "org.apache.maven.plugins:maven-dependency-plugin:3.8.1:resolve",
         "-DoutputFile=dependencies.txt"], source, dict(os.environ))
    unexpected = sorted(p.relative_to(local).as_posix() for p in central.primary_files(local)
                        if p.relative_to(local).as_posix() not in ownership)
    if unexpected:
        raise ValueError(f"release requires DL4J artifacts not built by this run: {unexpected}")


def finalize(local: Path, output: Path, ownership: dict, version: str, commit: str,
             found: dict, recovery: dict | None = None) -> None:
    repository = output / "maven-repository"
    repository.mkdir()
    files = []
    for relative, owner in sorted(ownership.items()):
        path = local / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Finalize only after Maven has stopped writing: moving avoids doubling
        # a multi-platform repository's native payload on the assembly runner.
        shutil.move(str(path), destination)
        files.append({"path": relative, "sha256": central.digest(destination),
                      "size": destination.stat().st_size, "shards": [owner]})
    manifest = {"schemaVersion": 1, "releaseVersion": version, "commit": commit,
                "workloads": ["maven"], "assembly": "canonical-full-repository",
                "workers": ["/".join(k) for k in sorted(found)], "files": files}
    if recovery is not None:
        manifest["metadataRecovery"] = recovery
    path = output / "repository-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(str(path) + ".sha256").write_text(f"{central.digest(path)}  {path.name}\n", encoding="ascii")
    central.verify(repository, path, version, commit)
    if not version.endswith("-SNAPSHOT"):
        central.verify_release_metadata(repository)
    print(f"Verified {len(found)} native workers and {len(files)} repository files; no publication requested.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workers", "source", "local-repository", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("release-version", "snapshot-version", "commit", "run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--metadata-fix-source", type=Path)
    parser.add_argument("--metadata-fix-commit")
    args = parser.parse_args()
    if bool(args.metadata_fix_source) != bool(args.metadata_fix_commit):
        raise ValueError("metadata recovery requires both source checkout and explicit fix commit")
    source, local, output = args.source.resolve(), args.local_repository.resolve(), args.output.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("full repository requires an immutable 40-character source SHA")
    for version in (args.release_version, args.snapshot_version):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
            raise ValueError(f"invalid Maven version: {version}")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != args.commit:
        raise ValueError("assembly source checkout does not match native worker commit")
    if (local.exists() and any(local.iterdir())) or (output.exists() and any(output.iterdir())):
        raise ValueError("assembly output and local Maven repository must be fresh, not cached")
    local.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    plan = load_json(source / "release/aws/release-plan.json")
    found = inspect_workers(args.workers, plan, args.release_version, args.commit, args.run_id)
    supplements, recovery = {}, None
    if args.metadata_fix_commit:
        metadata = load_module("metadata_recovery", ROOT / "release/github/metadata_recovery.py")
        supplements, recovery = metadata.prepare(source, args.metadata_fix_source.resolve(),
            args.metadata_fix_commit, args.commit, args.release_version, output)
    ownership = seed_native(found, plan, local, args.release_version, supplements)
    build_java(source, local, output, args.release_version, args.snapshot_version, ownership, plan)
    verify_consumer(local, output, ownership, source)
    finalize(local, output, ownership, args.release_version, args.commit, found, recovery)


if __name__ == "__main__":
    main()
