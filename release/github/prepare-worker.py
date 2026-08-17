#!/usr/bin/env python3
"""Create GitHub Actions matrices and configs for the shared release worker."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "release/aws/release-plan.json"
DEFAULT_MATRIX = ROOT / "release/github/workflow-matrix.json"
AZURE_CACHE_PREFIX = "deeplearning4j/releases/compiler-cache/v1"
PUBLIC_DEPENDENCY_CACHE_MANIFEST_URL = (
    "https://dl4jrel26302370c1eeb25.blob.core.windows.net/releases/"
    "deeplearning4j/releases/dependency-cache/v2/public-manifest.json"
)
PUBLIC_DEPENDENCY_CACHE = {
    "publicBaseUrl": (
        "https://dl4jrel26302370c1eeb25.blob.core.windows.net/releases"
    ),
    "host": {
        "identity": "abe03741e11341b4eb4587fc2be01c91c4aaf39f6378691da79fc43b6bffd59e",
        "indexObject": (
            "deeplearning4j/releases/dependency-cache/v2/host-index/"
            "abe03741e11341b4eb4587fc2be01c91c4aaf39f6378691da79fc43b6bffd59e.json"
        ),
        "archiveObject": (
            "deeplearning4j/releases/dependency-cache/v2/objects/"
            "47b6220b71103a0f63de444e305de9d961ae54ba0dac1fa4360534830b2c8992.tar.gz"
        ),
    },
    "targets": [
        {
            "identity": "28f4929d991a53575e9ed3f6f1ce07abbd2c874479312e3f0181ed5bf1d0f50c",
            "indexObject": (
                "deeplearning4j/releases/dependency-cache/v2/index/"
                "28f4929d991a53575e9ed3f6f1ce07abbd2c874479312e3f0181ed5bf1d0f50c.json"
            ),
            "archiveObject": (
                "deeplearning4j/releases/dependency-cache/v2/objects/"
                "0ee34de538482ae624929dce0be4f5aa5b913e5a8eb2a39237dbb813d10622dd.tar.gz"
            ),
            "compatibility": {
                "javacppPlatform": "android-arm64",
                "nativeBackend": "cpu",
            },
        },
        {
            "identity": "b57a0f6d3f7a5b9f9ece17052ac39f90419c757d13769db20d6c5aff1d295405",
            "indexObject": (
                "deeplearning4j/releases/dependency-cache/v2/index/"
                "b57a0f6d3f7a5b9f9ece17052ac39f90419c757d13769db20d6c5aff1d295405.json"
            ),
            "archiveObject": (
                "deeplearning4j/releases/dependency-cache/v2/objects/"
                "a7b9f4f2ebd969813f1b21e3867796ef12bdb86db463c13917d0dfd5e3495505.tar.gz"
            ),
            "compatibility": {
                "javacppPlatform": "android-x86_64",
                "nativeBackend": "cpu",
            },
        },
    ],
}


def load_public_dependency_cache() -> dict:
    """Load the controller-maintained cache index, with a safe static fallback.

    GitHub workers must not enumerate the release container or receive storage
    credentials.  Azure publishes this small manifest after each controller
    refresh; the immutable archives it references are still verified by the
    build driver.  A stale-but-known-good index keeps older branches usable when
    the manifest is temporarily unavailable.
    """
    manifest_url = os.environ.get(
        "DL4J_DEPENDENCY_CACHE_MANIFEST_URL", PUBLIC_DEPENDENCY_CACHE_MANIFEST_URL
    )
    connection = os.environ.get("DL4J_AZURE_CONNECTION_STRING") or os.environ.get(
        "SCCACHE_AZURE_CONNECTION_STRING", ""
    )
    sas = next(
        (
            part.split("=", 1)[1].lstrip("?")
            for part in connection.split(";")
            if part.startswith("SharedAccessSignature=")
        ),
        "",
    )
    if sas and "?" not in manifest_url:
        manifest_url = f"{manifest_url}?{sas}"
    try:
        request = urllib.request.Request(
            manifest_url, headers={"User-Agent": "dl4j-github-release-worker"}
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            candidate = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return copy.deepcopy(PUBLIC_DEPENDENCY_CACHE)

    if not isinstance(candidate, dict) or candidate.get("schemaVersion") != 1:
        return copy.deepcopy(PUBLIC_DEPENDENCY_CACHE)
    if not isinstance(candidate.get("publicBaseUrl"), str):
        return copy.deepcopy(PUBLIC_DEPENDENCY_CACHE)
    host = candidate.get("host")
    targets = candidate.get("targets")
    if not isinstance(host, dict) or not isinstance(targets, list):
        return copy.deepcopy(PUBLIC_DEPENDENCY_CACHE)
    if not all(
        isinstance(item, dict)
        and all(isinstance(item.get(key), str) and item[key] for key in ("identity", "indexObject", "archiveObject"))
        for item in [host, *targets]
    ):
        return copy.deepcopy(PUBLIC_DEPENDENCY_CACHE)
    return candidate


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def plan_shards(plan: dict) -> dict[str, dict]:
    shards = {str(shard["id"]): shard for shard in plan.get("shards", [])}
    if len(shards) != len(plan.get("shards", [])):
        raise ValueError("release plan contains duplicate shard ids")
    return shards


def public_artifact_id(
    shard_id: str, variant_name: str, variant: dict | None = None
) -> str:
    """Return the public row ID for one planned shard/variant.

    Classifier IDs always use single hyphens. ZLUDA rows additionally carry
    the complete ROCm-qualified native classifier suffix so distinct ROCm
    runtimes cannot share a staging ID. The ``-zluda`` shard marker and any
    ROCm version already present in the shard ID are removed before appending
    that suffix. This keeps versioned shards from emitting the ROCm qualifier
    twice.
    """
    shard_id = str(shard_id).strip("-")
    variant_name = str(variant_name).strip("-")
    if not shard_id or not variant_name:
        raise ValueError("release artifact ID cannot be empty")

    classifier_suffix = ""
    if variant:
        candidate = str(variant.get("classifierSuffix", "")).strip("-")
        if "-zluda-rocm-" in candidate:
            classifier_suffix = candidate

    if classifier_suffix:
        # Versioned ZLUDA shard IDs carry their ROCm version in the
        # shard key, while the variant suffix carries the complete published
        # classifier. Strip that shard marker before appending the suffix.
        base_shard = re.sub(r"-zluda(?:-rocm-[0-9]+(?:\.[0-9]+)*)?$", "", shard_id)
        value = f"{base_shard}-{classifier_suffix}"
    elif shard_id == variant_name or shard_id.endswith(f"-{variant_name}"):
        value = shard_id
    else:
        value = f"{shard_id}-{variant_name}"

    artifact_id = re.sub(r"-+", "-", value).strip("-")
    if not artifact_id:
        raise ValueError("release artifact ID cannot be empty")
    return artifact_id


def canonical_variant_name(shard: dict, variant_name: str) -> str:
    """Resolve legacy ZLUDA selectors to the CUDA-versioned variant name.

    The release plans use ``cuda-12.9`` as the public variant label because
    ZLUDA is a CUDA 12.9 ABI variant. Older dispatches used ``zluda`` as the
    label; accepting that alias keeps queued/manual workflow invocations
    compatible without emitting duplicate ``zluda-zluda`` artifact IDs.
    """
    if variant_name != "zluda":
        return variant_name
    build = shard.get("build", {})
    if not build.get("zludaVersion"):
        return variant_name
    variants = build.get("variants", [])
    canonical_names = [
        str(variant.get("name"))
        for variant in variants
        if str(variant.get("name")) != "zluda"
    ]
    return canonical_names[0] if len(canonical_names) == 1 else variant_name


def canonical_selector(shards: dict[str, dict], selector: str) -> str:
    """Return the canonical published classifier ID for an exact classifier."""
    selector = selector.strip()
    matches: list[str] = []
    for candidate_shard_id, shard in shards.items():
        variants = shard.get("build", {}).get("variants", [])
        for variant in variants:
            candidate_variant_name = canonical_variant_name(
                shard, str(variant["name"])
            )
            candidate_id = public_artifact_id(
                candidate_shard_id, candidate_variant_name, variant
            )
            if candidate_id == selector:
                matches.append(candidate_id)

    if len(matches) > 1:
        raise ValueError(
            f"published classifier {selector!r} is ambiguous: "
            + ", ".join(sorted(matches))
        )
    return matches[0] if matches else selector


def dependency_cache_key(shard_id: str, variant: dict) -> str:
    """Group variants only when their managed dependency contract is identical."""
    ignored = {"name", "suffix", "classifierSuffix"}
    contract = {}
    for key, value in variant.items():
        if key in ignored:
            continue
        if key == "extension" and value in {"avx2", "avx512"}:
            continue
        if value is None or value == "" or value == [] or value == {}:
            continue
        contract[key] = value
    if not contract:
        return public_artifact_id(shard_id, "default")

    labels: list[str] = []
    helper = str(contract.get("helper", "")).strip()
    if helper:
        labels.append(helper)
    if contract.get("mlir"):
        labels.append("mlir")
    if "triton" in contract:
        labels.append("triton" if contract["triton"] else "no-triton")
    if contract.get("windowsNativeCompile"):
        labels.append("windows-native-compile")
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return public_artifact_id(shard_id, "-".join(labels or ["custom"]) + f"-{digest}")


def workflow_rows(
    plan: dict,
    matrix: dict,
    workflow: str,
    group: str,
    runner_override: str = "",
    classifiers: str = "",
    selection_mode: str = "complete",
) -> list[dict]:
    if matrix.get("schemaVersion") != 1:
        raise ValueError("workflow matrix schemaVersion must be 1")
    workflows = matrix.get("workflows", {})
    if workflow not in workflows:
        raise ValueError(f"unknown release workflow {workflow!r}")
    shards = plan_shards(plan)
    runtimes = matrix.get("shards", {})
    requested = {
        canonical_selector(shards, classifier.strip())
        for classifier in classifiers.split(",")
        if classifier.strip()
    }
    if selection_mode not in {"complete", "targeted"}:
        raise ValueError(f"unknown matrix selection mode {selection_mode!r}")
    if requested and selection_mode != "targeted":
        raise ValueError(
            "classifier filters require selection_mode='targeted'; complete runs "
            "must execute the canonical workflow matrix"
        )
    if selection_mode == "targeted" and not requested:
        raise ValueError("targeted matrix selection requires at least one classifier")

    def available_selectors(workflow_name: str) -> set[str]:
        selectors: set[str] = set()
        for selection in workflows[workflow_name]:
            shard_id = str(selection["shard"])
            if shard_id not in shards:
                continue
            variants = shards[shard_id]["build"].get("variants", [])
            variant_by_name = {
                canonical_variant_name(shards[shard_id], str(variant["name"])): variant
                for variant in variants
            }
            selected_names = [
                canonical_variant_name(shards[shard_id], str(name))
                for name in (selection.get("variants") or list(variant_by_name))
            ]
            selectors.update(
                public_artifact_id(shard_id, name, variant_by_name[name])
                for name in selected_names
            )
        return selectors

    if requested and not requested.issubset(available_selectors(workflow)):
        owners = [
            workflow_name
            for workflow_name in workflows
            if requested.issubset(available_selectors(workflow_name))
        ]
        if len(owners) == 1:
            workflow = owners[0]

    selections: list[tuple[str, dict, dict[str, dict], list[str]]] = []
    available: set[str] = set()
    for selection in workflows[workflow]:
        shard_id = str(selection["shard"])
        if shard_id not in shards:
            raise ValueError(f"workflow {workflow!r} references unknown shard {shard_id!r}")
        if shard_id not in runtimes:
            raise ValueError(f"workflow matrix has no runtime for shard {shard_id!r}")
        runtime = runtimes[shard_id]
        variants = shards[shard_id]["build"].get("variants", [])
        by_name = {
            canonical_variant_name(shards[shard_id], str(variant["name"])): variant
            for variant in variants
        }
        selected_names = [
            canonical_variant_name(shards[shard_id], str(name))
            for name in (selection.get("variants") or list(by_name))
        ]
        for variant_name in selected_names:
            if variant_name not in by_name:
                raise ValueError(
                    f"workflow {workflow!r} references unknown variant "
                    f"{shard_id}-{variant_name}"
                )
            available.add(
                public_artifact_id(shard_id, variant_name, by_name[variant_name])
            )
        selections.append((shard_id, runtime, by_name, selected_names))

    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"workflow {workflow!r} does not contain requested classifiers: "
            + ", ".join(unknown)
        )

    rows: list[dict] = []
    public_ids: dict[str, tuple[str, str]] = {}
    for shard_id, runtime, by_name, selected_names in selections:
        if runtime.get("group") != group:
            continue
        for variant_name in selected_names:
            variant = by_name[variant_name]
            artifact_id = public_artifact_id(shard_id, variant_name, variant)
            if requested and artifact_id not in requested:
                continue
            previous_selector = public_ids.get(artifact_id)
            identity = (shard_id, variant_name)
            if previous_selector and previous_selector != identity:
                previous_id = f"{previous_selector[0]}-{previous_selector[1]}"
                raise ValueError(
                    "workflow matrix maps multiple selectors to public artifact ID "
                    f"{artifact_id!r}: {previous_id}, {shard_id}-{variant_name}"
                )
            public_ids[artifact_id] = identity
            row = {
                "name": artifact_id,
                "artifactId": artifact_id,
                "selector": artifact_id,
                "shard": shard_id,
                "variant": variant_name,
                "runner": runner_override or str(runtime["runner"]),
                "os": str(shards[shard_id]["os"]),
                # Variants may share native objects but require different
                # downloaded toolchains (for example base vs MLIR/Triton).
                # Use a dependency-contract scope so parallel variants cannot
                # race to save incompatible contents under one cache key.
                "dependencyCacheKey": dependency_cache_key(shard_id, variant),
            }
            if group == "linux":
                container = str(runtime.get("container", "")).strip()
                if not container:
                    raise ValueError(f"Linux shard {shard_id!r} has no container image")
                row["container"] = container
            rows.append(row)
    return rows


def infer_release_version(source: Path) -> str:
    root = ET.parse(source / "pom.xml").getroot()
    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
    version = root.findtext("m:version", namespaces=namespace)
    if not version:
        version = root.findtext("m:parent/m:version", namespaces=namespace)
    if not version:
        raise ValueError("could not infer the Maven release version from pom.xml")
    return version.strip()


def worker_config(args: argparse.Namespace) -> dict:
    plan = load_json(args.plan)
    shards = plan_shards(plan)
    if args.shard not in shards:
        raise ValueError(f"unknown shard {args.shard!r}")
    shard = copy.deepcopy(shards[args.shard])
    variants = shard["build"].get("variants", [])
    variant_name = canonical_variant_name(shard, args.variant)
    selected = [variant for variant in variants if variant["name"] == variant_name]
    if len(selected) != 1:
        raise ValueError(f"unknown variant {args.shard}--{args.variant}")
    shard["build"]["variants"] = selected

    if args.build_threads:
        threads = int(args.build_threads)
        if threads < 1:
            raise ValueError("build threads must be positive")
    else:
        planned_threads = int(shard["build"].get("buildThreads", 1))
        threads = min(planned_threads, max(1, os.cpu_count() or 1))
    shard["build"]["buildThreads"] = threads
    shard["build"]["workflowMvnFlags"] = args.maven_flags
    shard["build"]["buildAot"] = bool(
        args.build_aot and (args.aot_all_spins or args.variant == "base")
    )
    if args.libnd4j_url:
        shard["build"]["libnd4jUrl"] = args.libnd4j_url

    current_version = infer_release_version(args.source)
    release_version = args.release_version or current_version
    config = {
        "schemaVersion": 1,
        "provider": "github-actions",
        "runId": args.run_id,
        "commit": args.commit,
        "releaseVersion": release_version,
        "snapshotVersion": args.snapshot_version or current_version,
        "shard": shard,
        "selectedMachine": {
            "name": os.environ.get("RUNNER_NAME"),
            "architecture": os.environ.get("RUNNER_ARCH"),
            "os": os.environ.get("RUNNER_OS"),
        },
        # Published dependency snapshots are immutable and anonymously readable.
        # Keep them separate from the authenticated compiler-object cache so a
        # GitHub worker can restore LLVM/MLIR without receiving storage keys.
        "dependencyCache": load_public_dependency_cache(),
    }
    if args.azure_cache:
        config["compilerCache"] = {
            "backend": "azure",
            "account": "dl4jrel26302370c1eeb25",
            "connectionStringEnv": "SCCACHE_AZURE_CONNECTION_STRING",
            "container": "releases",
            "keyPrefix": AZURE_CACHE_PREFIX,
            "toolchainCache": {
                "schemaVersion": 1,
                "keyPrefix": "deeplearning4j/releases/toolchain-cache/v1",
            },
        }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--workflow", required=True)
    matrix_parser.add_argument("--group", choices=("linux", "host"), required=True)
    matrix_parser.add_argument("--runner-override", default="")
    matrix_parser.add_argument(
        "--classifiers",
        default="",
        help="Comma-separated published classifier IDs to include",
    )
    matrix_parser.add_argument(
        "--selection-mode",
        choices=("complete", "targeted"),
        default="complete",
        help="Complete runs reject filters; targeted runs require explicit classifiers",
    )

    config_parser = subparsers.add_parser("config")
    config_parser.add_argument("--source", type=Path, default=ROOT)
    config_parser.add_argument("--output", type=Path, required=True)
    config_parser.add_argument("--shard", required=True)
    config_parser.add_argument("--variant", required=True)
    config_parser.add_argument("--build-threads", default="")
    config_parser.add_argument("--maven-flags", default="")
    config_parser.add_argument("--libnd4j-url", default="")
    config_parser.add_argument("--build-aot", action="store_true")
    config_parser.add_argument("--aot-all-spins", action="store_true")
    config_parser.add_argument("--azure-cache", action="store_true")
    config_parser.add_argument("--release-version", default="")
    config_parser.add_argument("--snapshot-version", default="")
    config_parser.add_argument("--run-id", required=True)
    config_parser.add_argument("--commit", required=True)

    version_parser = subparsers.add_parser("release-version")
    version_parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "release-version":
        config = load_json(args.config)
        release_version = str(config.get("releaseVersion", "")).strip()
        if not release_version:
            raise ValueError(f"worker config {args.config} does not define releaseVersion")
        print(release_version)
        return

    plan = load_json(args.plan)
    if args.command == "matrix":
        matrix = load_json(args.matrix)
        print(json.dumps(workflow_rows(
            plan,
            matrix,
            args.workflow,
            args.group,
            args.runner_override.strip(),
            args.classifiers.strip(),
            args.selection_mode,
        ), separators=(",", ":")))
        return

    config = worker_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
