#!/usr/bin/env python3
"""Create GitHub Actions matrices and configs for the shared release worker."""

from __future__ import annotations

import argparse
import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "release/aws/release-plan.json"
DEFAULT_MATRIX = ROOT / "release/github/workflow-matrix.json"
AZURE_CACHE_PREFIX = "deeplearning4j/releases/compiler-cache/v1"


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


def workflow_rows(
    plan: dict,
    matrix: dict,
    workflow: str,
    group: str,
    runner_override: str = "",
) -> list[dict]:
    if matrix.get("schemaVersion") != 1:
        raise ValueError("workflow matrix schemaVersion must be 1")
    workflows = matrix.get("workflows", {})
    if workflow not in workflows:
        raise ValueError(f"unknown release workflow {workflow!r}")
    shards = plan_shards(plan)
    runtimes = matrix.get("shards", {})
    rows: list[dict] = []
    for selection in workflows[workflow]:
        shard_id = str(selection["shard"])
        if shard_id not in shards:
            raise ValueError(f"workflow {workflow!r} references unknown shard {shard_id!r}")
        if shard_id not in runtimes:
            raise ValueError(f"workflow matrix has no runtime for shard {shard_id!r}")
        runtime = runtimes[shard_id]
        if runtime.get("group") != group:
            continue
        variants = shards[shard_id]["build"].get("variants", [])
        by_name = {str(variant["name"]): variant for variant in variants}
        selected_names = selection.get("variants") or list(by_name)
        for variant_name in selected_names:
            if variant_name not in by_name:
                raise ValueError(
                    f"workflow {workflow!r} references unknown variant "
                    f"{shard_id}--{variant_name}"
                )
            row = {
                "name": f"{shard_id}--{variant_name}",
                "shard": shard_id,
                "variant": variant_name,
                "runner": runner_override or str(runtime["runner"]),
                "os": str(shards[shard_id]["os"]),
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
    selected = [variant for variant in variants if variant["name"] == args.variant]
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
    }
    if args.azure_cache:
        config["compilerCache"] = {
            "backend": "azure",
            "connectionStringEnv": "SCCACHE_AZURE_CONNECTION_STRING",
            "container": "releases",
            "keyPrefix": AZURE_CACHE_PREFIX,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_json(args.plan)
    if args.command == "matrix":
        matrix = load_json(args.matrix)
        print(json.dumps(workflow_rows(
            plan,
            matrix,
            args.workflow,
            args.group,
            args.runner_override.strip(),
        ), separators=(",", ":")))
        return

    config = worker_config(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
