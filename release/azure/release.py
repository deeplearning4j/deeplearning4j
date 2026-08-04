#!/usr/bin/env python3
"""Run the portable DL4J release matrix on Microsoft Azure.

The controller packs compatible work onto persistent VM lanes and launches all
selected lanes concurrently. Shards remain isolated artifact units while a lane
reuses its toolchains, dependency repository, and compiler cache. Accelerator
backends are compile-only and therefore use CPU VMs. Azure does not provide the
macOS lane or a Cloud TPU smoke-test equivalent.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import copy
import datetime as dt
import hashlib
import itertools
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import random
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable
import urllib.request
import uuid

DEFAULT_REPOSITORY = "https://github.com/deeplearning4j/deeplearning4j.git"
MANAGED_TAG = "dl4j-release-managed"
RUN_TAG = "dl4j-run"
SHARD_TAG = "dl4j-shard"
CONTROLLER_EPOCH_TAG = "dl4j-controller-epoch"
STORAGE_BLOB_DATA_CONTRIBUTOR = "ba92f5b4-2d11-453d-a403-e96b0029c9fe"
MANAGEMENT_SCOPE = "https://management.azure.com/.default"
AZURE_SUBSCRIPTION_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
AZURE_LOCATION_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62})$")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "release/azure/release-plan.json"
BUILD_DRIVER = ROOT / "release/aws/build-platform.py"
CLOUD_IO = ROOT / "release/azure/cloud-io.py"
LOG_STREAM_CHUNK_BYTES = 8 * 1024 * 1024
LOG_STREAM_CONFLICT_RETRIES = 3
RESOURCE_CLEANUP_ATTEMPTS = 12
RESOURCE_CLEANUP_RETRY_SECONDS = 5
RESOURCE_RECONCILE_TIMEOUT_SECONDS = 300


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Azure release plan root must be an object")
    if value.get("schemaVersion") != 1:
        raise ValueError(f"unsupported Azure release plan schema: {value.get('schemaVersion')}")
    defaults = value.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("Azure release plan requires a defaults object")
    for key in ("x86MachineCandidates", "armMachineCandidates"):
        candidates = defaults.get(key)
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(item, str) or not item for item in candidates)
        ):
            raise ValueError(f"Azure release plan defaults.{key} must be a non-empty string list")
    shards = value.get("shards")
    if not isinstance(shards, list) or not shards or any(
        not isinstance(item, dict) for item in shards
    ):
        raise ValueError("Azure release plan requires a non-empty shard object list")
    ids = [item.get("id") for item in shards]
    if (
        any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("release plan requires unique, non-empty shard ids")
    for shard in shards:
        shard_id = shard["id"]
        if shard.get("os") not in {"linux", "windows"}:
            raise ValueError(f"Azure shard {shard_id} has unsupported OS {shard.get('os')!r}")
        if shard.get("architecture") not in {"x86_64", "arm64"}:
            raise ValueError(f"Azure shard {shard_id} has unsupported architecture")
        expected_worker = "worker.ps1" if shard["os"] == "windows" else "worker.sh"
        if shard.get("worker") != expected_worker:
            raise ValueError(
                f"Azure shard {shard_id} must use {expected_worker} for {shard['os']}"
            )
        expected_class = "arm" if shard["architecture"] == "arm64" else "x86"
        if shard.get("machineClass") != expected_class:
            raise ValueError(
                f"Azure shard {shard_id} must use machineClass {expected_class}"
            )
        candidates = shard.get("machineCandidates")
        if candidates is not None and (
            not isinstance(candidates, list)
            or not candidates
            or any(not isinstance(item, str) or not item for item in candidates)
        ):
            raise ValueError(
                f"Azure shard {shard_id} machineCandidates must be a non-empty string list"
            )
        build = shard.get("build")
        variants = build.get("variants") if isinstance(build, dict) else None
        if (
            not isinstance(variants, list)
            or not variants
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item.get("name")
                for item in variants
            )
        ):
            raise ValueError(f"shard {shard_id} has no valid build variants")
        if shard["os"] == "windows":
            unsupported = [
                variant["name"]
                for variant in variants
                if (
                    variant.get("mlir")
                    or variant.get("triton")
                    or variant["name"] == "compile"
                    or variant["name"].endswith("-compile")
                )
            ]
            if unsupported:
                raise ValueError(
                    f"Windows shard {shard_id} requests managed LLVM/MLIR variants "
                    f"unsupported by MSVC: {', '.join(unsupported)}"
                )
        workloads = shard.get("workloads")
        if (
            not isinstance(workloads, list)
            or not workloads
            or any(not isinstance(item, str) for item in workloads)
            or not set(workloads) <= {"maven", "sdk"}
        ):
            raise ValueError(f"shard {shard_id} has an unknown workload")
        image = shard.get("image")
        if not isinstance(image, dict):
            raise ValueError(f"shard {shard_id} has incomplete Azure image metadata")
        required = {"publisher", "offer", "sku", "version", "architecture"}
        if not required <= set(image) or any(
            not isinstance(image.get(key), str) or not image.get(key) for key in required
        ):
            raise ValueError(f"shard {shard_id} has incomplete Azure image metadata")
        expected_image_arch = "Arm64" if shard["architecture"] == "arm64" else "x64"
        if image["architecture"] != expected_image_arch:
            raise ValueError(
                f"Azure shard {shard_id} image architecture {image['architecture']!r} "
                f"does not match {shard['architecture']!r} (expected {expected_image_arch})"
            )
        lane = shard.get("lane")
        if lane is not None and (
            not isinstance(lane, str)
            or not lane
            or lane != normalize_name(lane, 63)
        ):
            raise ValueError(
                f"shard {shard_id} has invalid Azure lane id {lane!r}"
            )
    return value


def normalize_name(value: str, maximum: int = 63) -> str:
    result = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    if not result or not result[0].isalpha():
        result = "r-" + result
    return result[:maximum].rstrip("-")


def resource_name(prefix: str, run_id: str, shard: str = "", maximum: int = 63) -> str:
    """Return a deterministic Azure-safe name without exposing user-supplied text."""
    if maximum < 10:
        raise ValueError("Azure resource names require room for an opaque suffix")
    source = f"{run_id}/{shard}" if shard else run_id
    digest = hashlib.sha256(source.encode()).hexdigest()
    # Azure rejects reserved words such as WINDOWS even as substrings. Run and
    # lane IDs remain available through tags and manifests instead of names.
    stem = normalize_name(prefix, maximum - 9)
    digest_length = min(16, maximum - len(stem) - 1)
    return f"{stem}-{digest[:digest_length]}"


def resource_group_name(location: str, override: str | None = None) -> str:
    return override or normalize_name(f"dl4j-release-{location}", 90)


def storage_account_name(subscription: str, location: str, override: str | None = None) -> str:
    if override:
        value = override.lower()
        if not re.fullmatch(r"[a-z0-9]{3,24}", value):
            raise ValueError("Azure storage account names must be 3-24 lowercase letters/digits")
        return value
    digest = hashlib.sha1(f"{subscription}/{location}".encode()).hexdigest()[:15]
    return f"dl4jrel{digest}"


def artifact_container_name(plan: dict[str, Any]) -> str:
    return plan.get("artifactContainer", "releases")


def control_container_name(plan: dict[str, Any]) -> str:
    return plan.get("controlContainer", "control")


def compiler_cache_key_prefix(plan: dict[str, Any]) -> str:
    return f"{plan.get('artifactPrefix', 'deeplearning4j/releases').strip('/')}/compiler-cache/v1"


def compiler_cache_metadata(plan: dict[str, Any], account_name: str) -> dict[str, str]:
    return {
        "backend": "azure",
        "account": account_name,
        "container": artifact_container_name(plan),
        "keyPrefix": compiler_cache_key_prefix(plan),
    }


def kill_switch_blob(plan: dict[str, Any]) -> str:
    """Global, forced emergency-stop switch retained for stop-everything."""
    return f"{plan.get('artifactPrefix', 'deeplearning4j/releases').strip('/')}/control/kill-switch.json"


def run_controller_prefix(plan: dict[str, Any]) -> str:
    return f"{plan.get('artifactPrefix', 'deeplearning4j/releases').strip('/')}/control/runs/"


def run_kill_switch_blob(plan: dict[str, Any], run_id: str) -> str:
    return f"{run_controller_prefix(plan)}{run_id}/kill-switch.json"


def controller_lock_blob(plan: dict[str, Any], run_id: str | None = None) -> str:
    # A lease must protect the switch it fences. Run controllers therefore lease
    # their own switch, while administrative operations retain the global switch.
    return run_kill_switch_blob(plan, run_id) if run_id else kill_switch_blob(plan)


def shard_contract_digest(shard: dict[str, Any]) -> str:
    """Identify every plan field that can affect one shard's artifacts."""
    value = copy.deepcopy(shard)
    value.pop("contractDigest", None)
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def with_shard_contract_digest(shard: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(shard)
    value["contractDigest"] = shard_contract_digest(value)
    return value


def shard_status_identity(
    config: dict[str, Any], shard: dict[str, Any]
) -> dict[str, Any]:
    return {
        "controllerEpoch": config["controllerEpoch"],
        "runId": config["runId"],
        "shard": shard["id"],
        "repository": config["repository"],
        "commit": config["commit"],
        "releaseVersion": config["releaseVersion"],
        "snapshotVersion": config["snapshotVersion"],
        "contractDigest": shard["contractDigest"],
        "variants": [item["name"] for item in shard["build"]["variants"]],
    }


def shard_status_matches(
    status: dict[str, Any], expected: dict[str, Any]
) -> bool:
    return all(status.get(key) == value for key, value in expected.items())


def interactive_wizard_enabled(allow_wizard: bool) -> bool:
    ci = any(
        os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no"}
        for name in ("CI", "GITHUB_ACTIONS")
    )
    return bool(allow_wizard and not ci and getattr(sys.stdin, "isatty", lambda: False)())


def prompt_value(label: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        print(f"{label}{suffix}: ", end="", file=sys.stderr, flush=True)
        value = input().strip()
    except EOFError as exc:
        raise SystemExit("Azure configuration wizard lost its interactive input") from exc
    if not value and default is not None:
        value = default
    if not value:
        raise azure_configuration_error(f"no value was entered for {label}")
    return value


def azure_configuration_error(problem: str) -> SystemExit:
    return SystemExit(
        f"Azure release configuration is incomplete: {problem}. Authenticate with "
        "az login or standard AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET "
        "credentials, set AZURE_SUBSCRIPTION_ID and AZURE_LOCATION, then run "
        "python3 release/azure/release.py configure in an interactive terminal."
    )


def azure_modules() -> dict[str, Any]:
    try:
        from azure.core import MatchConditions
        from azure.core.exceptions import HttpResponseError, ResourceExistsError, ResourceNotFoundError
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.authorization import AuthorizationManagementClient
        from azure.mgmt.compute import ComputeManagementClient
        from azure.mgmt.msi import ManagedServiceIdentityClient
        from azure.mgmt.network import NetworkManagementClient
        from azure.mgmt.resource import ResourceManagementClient, SubscriptionClient
        from azure.mgmt.storage import StorageManagementClient
        from azure.storage.blob import (
            BlobLeaseClient,
            BlobSasPermissions,
            BlobServiceClient,
            ContainerSasPermissions,
            ContentSettings,
            generate_blob_sas,
            generate_container_sas,
        )
    except ImportError as exc:
        raise SystemExit(
            "Azure dependencies are missing. Run: "
            "python3 -m pip install -r release/azure/requirements.txt"
        ) from exc
    return {
        "MatchConditions": MatchConditions,
        "HttpResponseError": HttpResponseError,
        "ResourceExistsError": ResourceExistsError,
        "ResourceNotFoundError": ResourceNotFoundError,
        "DefaultAzureCredential": DefaultAzureCredential,
        "AuthorizationManagementClient": AuthorizationManagementClient,
        "ComputeManagementClient": ComputeManagementClient,
        "ManagedServiceIdentityClient": ManagedServiceIdentityClient,
        "NetworkManagementClient": NetworkManagementClient,
        "ResourceManagementClient": ResourceManagementClient,
        "SubscriptionClient": SubscriptionClient,
        "StorageManagementClient": StorageManagementClient,
        "BlobLeaseClient": BlobLeaseClient,
        "BlobSasPermissions": BlobSasPermissions,
        "BlobServiceClient": BlobServiceClient,
        "ContainerSasPermissions": ContainerSasPermissions,
        "ContentSettings": ContentSettings,
        "generate_blob_sas": generate_blob_sas,
        "generate_container_sas": generate_container_sas,
    }


def safe_credential_problem(exc: BaseException) -> str:
    return f"DefaultAzureCredential validation returned {exc.__class__.__name__}"


def load_credential(modules: dict[str, Any]) -> tuple[Any | None, str | None]:
    try:
        credential = modules["DefaultAzureCredential"](
            exclude_interactive_browser_credential=True
        )
        credential.get_token(MANAGEMENT_SCOPE)
        return credential, None
    except Exception as exc:
        return None, safe_credential_problem(exc)


def configure_azure_credentials(modules: dict[str, Any], problem: str) -> Any:
    print(f"Azure credentials are required ({problem}).", file=sys.stderr)
    executable = shutil.which("az")
    if not executable:
        raise azure_configuration_error(
            "Azure CLI is not installed and DefaultAzureCredential could not authenticate"
        )
    answer = prompt_value("Enter az to run az login")
    if answer.lower() != "az":
        raise azure_configuration_error("expected az at the Azure sign-in prompt")
    if subprocess.run([executable, "login"], check=False).returncode != 0:
        raise azure_configuration_error("az login failed")
    credential, retry_problem = load_credential(modules)
    if retry_problem:
        raise azure_configuration_error(f"Azure rejected the new login: {retry_problem}")
    return credential


def configured_subscription(override: str | None = None) -> str | None:
    return override or os.environ.get("AZURE_SUBSCRIPTION_ID")


def valid_subscription(value: str | None) -> bool:
    return bool(value and AZURE_SUBSCRIPTION_PATTERN.fullmatch(value))


def cloud_context(subscription_override: str | None = None, *, allow_wizard: bool = True) -> dict[str, Any]:
    modules = azure_modules()
    credential, problem = load_credential(modules)
    if problem:
        if not interactive_wizard_enabled(allow_wizard):
            raise azure_configuration_error(problem)
        print("Azure release environment wizard", file=sys.stderr)
        credential = configure_azure_credentials(modules, problem)
    subscription = configured_subscription(subscription_override)
    subscription_client = modules["SubscriptionClient"](credential)
    if not valid_subscription(subscription):
        available = [
            str(item.subscription_id)
            for item in subscription_client.subscriptions.list()
            if getattr(item, "state", "Enabled") == "Enabled"
        ]
        if len(available) == 1:
            subscription = available[0]
        elif interactive_wizard_enabled(allow_wizard):
            print("Azure release environment wizard", file=sys.stderr)
            subscription = prompt_value(
                "Azure subscription ID (AZURE_SUBSCRIPTION_ID)",
                default=subscription,
            )
        else:
            raise azure_configuration_error("no valid AZURE_SUBSCRIPTION_ID was resolved")
    if not valid_subscription(subscription):
        raise azure_configuration_error(
            f"{subscription!r} is not a valid AZURE_SUBSCRIPTION_ID"
        )
    return {
        "modules": modules,
        "credential": credential,
        "subscription": subscription,
        "subscriptions": subscription_client,
        "authorization": modules["AuthorizationManagementClient"](credential, subscription),
        "compute": modules["ComputeManagementClient"](credential, subscription),
        "identity": modules["ManagedServiceIdentityClient"](credential, subscription),
        "network": modules["NetworkManagementClient"](credential, subscription),
        "resource": modules["ResourceManagementClient"](credential, subscription),
        "storage": modules["StorageManagementClient"](credential, subscription),
    }


def resolve_location(value: str | None, *, allow_wizard: bool = True) -> str:
    location = (
        value
        or os.environ.get("AZURE_LOCATION")
        or os.environ.get("AZURE_DEFAULTS_LOCATION")
    )
    if not location or not AZURE_LOCATION_PATTERN.fullmatch(location.lower()):
        if not interactive_wizard_enabled(allow_wizard):
            raise azure_configuration_error("no valid AZURE_LOCATION was resolved")
        print("Azure release environment wizard", file=sys.stderr)
        location = prompt_value("Azure location (AZURE_LOCATION)", default=location)
    location = location.lower().replace(" ", "")
    if not AZURE_LOCATION_PATTERN.fullmatch(location):
        raise azure_configuration_error(f"{location!r} is not a valid Azure location")
    return location


def configure_environment(args: argparse.Namespace) -> None:
    context = cloud_context(
        args.subscription, allow_wizard=not getattr(args, "no_wizard", False)
    )
    location = resolve_location(
        args.location, allow_wizard=not getattr(args, "no_wizard", False)
    )
    known = {
        str(item.name).lower()
        for item in context["subscriptions"].subscriptions.list_locations(
            context["subscription"]
        )
    }
    if location not in known:
        raise RuntimeError(f"Azure location {location!r} is not enabled for this subscription")
    print(json.dumps({
        "configured": True,
        "subscription": context["subscription"],
        "location": location,
        "credentialSource": context["credential"].__class__.__name__,
        "nonSecretEnvironmentForFutureCommands": {
            "AZURE_SUBSCRIPTION_ID": context["subscription"],
            "AZURE_LOCATION": location,
        },
        "environmentPersisted": False,
    }, indent=2))


def resolve_commit(repository: str, branch: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", repository, f"refs/heads/{branch}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "branch was not found"
        raise SystemExit(f"unable to resolve branch {branch!r} in {repository}: {detail}")
    matches = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    if len(matches) != 1 or not re.fullmatch(r"[0-9a-fA-F]{40}", matches[0]):
        raise SystemExit(f"branch {branch!r} did not resolve to one Git commit")
    return matches[0].lower()


def run_id_for(version: str, commit: str) -> str:
    suffix = f"{commit[:10]}-{random.SystemRandom().randrange(16**6):06x}"
    return normalize_name(f"{version}-{suffix}", 63)


def _selector_parts(selector: str) -> tuple[str, str | None]:
    if "--" not in selector:
        return selector, None
    return tuple(selector.rsplit("--", 1))  # type: ignore[return-value]


def selected_executions(
    plan: dict[str, Any],
    selectors: list[str] | None = None,
    exclusions: list[str] | None = None,
) -> list[dict[str, Any]]:
    selectors = selectors or []
    exclusions = exclusions or []
    by_id = {item["id"]: item for item in plan["shards"]}
    selected: list[dict[str, Any]] = []
    if not selectors:
        selected = [copy.deepcopy(item) for item in plan["shards"]]
    else:
        seen: set[tuple[str, str | None]] = set()
        for selector in selectors:
            parent, variant = _selector_parts(selector)
            if parent not in by_id:
                raise ValueError(f"unknown shard selector: {selector}")
            key = (parent, variant)
            if key in seen:
                continue
            seen.add(key)
            shard = copy.deepcopy(by_id[parent])
            if variant is not None:
                variants = [
                    item for item in shard["build"]["variants"]
                    if item["name"] == variant
                ]
                if not variants:
                    raise ValueError(f"unknown variant selector: {selector}")
                shard["build"]["variants"] = variants
                shard["parentShard"] = parent
                shard["id"] = selector
            selected.append(shard)
    excluded_lanes = {value for value in exclusions if "--" not in value}
    excluded_variants: dict[str, set[str]] = {}
    for value in exclusions:
        parent, variant = _selector_parts(value)
        if parent not in by_id:
            raise ValueError(f"unknown excluded shard: {value}")
        if variant:
            names = {item["name"] for item in by_id[parent]["build"]["variants"]}
            if variant not in names:
                raise ValueError(f"unknown excluded variant: {value}")
            excluded_variants.setdefault(parent, set()).add(variant)
    result: list[dict[str, Any]] = []
    for shard in selected:
        parent = shard.get("parentShard", shard["id"])
        if parent in excluded_lanes:
            continue
        blocked = excluded_variants.get(parent, set())
        shard["build"]["variants"] = [
            item for item in shard["build"]["variants"]
            if item["name"] not in blocked
        ]
        if shard["build"]["variants"]:
            result.append(shard)
    if not result:
        raise ValueError("selection leaves no build lanes")
    return result


def matrix_coverage(plan: dict[str, Any], shard_ids: Iterable[str]) -> set[str]:
    by_id = {item["id"]: item for item in plan["shards"]}
    covered: set[str] = set()
    for shard_id in shard_ids:
        parent, variant = _selector_parts(shard_id)
        shard = by_id.get(parent)
        if not shard:
            continue
        if variant is None:
            covered.update(
                f"{parent}--{item['name']}" for item in shard["build"]["variants"]
            )
        elif any(item["name"] == variant for item in shard["build"]["variants"]):
            covered.add(f"{parent}--{variant}")
    return covered


def execution_matrix_coverage(executions: Iterable[dict[str, Any]]) -> set[str]:
    covered: set[str] = set()
    for execution in executions:
        shard = execution["shard"]
        parent = shard.get("parentShard") or _selector_parts(shard["id"])[0]
        covered.update(
            f"{parent}--{variant['name']}"
            for variant in shard["build"]["variants"]
        )
    return covered


def merged_release_provider(existing_manifest: dict[str, Any] | None) -> str:
    if not existing_manifest:
        return "azure"
    return "azure" if existing_manifest.get("provider") == "azure" else "hybrid"


def _normalized_asset(item: dict[str, Any]) -> dict[str, Any]:
    value = dict(item)
    sources = [dict(source) for source in value.pop("sources", [])]
    singular = {
        key: value.pop(key)
        for key in ("provider", "sourceObject", "sourceKey")
        if key in value
    }
    if singular:
        sources.append(singular)
    if sources:
        unique = {
            json.dumps(source, sort_keys=True, separators=(",", ":")): source
            for source in sources
        }
        value["sources"] = [unique[key] for key in sorted(unique)]
    return value


def merge_release_assets(
    existing: Iterable[dict[str, Any]], current: Iterable[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for raw_item in [*existing, *current]:
        item = _normalized_asset(raw_item)
        name = str(item.get("fileName", ""))
        if not name:
            raise RuntimeError("release manifest asset is missing fileName")
        previous = merged.get(name)
        if previous is not None:
            immutable = sorted((set(previous) | set(item)) - {"sources"})
            conflicts = [key for key in immutable if previous.get(key) != item.get(key)]
            if conflicts:
                raise RuntimeError(
                    f"release asset {name!r} conflicts across providers in "
                    f"{', '.join(conflicts)}"
                )
            sources = _normalized_asset({
                **previous,
                "sources": [*previous.get("sources", []), *item.get("sources", [])],
            }).get("sources", [])
            if sources:
                previous["sources"] = sources
            continue
        merged[name] = item
    return merged


def candidate_names(
    plan: dict[str, Any], shard: dict[str, Any], override: str | None = None
) -> list[str]:
    if override:
        return [override]
    explicit = shard.get("machineCandidates")
    if explicit:
        return list(explicit)
    key = "armMachineCandidates" if shard.get("machineClass") == "arm" else "x86MachineCandidates"
    return list(plan["defaults"][key])


def derived_lane_id(shard: dict[str, Any]) -> str:
    """Return a stable compatibility lane for plans that predate explicit lanes."""
    explicit = shard.get("lane")
    if explicit:
        return str(explicit)
    image = shard["image"]
    image_name = f"{image['offer']}-{image['sku']}"
    return normalize_name(
        f"{shard['os']}-{shard['architecture']}-{image_name}",
        63,
    )


def group_execution_lanes(
    plan: dict[str, Any], executions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group selected shards by VM-compatible lane without changing shard identity."""
    lanes: dict[str, dict[str, Any]] = {}
    compatibility_fields = ("os", "architecture", "worker")
    for shard in executions:
        lane_id = derived_lane_id(shard)
        image = copy.deepcopy(shard["image"])
        compatibility = {
            **{name: shard[name] for name in compatibility_fields},
            "image": image,
        }
        lane = lanes.get(lane_id)
        names = candidate_names(plan, shard)
        if lane is None:
            lanes[lane_id] = {
                "id": lane_id,
                **compatibility,
                "candidateNames": names,
                "shards": [shard],
            }
            continue
        actual = {
            **{name: lane[name] for name in compatibility_fields},
            "image": lane["image"],
        }
        if actual != compatibility:
            raise ValueError(
                f"Azure lane {lane_id!r} mixes incompatible OS, architecture, "
                "worker, or Marketplace image contracts"
            )
        allowed = set(names)
        lane["candidateNames"] = [
            name for name in lane["candidateNames"] if name in allowed
        ]
        if not lane["candidateNames"]:
            raise ValueError(
                f"Azure lane {lane_id!r} has no VM candidate shared by all shards"
            )
        lane["shards"].append(shard)
    return list(lanes.values())


def parse_lane_machine_overrides(values: Iterable[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        lane, separator, machine = value.partition("=")
        if not separator or not lane.strip() or not machine.strip():
            raise ValueError("--lane-machine must use LANE=AZURE_VM_SIZE")
        lane = lane.strip()
        if lane in result:
            raise ValueError(f"duplicate --lane-machine override for {lane!r}")
        result[lane] = machine.strip()
    return result


def object_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def sku_capabilities(sku: Any) -> dict[str, str]:
    return {
        str(object_value(item, "name")): str(object_value(item, "value"))
        for item in (object_value(sku, "capabilities", []) or [])
    }


def sku_zones(sku: Any, location: str) -> set[str]:
    zones: set[str] = set()
    for info in object_value(sku, "location_info", []) or []:
        if str(object_value(info, "location", "")).lower() == location.lower():
            zones.update(str(item) for item in (object_value(info, "zones", []) or []))
    return zones


def sku_restricted(sku: Any, location: str, zone: str | None) -> bool:
    requested_location = location.lower()
    requested_zone = str(zone) if zone else None
    for restriction in object_value(sku, "restrictions", []) or []:
        reason = str(object_value(restriction, "reason_code", ""))
        if not reason:
            continue
        restriction_type = str(object_value(restriction, "type", "")).lower()
        info = object_value(restriction, "restriction_info", None)
        raw_locations = object_value(info, "locations", []) or object_value(
            restriction, "values", []
        )
        locations = {str(item).lower() for item in (raw_locations or [])}
        if locations and requested_location not in locations:
            continue
        zones = {
            str(item) for item in (object_value(info, "zones", []) or [])
        }
        if restriction_type == "zone":
            if requested_zone and (not zones or requested_zone in zones):
                return True
            continue
        if restriction_type == "location":
            return True
        # Azure may add restriction types over time. Unknown restrictions that
        # apply to the requested location remain fail-closed.
        return True
    return False


def quota_limits_by_name(
    usage: Iterable[Any],
) -> dict[str, tuple[float, float]]:
    by_name: dict[str, tuple[float, float]] = {}
    for item in usage:
        name = object_value(object_value(item, "name", None), "value", "")
        by_name[str(name).lower()] = (
            float(object_value(item, "current_value", 0)),
            float(object_value(item, "limit", 0)),
        )
    return by_name


def verified_size_options(
    skus: Iterable[Any],
    candidates: list[str],
    architecture: str,
    location: str,
    max_cores: int | None,
    zone: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_name = {
        str(object_value(item, "name")): item
        for item in skus
        if str(object_value(item, "resource_type", "")).lower() == "virtualmachines"
        and location.lower() in {
            str(value).lower() for value in (object_value(item, "locations", []) or [])
        }
    }
    options: list[dict[str, Any]] = []
    rejected: list[str] = []
    for name in candidates:
        sku = by_name.get(name)
        if sku is None:
            rejected.append(f"{name}: unavailable in {location}")
            continue
        caps = sku_capabilities(sku)
        actual_arch = caps.get("CpuArchitectureType", caps.get("CPUArchitecture", "x64"))
        if actual_arch.lower() != architecture.lower():
            rejected.append(f"{name}: architecture {actual_arch}, expected {architecture}")
            continue
        vcpus = int(float(caps.get("vCPUs", caps.get("vCPUsAvailable", "0"))))
        memory = float(caps.get("MemoryGB", "0"))
        if vcpus < 1 or memory <= 0:
            rejected.append(f"{name}: missing vCPU/memory capabilities")
            continue
        if max_cores is not None and vcpus > max_cores:
            rejected.append(f"{name}: {vcpus} vCPUs exceeds --max-cores {max_cores}")
            continue
        zones = sku_zones(sku, location)
        if zone and zone not in zones:
            rejected.append(f"{name}: not offered in availability zone {zone}")
            continue
        if sku_restricted(sku, location, zone):
            rejected.append(f"{name}: subscription restriction")
            continue
        family = str(object_value(sku, "family", ""))
        options.append({
            "name": name,
            "vcpus": vcpus,
            "memoryGiB": memory,
            "family": family,
            "zones": sorted(zones),
        })
    return options, rejected


def choose_size_from_skus(
    skus: Iterable[Any],
    candidates: list[str],
    architecture: str,
    location: str,
    max_cores: int | None,
    zone: str | None,
    quota_limits: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    options, rejected = verified_size_options(
        skus, candidates, architecture, location, max_cores, zone
    )
    for selected in options:
        name = selected["name"]
        vcpus = int(selected["vcpus"])
        family = str(selected["family"])
        if quota_limits is not None:
            total = quota_limits.get("cores")
            if total is None:
                rejected.append(
                    f"{name}: Azure did not return total regional vCPU quota"
                )
                continue
            total_remaining = max(0.0, total[1] - total[0])
            if vcpus > total_remaining:
                rejected.append(
                    f"{name}: total regional quota requires {vcpus} vCPUs, "
                    f"only {total_remaining:g} remain"
                )
                continue
            family_key = family.lower()
            family_limit = quota_limits.get(family_key)
            if not family or family_limit is None:
                rejected.append(
                    f"{name}: Azure did not return "
                    f"{family or 'the SKU VM-family'} quota"
                )
                continue
            family_remaining = max(0.0, family_limit[1] - family_limit[0])
            if vcpus > family_remaining:
                rejected.append(
                    f"{name}: {family} quota requires {vcpus} vCPUs, "
                    f"only {family_remaining:g} remain"
                )
                continue
        return selected
    raise RuntimeError(
        "no Azure VM size satisfies the lane: " + "; ".join(rejected)
    )


def choose_parallel_lane_machines(
    skus: Iterable[Any],
    lanes: list[dict[str, Any]],
    location: str,
    max_cores: int | None,
    max_total_cores: int | None,
    zone: str | None,
    quota_limits: dict[str, tuple[float, float]],
    machine_type: str | None = None,
    lane_machine_values: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Choose one size per lane under aggregate regional and family quotas."""
    overrides = parse_lane_machine_overrides(lane_machine_values)
    lane_ids = {lane["id"] for lane in lanes}
    unknown = sorted(set(overrides) - lane_ids)
    if unknown:
        raise ValueError(
            "--lane-machine references unselected Azure lane(s): "
            + ", ".join(unknown)
        )
    total_quota = quota_limits.get("cores")
    if total_quota is None:
        raise RuntimeError("Azure did not return total regional vCPU quota")
    total_remaining = max(0.0, total_quota[1] - total_quota[0])
    total_budget = min(
        total_remaining,
        float(max_total_cores) if max_total_cores is not None else total_remaining,
    )

    all_options: list[list[dict[str, Any]]] = []
    rejection_details: list[str] = []
    for lane in lanes:
        forced = overrides.get(lane["id"]) or machine_type
        candidates = [forced] if forced else list(lane["candidateNames"])
        architecture = "Arm64" if lane["architecture"] == "arm64" else "x64"
        options, rejected = verified_size_options(
            skus, candidates, architecture, location, max_cores, zone
        )
        quota_eligible: list[dict[str, Any]] = []
        for option in options:
            family = str(option.get("family", "")).lower()
            if not family or family not in quota_limits:
                rejected.append(
                    f"{option['name']}: Azure did not return "
                    f"{option.get('family') or 'the SKU VM-family'} quota"
                )
                continue
            quota_eligible.append(option)
        if not quota_eligible:
            detail = "; ".join(rejected) or "no candidates"
            raise RuntimeError(
                f"no Azure VM size satisfies concurrent lane {lane['id']}: {detail}"
            )
        all_options.append(quota_eligible)
        rejection_details.append(
            f"{lane['id']}=[{', '.join(item['name'] for item in quota_eligible)}]"
        )

    best: tuple[Any, ...] | None = None
    best_combination: tuple[dict[str, Any], ...] | None = None
    for combination in itertools.product(*all_options):
        cores = [int(item["vcpus"]) for item in combination]
        total = sum(cores)
        if total > total_budget:
            continue
        required_by_family: dict[str, int] = {}
        eligible = True
        for item in combination:
            family = str(item["family"]).lower()
            required_by_family[family] = (
                required_by_family.get(family, 0) + int(item["vcpus"])
            )
        for family, required in required_by_family.items():
            current, limit = quota_limits[family]
            if required > max(0.0, limit - current):
                eligible = False
                break
        if not eligible:
            continue
        ranks = tuple(
            -all_options[index].index(item)
            for index, item in enumerate(combination)
        )
        # Favor useful capacity on every box first, then total throughput, then
        # a balanced split and finally the release-plan candidate preference.
        score = (
            min(cores),
            total,
            math.prod(cores),
            ranks,
        )
        if best is None or score > best:
            best = score
            best_combination = combination
    if best_combination is None:
        cap = (
            f", --max-total-cores={max_total_cores}"
            if max_total_cores is not None
            else ""
        )
        raise RuntimeError(
            "no concurrent Azure VM combination fits current total-regional "
            f"and VM-family quota (total remaining={total_remaining:g}{cap}); "
            + "; ".join(rejection_details)
        )
    return [copy.deepcopy(item) for item in best_combination]


def adapt_build_resources(
    shard: dict[str, Any],
    vcpus: int,
    memory_gib: float,
    threads_override: int | None,
) -> None:
    plan_threads = int(shard["build"].get("buildThreads", 32))
    threads = threads_override or min(plan_threads, max(1, vcpus // 2))
    plan_heap = int(shard["build"].get("mavenHeapGiB", 24))
    heap = min(plan_heap, max(2, int(memory_gib) - 4))
    shard["build"]["buildThreads"] = threads
    shard["build"]["mavenHeapGiB"] = heap


def quota_report(usage: Iterable[Any], lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = quota_limits_by_name(usage)
    required: dict[str, int] = {}
    concurrent_total = 0
    for item in lanes:
        family = str(item["selectedMachine"].get("family", "")).lower()
        vcpus = int(item["selectedMachine"]["vcpus"])
        concurrent_total += vcpus
        if family:
            required[family] = required.get(family, 0) + vcpus
    report: list[dict[str, Any]] = []
    failures: list[str] = []
    total_current, total_limit = by_name.get("cores", (0.0, 0.0))
    total_remaining = max(0.0, total_limit - total_current)
    report.append({
        "family": "cores",
        "requiredConcurrentPeak": concurrent_total,
        "current": total_current,
        "limit": total_limit,
        "remaining": total_remaining,
    })
    if "cores" not in by_name:
        failures.append("cores: total regional vCPU quota was not returned by Azure")
    elif concurrent_total > total_remaining:
        failures.append(
            f"cores: requires {concurrent_total}, only {total_remaining:g} "
            "total regional vCPUs remain"
        )
    for family, needed in sorted(required.items()):
        current, limit = by_name.get(family, (0.0, 0.0))
        remaining = max(0.0, limit - current)
        row = {
            "family": family,
            "requiredConcurrentPeak": needed,
            "current": current,
            "limit": limit,
            "remaining": remaining,
        }
        report.append(row)
        if family not in by_name:
            failures.append(f"{family}: quota was not returned by Azure")
        elif needed > remaining:
            failures.append(f"{family}: requires {needed}, only {remaining:g} vCPUs remain")
    if failures:
        raise RuntimeError("Azure VM-family quota is insufficient: " + "; ".join(failures))
    return report


def is_not_found(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 404 or exc.__class__.__name__ == "ResourceNotFoundError"


def azure_resource_id(
    context: dict[str, Any],
    group: str,
    provider: str,
    resource_type: str,
    name: str,
    resource: Any,
    description: str,
) -> str:
    """Return an SDK resource ID, deriving it when an Azure LRO omits the field."""
    resource_id = getattr(resource, "id", None)
    if resource_id is None and isinstance(resource, dict):
        resource_id = resource.get("id")
    if resource_id:
        return str(resource_id)
    subscription = context.get("subscription")
    if not subscription:
        raise RuntimeError(
            f"Azure {description} result omitted its resource ID and the "
            "subscription is unavailable"
        )
    return (
        f"/subscriptions/{subscription}/resourceGroups/{group}/providers/"
        f"{provider}/{resource_type}/{name}"
    )


def is_transient_delete_error(exc: BaseException) -> bool:
    """Recognize retryable Azure dependency, throttling, and service failures."""
    if isinstance(exc, TimeoutError):
        return True
    try:
        status = int(getattr(exc, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status = 0
    if status in {408, 429, 500, 502, 503, 504}:
        return True
    code = str(getattr(exc, "error_code", "") or "").replace(" ", "").lower()
    compact = str(exc).replace(" ", "").lower()
    dependency_markers = (
        "publicipaddresscannotbedeleted",
        "networkinterfacecannotbedeleted",
    )
    return (
        any(marker in code or marker in compact for marker in dependency_markers)
        or (
            "cannotbedeleted" in compact
            and any(
                marker in compact
                for marker in ("stillallocated", "inuse", "usedby")
            )
        )
    )


def resolve_marketplace_image_version(
    images: Any, location: str, image: dict[str, Any]
) -> str:
    coordinates = (
        location,
        image["publisher"],
        image["offer"],
        image["sku"],
    )
    version = str(image["version"])
    if version.lower() == "latest":
        versions = list(
            images.list(*coordinates, top=1, orderby="name desc")
        )
        if not versions:
            raise RuntimeError(
                "no Azure Marketplace image versions found for "
                f"{image['publisher']}:{image['offer']}:{image['sku']} "
                f"in {location}"
            )
        version = str(object_value(versions[0], "name", ""))
        if not version or version.lower() == "latest":
            raise RuntimeError(
                "Azure returned an invalid Marketplace image version for "
                f"{image['publisher']}:{image['offer']}:{image['sku']} "
                f"in {location}"
            )
    images.get(*coordinates, version)
    return version


def preflight_data(args: argparse.Namespace, *, include_context: bool = False) -> dict[str, Any]:
    plan = load_plan(args.plan)
    executions = selected_executions(plan, args.shard, args.exclude_shard)
    build_threads = getattr(args, "build_threads", None)
    if build_threads is not None and build_threads < 1:
        raise ValueError("--build-threads must be positive")
    context = cloud_context(
        args.subscription, allow_wizard=not getattr(args, "no_wizard", False)
    )
    location = resolve_location(
        args.location, allow_wizard=not getattr(args, "no_wizard", False)
    )
    known_locations = {
        str(item.name).lower()
        for item in context["subscriptions"].subscriptions.list_locations(
            context["subscription"]
        )
    }
    if location not in known_locations:
        raise RuntimeError(f"Azure location {location!r} is not enabled for this subscription")
    if getattr(args, "zone", None) and args.zone not in {"1", "2", "3"}:
        raise ValueError("--zone must be 1, 2, or 3")
    root_gib = getattr(args, "root_volume_gib", None) or int(plan["defaults"]["rootVolumeGiB"])
    if root_gib < 64 or root_gib > 4095:
        raise ValueError("Azure OS disk size must be between 64 and 4095 GiB")
    skus = list(context["compute"].resource_skus.list())
    quota_usage = list(context["compute"].usage.list(location))
    quota_limits = quota_limits_by_name(quota_usage)
    lane_specs = group_execution_lanes(plan, executions)
    selected_machines = choose_parallel_lane_machines(
        skus,
        lane_specs,
        location,
        getattr(args, "max_cores", None),
        getattr(args, "max_total_cores", None),
        getattr(args, "zone", None),
        quota_limits,
        getattr(args, "machine_type", None),
        getattr(args, "lane_machine", None),
    )
    resolved: list[dict[str, Any]] = []
    resolved_lanes: list[dict[str, Any]] = []
    for lane, selected in zip(lane_specs, selected_machines):
        image = lane["image"]
        image["version"] = resolve_marketplace_image_version(
            context["compute"].virtual_machine_images,
            location,
            image,
        )
        execution_ids: list[str] = []
        for shard in lane["shards"]:
            shard["image"]["version"] = image["version"]
            adapt_build_resources(
                shard,
                selected["vcpus"],
                selected["memoryGiB"],
                build_threads,
            )
            execution_ids.append(shard["id"])
            resolved.append({
                "id": shard["id"],
                "parentShard": shard.get("parentShard", shard["id"]),
                "laneId": lane["id"],
                "shard": shard,
                "selectedMachine": copy.deepcopy(selected),
                "rootVolumeGiB": root_gib,
                "zone": getattr(args, "zone", None),
            })
        resolved_lanes.append({
            "id": lane["id"],
            "os": lane["os"],
            "architecture": lane["architecture"],
            "worker": lane["worker"],
            "image": copy.deepcopy(image),
            "executionIds": execution_ids,
            "selectedMachine": selected,
            "rootVolumeGiB": root_gib,
            "zone": getattr(args, "zone", None),
        })
    quota = quota_report(quota_usage, resolved_lanes)
    group = resource_group_name(location, getattr(args, "resource_group", None))
    account = storage_account_name(
        context["subscription"], location, getattr(args, "storage_account", None)
    )
    try:
        context["storage"].storage_accounts.get_properties(group, account)
        storage_state = "existing"
    except Exception as exc:
        if not is_not_found(exc):
            raise
        availability = context["storage"].storage_accounts.check_name_availability(
            {"name": account}
        )
        if not bool(object_value(availability, "name_available", False)):
            raise RuntimeError(
                f"Azure storage account {account!r} is unavailable: "
                f"{object_value(availability, 'reason', 'unknown reason')}"
            )
        storage_state = "available"
    result = {
        "schemaVersion": 1,
        "provider": "azure",
        "subscription": context["subscription"],
        "location": location,
        "resourceGroup": group,
        "storageAccount": account,
        "storageAccountState": storage_state,
        "serial": False,
        "parallel": True,
        "laneCount": len(resolved_lanes),
        "executionCount": len(resolved),
        "lanes": resolved_lanes,
        "executions": resolved,
        "quota": quota,
        "unsupportedWorkflows": plan.get("unsupportedWorkflows", {}),
    }
    if include_context:
        result["context"] = context
        result["plan"] = plan
    return result


def printable_preflight(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"context", "plan"}}


def preflight(args: argparse.Namespace) -> None:
    print(json.dumps(printable_preflight(preflight_data(args)), indent=2))


def ensure_resource_group(context: dict[str, Any], group: str, location: str) -> Any:
    return context["resource"].resource_groups.create_or_update(
        group,
        {
            "location": location,
            "tags": {MANAGED_TAG: "true", "dl4j-provider": "azure"},
        },
    )


def ensure_storage(
    context: dict[str, Any],
    group: str,
    location: str,
    account_name: str,
    plan: dict[str, Any],
) -> tuple[Any, Any, str]:
    storage = context["storage"].storage_accounts
    try:
        account = storage.get_properties(group, account_name)
    except Exception as exc:
        if not is_not_found(exc):
            raise
        account = storage.begin_create(
            group,
            account_name,
            {
                "location": location,
                "kind": "StorageV2",
                "sku": {"name": "Standard_LRS"},
                "allow_blob_public_access": False,
                "minimum_tls_version": "TLS1_2",
                "public_network_access": "Enabled",
                "tags": {MANAGED_TAG: "true", "dl4j-provider": "azure"},
            },
        ).result(timeout=1800)
    keys = storage.list_keys(group, account_name)
    values = list(object_value(keys, "keys", []) or [])
    if not values:
        raise RuntimeError(f"Azure storage account {account_name} returned no access keys")
    key = str(object_value(values[0], "value"))
    modules = context["modules"]
    service = modules["BlobServiceClient"](
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=key,
    )
    for name in (artifact_container_name(plan), control_container_name(plan)):
        container = service.get_container_client(name)
        try:
            container.create_container(metadata={"dl4j_release_managed": "true"})
        except Exception as exc:
            if exc.__class__.__name__ not in {"ResourceExistsError", "ContainerAlreadyExists"}:
                raise
    return account, service, key


def ensure_identity(
    context: dict[str, Any],
    group: str,
    location: str,
    run_id: str,
    storage_scope: str,
    fence_check: Callable[[], None] | None = None,
    controller_epoch: str | None = None,
) -> tuple[Any, dict[str, str]]:
    check = fence_check or (lambda: None)
    resource_run_id = (
        f"{run_id}-{controller_epoch}" if controller_epoch else run_id
    )
    name = resource_name("dl4j-release-identity", resource_run_id, maximum=64)
    tags = {MANAGED_TAG: "true", RUN_TAG: run_id}
    if controller_epoch:
        tags[CONTROLLER_EPOCH_TAG] = controller_epoch
    check()
    identity = context["identity"].user_assigned_identities.create_or_update(
        group,
        name,
        {
            "location": location,
            "tags": tags,
        },
    )
    check()
    role_definition = (
        f"/subscriptions/{context['subscription']}/providers/"
        f"Microsoft.Authorization/roleDefinitions/{STORAGE_BLOB_DATA_CONTRIBUTOR}"
    )
    assignment_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{storage_scope}/{identity.principal_id}/{STORAGE_BLOB_DATA_CONTRIBUTOR}",
        )
    )
    for attempt in range(12):
        check()
        try:
            context["authorization"].role_assignments.create(
                storage_scope,
                assignment_id,
                {
                    "role_definition_id": role_definition,
                    "principal_id": identity.principal_id,
                    "principal_type": "ServicePrincipal",
                },
            )
        except Exception as exc:
            check()
            if getattr(exc, "status_code", None) == 409:
                break
            principal_pending = (
                getattr(exc, "status_code", None) == 400
                and "principalnotfound" in str(exc).replace(" ", "").lower()
            )
            if not principal_pending or attempt == 11:
                check()
                try:
                    cleanup_operation = (
                        context["identity"].user_assigned_identities.delete(group, name)
                    )
                except Exception as cleanup_exc:
                    check()
                    if not is_not_found(cleanup_exc):
                        raise RuntimeError(
                            "managed identity creation failed and rollback also failed: "
                            f"{cleanup_exc}"
                        ) from exc
                else:
                    check()
                    try:
                        wait_operation(cleanup_operation)
                    except Exception as cleanup_exc:
                        check()
                        if not is_not_found(cleanup_exc):
                            raise RuntimeError(
                                "managed identity creation failed and rollback also failed: "
                                f"{cleanup_exc}"
                            ) from exc
                    check()
                raise
            time.sleep(5)
            check()
        else:
            check()
            break
    metadata = {
        "name": name,
        "id": str(identity.id),
        "clientId": str(identity.client_id),
        "principalId": str(identity.principal_id),
        "roleAssignmentId": assignment_id,
        "roleAssignmentScope": storage_scope,
    }
    if controller_epoch:
        metadata["controllerEpoch"] = controller_epoch
    return identity, metadata


def wait_operation(value: Any, timeout: int = 1800) -> None:
    if hasattr(value, "result"):
        value.result(timeout=timeout)


def fenced_azure_operation(
    begin: Callable[[], Any],
    fence_check: Callable[[], None] | None = None,
    *,
    timeout: int,
) -> Any:
    """Fence both submission and completion of a potentially long Azure LRO."""
    check = fence_check or (lambda: None)
    check()
    operation = begin()
    check()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Azure operation exceeded {timeout} seconds")
        try:
            result = operation.result(timeout=min(5, remaining))
        except TimeoutError:
            check()
            continue
        check()
        return result


def delete_run_identity(
    context: dict[str, Any],
    group: str,
    metadata: dict[str, str],
    fence_check: Callable[[], None] | None = None,
) -> list[str]:
    errors: list[str] = []
    check = fence_check or (lambda: None)
    check()
    try:
        wait_operation(
            context["authorization"].role_assignments.delete(
                metadata["roleAssignmentScope"], metadata["roleAssignmentId"]
            )
        )
    except Exception as exc:
        if not is_not_found(exc):
            errors.append(f"role assignment cleanup: {exc}")
    check()
    try:
        wait_operation(
            context["identity"].user_assigned_identities.delete(group, metadata["name"])
        )
    except Exception as exc:
        if not is_not_found(exc):
            errors.append(f"managed identity cleanup: {exc}")
    check()
    return errors


def cleanup_managed_identities(
    context: dict[str, Any],
    group: str,
    storage_scope: str,
    run_id: str | None = None,
    fence_check: Callable[[], None] | None = None,
) -> tuple[list[str], list[str]]:
    deleted: list[str] = []
    errors: list[str] = []
    check = fence_check or (lambda: None)
    check()
    try:
        identities = list(
            context["identity"].user_assigned_identities.list_by_resource_group(group)
        )
    except Exception as exc:
        if is_not_found(exc):
            return deleted, errors
        return deleted, [f"managed identity discovery: {exc}"]
    check()
    for identity in identities:
        check()
        tags = object_value(identity, "tags", {}) or {}
        if tags.get(MANAGED_TAG) != "true":
            continue
        if run_id is not None and tags.get(RUN_TAG) != run_id:
            continue
        principal_id = str(object_value(identity, "principal_id", ""))
        if not principal_id:
            errors.append(f"managed identity {identity.name}: missing principal ID")
            continue
        assignment_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{storage_scope}/{principal_id}/{STORAGE_BLOB_DATA_CONTRIBUTOR}",
            )
        )
        metadata = {
            "name": str(identity.name),
            "roleAssignmentId": assignment_id,
            "roleAssignmentScope": storage_scope,
        }
        item_errors = delete_run_identity(
            context, group, metadata, fence_check=check
        )
        if item_errors:
            errors.extend(f"{identity.name}: {item}" for item in item_errors)
        else:
            deleted.append(str(identity.name))
    return deleted, errors


def managed_run_resource_names(
    context: dict[str, Any],
    group: str,
    run_id: str,
    fence_check: Callable[[], None] | None = None,
) -> dict[str, list[str]]:
    """Inventory every run-tagged ephemeral resource under the controller fence."""
    check = fence_check or (lambda: None)
    sources = [
        (
            "virtualMachines",
            lambda: context["compute"].virtual_machines.list(group),
        ),
        (
            "networkInterfaces",
            lambda: context["network"].network_interfaces.list(group),
        ),
        (
            "publicIps",
            lambda: context["network"].public_ip_addresses.list(group),
        ),
        (
            "disks",
            lambda: context["compute"].disks.list_by_resource_group(group),
        ),
    ]
    inventory: dict[str, list[str]] = {}
    for key, list_resources in sources:
        check()
        try:
            values = list(list_resources())
        except Exception as exc:
            check()
            if is_not_found(exc):
                inventory[key] = []
                continue
            raise RuntimeError(f"{key} discovery failed: {exc}") from exc
        check()
        names = []
        for value in values:
            check()
            tags = object_value(value, "tags", {}) or {}
            name = str(object_value(value, "name", ""))
            if (
                tags.get(MANAGED_TAG) == "true"
                and tags.get(RUN_TAG) == run_id
            ):
                if name:
                    names.append(name)
        inventory[key] = sorted(names)
    check()
    return inventory


def reconcile_managed_run_resources(
    context: dict[str, Any],
    group: str,
    run_id: str,
    fence_check: Callable[[], None] | None = None,
    timeout_seconds: float = RESOURCE_RECONCILE_TIMEOUT_SECONDS,
    retry_seconds: float = RESOURCE_CLEANUP_RETRY_SECONDS,
) -> tuple[dict[str, list[str]], list[str]]:
    """Delete run resources repeatedly until Azure's inventory converges to zero."""
    check = fence_check or (lambda: None)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    phases = [
        (
            "virtualMachines",
            "VM",
            lambda name: context["compute"].virtual_machines.begin_delete(
                group, name
            ),
        ),
        (
            "networkInterfaces",
            "NIC",
            lambda name: context["network"].network_interfaces.begin_delete(
                group, name
            ),
        ),
        (
            "publicIps",
            "public IP",
            lambda name: context["network"].public_ip_addresses.begin_delete(
                group, name
            ),
        ),
        (
            "disks",
            "OS disk",
            lambda name: context["compute"].disks.begin_delete(group, name),
        ),
    ]
    last_errors: dict[str, str] = {}
    while True:
        try:
            inventory = managed_run_resource_names(
                context,
                group,
                run_id,
                fence_check=check,
            )
        except Exception as exc:
            return {}, [f"run resource reconciliation: {exc}"]
        if not any(inventory.values()):
            return inventory, []
        for key, label, begin_delete in phases:
            for name in inventory[key]:
                check()
                error_key = f"{label} {name}"
                remaining_budget = deadline - time.monotonic()
                if remaining_budget < 1:
                    break
                try:
                    fenced_azure_operation(
                        lambda name=name, begin_delete=begin_delete: begin_delete(name),
                        check,
                        timeout=min(1800, int(remaining_budget)),
                    )
                    last_errors.pop(error_key, None)
                except Exception as exc:
                    check()
                    if is_not_found(exc):
                        last_errors.pop(error_key, None)
                    elif not is_transient_delete_error(exc):
                        return inventory, [f"{error_key}: {exc}"]
                    else:
                        last_errors[error_key] = str(exc)
        try:
            remaining = managed_run_resource_names(
                context,
                group,
                run_id,
                fence_check=check,
            )
        except Exception as exc:
            return inventory, [f"run resource reconciliation: {exc}"]
        if not any(remaining.values()):
            return remaining, []
        if time.monotonic() >= deadline:
            names = ", ".join(
                f"{key}={','.join(values)}"
                for key, values in remaining.items()
                if values
            )
            errors = [
                "run resource cleanup deadline exceeded; remaining " + names
            ]
            errors.extend(
                f"{label}: {message}"
                for label, message in sorted(last_errors.items())
            )
            return remaining, errors
        check()
        time.sleep(min(retry_seconds, max(0.0, deadline - time.monotonic())))
        check()


def ensure_network(
    context: dict[str, Any],
    group: str,
    location: str,
    fence_check: Callable[[], None] | None = None,
) -> tuple[str, str]:
    network = context["network"]
    tags = {MANAGED_TAG: "true", "dl4j-provider": "azure"}
    vnet_name = "dl4j-release-vnet"
    nsg_name = "dl4j-release-nsg"
    nsg = fenced_azure_operation(
        lambda: network.network_security_groups.begin_create_or_update(
            group,
            nsg_name,
            {"location": location, "security_rules": [], "tags": tags},
        ),
        fence_check,
        timeout=600,
    )
    nsg_id = object_value(nsg, "id")
    if not nsg_id:
        subscription = str(context.get("subscription") or "").strip()
        if not subscription:
            raise RuntimeError(
                "Azure network security group result omitted its resource ID "
                "and the subscription is unavailable"
            )
        nsg_id = (
            f"/subscriptions/{subscription}/resourceGroups/{group}"
            f"/providers/Microsoft.Network/networkSecurityGroups/{nsg_name}"
        )
    vnet = fenced_azure_operation(
        lambda: network.virtual_networks.begin_create_or_update(
            group,
            vnet_name,
            {
                "location": location,
                "address_space": {"address_prefixes": ["10.78.0.0/16"]},
                "tags": tags,
            },
        ),
        fence_check,
        timeout=600,
    )
    subnet_name = "builders"
    subnet = fenced_azure_operation(
        lambda: network.subnets.begin_create_or_update(
            group,
            vnet_name,
            subnet_name,
            {
                "address_prefix": "10.78.0.0/24",
                "network_security_group": {"id": nsg_id},
            },
        ),
        fence_check,
        timeout=600,
    )
    subnet_id = azure_resource_id(
        context,
        group,
        "Microsoft.Network",
        f"virtualNetworks/{vnet_name}/subnets",
        subnet_name,
        subnet,
        "subnet",
    )
    return subnet_id, nsg_id


def get_json(container: Any, name: str) -> dict[str, Any] | None:
    try:
        return json.loads(container.download_blob(name).readall().decode("utf-8"))
    except Exception as exc:
        if is_not_found(exc) or exc.__class__.__name__ == "ResourceNotFoundError":
            return None
        raise


def controller_epoch(controller_lease: Any) -> str:
    value = getattr(controller_lease, "epoch", None)
    if not isinstance(value, str) or not value:
        value = uuid.uuid4().hex
        controller_lease.epoch = value
    return value


def put_json(
    container: Any,
    name: str,
    value: dict[str, Any],
    modules: dict[str, Any],
    *,
    controller_lease: Any | None = None,
    create_only: bool = False,
) -> None:
    """Write JSON, using a blob lease or ETag CAS for controller-owned state."""
    payload_value = copy.deepcopy(value)
    upload_options: dict[str, Any] = {"overwrite": not create_only}
    if controller_lease is not None:
        controller_lease.check()
        epoch = controller_epoch(controller_lease)
        existing_epoch = payload_value.get("controllerEpoch")
        if existing_epoch not in {None, epoch}:
            raise RuntimeError(
                f"refusing to write controller state owned by epoch {existing_epoch}"
            )
        payload_value["controllerEpoch"] = epoch
        if name == getattr(controller_lease, "name", None):
            # The global controller lease lives on the kill-switch blob itself.
            # Supplying the lease makes Azure reject writes immediately after a
            # takeover, closing the old check/write race for worker cancellation.
            upload_options["lease"] = controller_lease.lease
        elif not create_only:
            downloader = container.download_blob(name)
            current = json.loads(downloader.readall().decode("utf-8"))
            if current.get("controllerEpoch") != epoch:
                raise RuntimeError(
                    f"controller epoch changed for Azure Blob {name!r}"
                )
            properties = object_value(downloader, "properties", {}) or {}
            etag = object_value(properties, "etag", "")
            if not etag:
                raise RuntimeError(
                    f"Azure Blob {name!r} did not expose an ETag for fenced update"
                )
            upload_options["etag"] = etag
            upload_options["match_condition"] = modules[
                "MatchConditions"
            ].IfNotModified
    container.upload_blob(
        name,
        json.dumps(payload_value, indent=2, sort_keys=True).encode("utf-8"),
        content_settings=modules["ContentSettings"](content_type="application/json"),
        **upload_options,
    )
    if controller_lease is not None:
        controller_lease.check()


def set_kill_switch(
    container: Any,
    plan: dict[str, Any],
    enabled: bool,
    modules: dict[str, Any],
    reason: str = "controller",
    *,
    controller_lease: Any | None = None,
    force: bool = False,
    object_name: str | None = None,
) -> None:
    put_json(container, object_name or kill_switch_blob(plan), {
        "enabled": bool(enabled),
        "updatedAt": utc_now(),
        "reason": reason,
        "force": bool(force),
    }, modules, controller_lease=controller_lease)


def kill_switch_enabled(
    container: Any,
    plan: dict[str, Any],
    expected_controller_epoch: str | None = None,
    *,
    object_name: str | None = None,
) -> bool:
    value = get_json(container, object_name or kill_switch_blob(plan))
    if value is None or not isinstance(value.get("enabled"), bool):
        raise RuntimeError("Azure release kill switch is missing or malformed")
    if (
        expected_controller_epoch
        and value.get("controllerEpoch") != expected_controller_epoch
        and value.get("force") is not True
    ):
        raise RuntimeError("Azure release kill switch controller epoch does not match")
    return bool(value["enabled"])


def emergency_kill_switch_enabled(
    container: Any,
    plan: dict[str, Any],
) -> bool:
    value = get_json(container, kill_switch_blob(plan))
    if value is None or not isinstance(value.get("enabled"), bool):
        raise RuntimeError("Azure global emergency kill switch is missing or malformed")
    # Normal run-scoped cancellation never sets force. This also lets a new
    # controller coexist with a legacy controller that still owns this blob.
    return value["enabled"] is True and value.get("force") is True


def assert_emergency_kill_switch_disabled(
    container: Any,
    plan: dict[str, Any],
) -> None:
    if emergency_kill_switch_enabled(container, plan):
        raise RuntimeError("Azure global emergency kill switch was enabled")


def prepare_emergency_kill_switch(
    container: Any,
    plan: dict[str, Any],
    modules: dict[str, Any],
    reset_requested: bool,
) -> None:
    value = get_json(container, kill_switch_blob(plan))
    malformed = value is None or not isinstance(value.get("enabled"), bool)
    forced = (
        not malformed
        and value.get("enabled") is True
        and value.get("force") is True
    )
    if malformed or forced:
        if not reset_requested:
            raise RuntimeError(
                "Azure global emergency kill switch is enabled, missing, or malformed; "
                "pass --reset-kill-switch only after confirming emergency shutdown is complete"
            )
        emergency_lease = ControllerLease(
            container, controller_lock_blob(plan)
        ).acquire()
        try:
            set_kill_switch(
                container,
                plan,
                False,
                modules,
                "start-reset-global-emergency",
                controller_lease=emergency_lease,
            )
        finally:
            release_errors = emergency_lease.release()
        if release_errors:
            raise RuntimeError(
                "Azure global emergency-switch lease cleanup failed: "
                + "; ".join(release_errors)
            )
    assert_emergency_kill_switch_disabled(container, plan)


class ControllerLease:
    """Renewable Blob lease that serializes one Azure release mutation scope."""

    def __init__(self, container: Any, name: str, *, duration: int = 60) -> None:
        self.container = container
        self.blob = container.get_blob_client(name)
        self.name = name
        self.duration = duration
        self.epoch = uuid.uuid4().hex
        self.lease: Any | None = None
        self.failure: BaseException | None = None
        self.external_check: Callable[[], None] | None = None
        self._stop = threading.Event()
        self._renew_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def acquire(self) -> "ControllerLease":
        try:
            self.blob.upload_blob(
                json.dumps(
                    {
                        "enabled": True,
                        "updatedAt": utc_now(),
                        "reason": "controller-lock-initialized",
                        "force": True,
                        "controllerEpoch": None,
                    },
                    sort_keys=True,
                ).encode("utf-8"),
                overwrite=False,
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) != 409 and exc.__class__.__name__ not in {
                "ResourceExistsError",
                "BlobAlreadyExists",
            }:
                raise
        try:
            self.lease = self.blob.acquire_lease(lease_duration=self.duration)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 409:
                raise RuntimeError(
                    f"another Azure release controller already holds Blob lease {self.name!r}"
                ) from exc
            raise
        self._thread = threading.Thread(
            target=self._renew_loop,
            name="dl4j-azure-controller-lease",
            daemon=True,
        )
        self._thread.start()
        return self

    def _renew_loop(self) -> None:
        while not self._stop.wait(max(5, self.duration // 3)):
            try:
                with self._renew_lock:
                    self.lease.renew()
            except BaseException as exc:  # surfaced by check() on the controller thread
                self.failure = exc
                return

    def check(self) -> None:
        if self.failure is not None:
            raise RuntimeError("Azure controller Blob lease renewal failed") from self.failure
        try:
            with self._renew_lock:
                self.lease.renew()
        except BaseException as exc:
            self.failure = exc
            raise RuntimeError("Azure controller Blob lease renewal failed") from exc
        if self.external_check is not None:
            self.external_check()

    def assert_healthy(self) -> None:
        """Surface lease or global-emergency failure without forcing renewal."""
        if self.failure is not None:
            raise RuntimeError("Azure controller Blob lease renewal failed") from self.failure
        if self.external_check is not None:
            self.external_check()

    def release(self) -> list[str]:
        errors: list[str] = []
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.lease is not None:
            try:
                self.lease.release()
            except Exception as exc:
                if getattr(exc, "status_code", None) not in {404, 409}:
                    errors.append(f"controller lease release: {exc}")
        return errors


class ControllerLeaseGroup:
    """Hold a primary administrative fence plus every active run fence."""

    def __init__(
        self, primary: ControllerLease, additional: Iterable[ControllerLease]
    ) -> None:
        self.primary = primary
        self.additional = list(additional)
        self.name = primary.name
        self.epoch = primary.epoch
        self.lease = primary.lease

    def check(self) -> None:
        self.primary.check()
        for lease in self.additional:
            lease.check()

    def assert_healthy(self) -> None:
        self.primary.assert_healthy()
        for lease in self.additional:
            lease.assert_healthy()

    def release(self) -> list[str]:
        errors: list[str] = []
        for lease in reversed(self.additional):
            errors.extend(lease.release())
        errors.extend(self.primary.release())
        return errors


def render_worker(worker_path: Path, config: dict[str, Any]) -> bytes:
    content = worker_path.read_text(encoding="utf-8")
    replacements = {
        "__DL4J_WORKER_CONFIG_B64__": base64.b64encode(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).decode("ascii"),
        "__DL4J_BUILD_DRIVER_B64__": base64.b64encode(
            BUILD_DRIVER.read_bytes()
        ).decode("ascii"),
        "__DL4J_CLOUD_IO_B64__": base64.b64encode(
            CLOUD_IO.read_bytes()
        ).decode("ascii"),
    }
    for marker, value in replacements.items():
        content = content.replace(marker, value)
    unresolved = re.findall(r"__DL4J_[A-Z0-9_]+__", content)
    if unresolved:
        raise RuntimeError(f"unresolved worker placeholders: {sorted(set(unresolved))}")
    return content.encode("utf-8")


def worker_sas_url(
    context: dict[str, Any],
    service: Any,
    account_name: str,
    account_key: str,
    container_name: str,
    blob_name: str,
    timeout_hours: int,
) -> str:
    modules = context["modules"]
    token = modules["generate_blob_sas"](
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=modules["BlobSasPermissions"](read=True),
        expiry=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=timeout_hours + 2),
    )
    return service.get_blob_client(container_name, blob_name).url + "?" + token


def compiler_cache_config(
    context: dict[str, Any],
    account_name: str,
    account_key: str,
    plan: dict[str, Any],
    timeout_hours: int,
) -> dict[str, str]:
    modules = context["modules"]
    container = artifact_container_name(plan)
    token = modules["generate_container_sas"](
        account_name=account_name,
        container_name=container,
        account_key=account_key,
        permission=modules["ContainerSasPermissions"](
            read=True, write=True, create=True
        ),
        expiry=dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=timeout_hours + 2),
    ).lstrip("?")
    return {
        **compiler_cache_metadata(plan, account_name),
        "connectionString": (
            f"BlobEndpoint=https://{account_name}.blob.core.windows.net;"
            f"SharedAccessSignature={token}"
        ),
    }


def bootstrap_user_data(url: str) -> str:
    quoted = shlex.quote(url)
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
install -d -m 700 /opt/dl4j-release/bootstrap
python3 -c 'import sys,urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' {quoted} /opt/dl4j-release/bootstrap/worker.sh
chmod 700 /opt/dl4j-release/bootstrap/worker.sh
cat >/etc/systemd/system/dl4j-release-worker.service <<'DL4J_UNIT'
[Unit]
Description=DL4J Azure release worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/env bash /opt/dl4j-release/bootstrap/worker.sh
Restart=on-failure
RestartSec=15
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
DL4J_UNIT
systemctl daemon-reload
systemctl enable --now dl4j-release-worker.service
"""


def resolve_ssh_public_key(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        return path.read_text(encoding="utf-8").strip() if path.is_file() else explicit.strip()
    environment = os.environ.get("AZURE_SSH_PUBLIC_KEY")
    if environment:
        path = Path(environment).expanduser()
        return path.read_text(encoding="utf-8").strip() if path.is_file() else environment.strip()
    for candidate in (Path.home() / ".ssh/id_ed25519.pub", Path.home() / ".ssh/id_rsa.pub"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    with tempfile.TemporaryDirectory(prefix="dl4j-azure-key-") as temp:
        key_path = Path(temp) / "id_ed25519"
        result = subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "no SSH public key was found and ssh-keygen could not create an ephemeral key"
            )
        return key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()


def windows_worker_bootstrap_command() -> str:
    """Locate the one downloaded worker beneath the extension's sequence root."""
    return (
        "powershell -NoLogo -NonInteractive -ExecutionPolicy Bypass -Command "
        "\"$worker = Get-ChildItem -LiteralPath . -Filter 'worker.ps1' "
        "-Recurse -File | Select-Object -First 1; "
        "if ($null -eq $worker) { throw 'worker.ps1 was not downloaded' }; "
        "& $worker.FullName -Register\""
    )


def require_succeeded_provisioning_state(resource: Any, description: str) -> Any:
    """Turn terminal Azure resource failure states into controller failures."""
    state = getattr(resource, "provisioning_state", None)
    if state is None and isinstance(resource, dict):
        state = resource.get("provisioningState") or resource.get("provisioning_state")
    if state and str(state).lower() != "succeeded":
        raise RuntimeError(f"{description} provisioning failed: {state}")
    return resource


def _create_lane_vm_resources(
    context: dict[str, Any],
    group: str,
    location: str,
    run_id: str,
    item: dict[str, Any],
    identity: Any,
    subnet_id: str,
    worker_url: str,
    ssh_public_key: str,
    fence_check: Callable[[], None] | None = None,
    controller_epoch: str | None = None,
) -> dict[str, str]:
    lane_id = item.get("id") or item["shard"]["id"]
    lane_os = item.get("os") or item["shard"]["os"]
    image = item.get("image") or item["shard"]["image"]
    resource_run_id = (
        f"{run_id}-{controller_epoch}" if controller_epoch else run_id
    )
    vm_name = resource_name("dl4j", resource_run_id, lane_id, 64)
    nic_name = resource_name("dl4j-nic", resource_run_id, lane_id, 80)
    pip_name = resource_name("dl4j-pip", resource_run_id, lane_id, 80)
    disk_name = resource_name("dl4j-osdisk", resource_run_id, lane_id, 80)
    tags = {
        MANAGED_TAG: "true",
        RUN_TAG: run_id,
        SHARD_TAG: normalize_name(lane_id, 63),
        "dl4j-lane": normalize_name(lane_id, 63),
        "dl4j-provider": "azure",
    }
    if controller_epoch:
        tags[CONTROLLER_EPOCH_TAG] = controller_epoch
    public_ip = fenced_azure_operation(
        lambda: context["network"].public_ip_addresses.begin_create_or_update(
            group,
            pip_name,
            {
                "location": location,
                "public_ip_allocation_method": "Static",
                "sku": {"name": "Standard"},
                "tags": tags,
            },
        ),
        fence_check,
        timeout=600,
    )
    public_ip_id = azure_resource_id(
        context,
        group,
        "Microsoft.Network",
        "publicIPAddresses",
        pip_name,
        public_ip,
        "public IP",
    )
    nic = fenced_azure_operation(
        lambda: context["network"].network_interfaces.begin_create_or_update(
            group,
            nic_name,
            {
                "location": location,
                "enable_accelerated_networking": True,
                "ip_configurations": [{
                    "name": "primary",
                    "subnet": {"id": subnet_id},
                    "public_ip_address": {"id": public_ip_id},
                    "private_ip_allocation_method": "Dynamic",
                }],
                "tags": tags,
            },
        ),
        fence_check,
        timeout=600,
    )
    nic_id = azure_resource_id(
        context,
        group,
        "Microsoft.Network",
        "networkInterfaces",
        nic_name,
        nic,
        "network interface",
    )
    os_profile: dict[str, Any] = {
        "computer_name": (
            resource_name("dl4j", resource_run_id, lane_id, 15)
            if lane_os == "windows"
            else vm_name[:64]
        ),
        "admin_username": "dl4j",
    }
    if lane_os == "windows":
        os_profile["admin_password"] = secrets.token_urlsafe(24) + "aA1!"
    else:
        os_profile["custom_data"] = base64.b64encode(
            bootstrap_user_data(worker_url).encode("utf-8")
        ).decode("ascii")
        os_profile["linux_configuration"] = {
            "disable_password_authentication": True,
            "ssh": {
                "public_keys": [{
                    "path": "/home/dl4j/.ssh/authorized_keys",
                    "key_data": ssh_public_key,
                }]
            },
        }
    parameters: dict[str, Any] = {
        "location": location,
        "hardware_profile": {"vm_size": item["selectedMachine"]["name"]},
        "storage_profile": {
            "image_reference": {
                "publisher": image["publisher"],
                "offer": image["offer"],
                "sku": image["sku"],
                "version": image["version"],
            },
            "os_disk": {
                "name": disk_name,
                "create_option": "FromImage",
                "disk_size_gb": item["rootVolumeGiB"],
                "delete_option": "Delete",
                "managed_disk": {"storage_account_type": "Premium_LRS"},
            },
        },
        "os_profile": os_profile,
        "network_profile": {
            "network_interfaces": [{
                "id": nic_id,
                "primary": True,
                "delete_option": "Delete",
            }]
        },
        "diagnostics_profile": {"boot_diagnostics": {"enabled": True}},
        "identity": {
            "type": "UserAssigned",
            "user_assigned_identities": {identity.id: {}},
        },
        "tags": tags,
    }
    if item.get("zone"):
        parameters["zones"] = [item["zone"]]
    fenced_azure_operation(
        lambda: context["compute"].virtual_machines.begin_create_or_update(
            group, vm_name, parameters
        ),
        fence_check,
        timeout=1800,
    )
    fenced_azure_operation(
        lambda: context["compute"].disks.begin_update(
            group, disk_name, {"tags": tags}
        ),
        fence_check,
        timeout=600,
    )
    if lane_os == "windows":
        command = windows_worker_bootstrap_command()
        extension = fenced_azure_operation(
            lambda: context["compute"].virtual_machine_extensions.begin_create_or_update(
                group,
                vm_name,
                "dl4j-release-worker",
                {
                    "location": location,
                    "publisher": "Microsoft.Compute",
                    "type": "CustomScriptExtension",
                    "type_handler_version": "1.10",
                    "auto_upgrade_minor_version": True,
                    "protected_settings": {
                        "fileUris": [worker_url],
                        "commandToExecute": command,
                    },
                    "tags": tags,
                },
            ),
            fence_check,
            timeout=1800,
        )
        require_succeeded_provisioning_state(
            extension, "Windows worker extension"
        )
    return {
        "vm": vm_name,
        "nic": nic_name,
        "publicIp": pip_name,
        "disk": disk_name,
    }


def create_lane_vm(
    context: dict[str, Any],
    group: str,
    location: str,
    run_id: str,
    item: dict[str, Any],
    identity: Any,
    subnet_id: str,
    worker_url: str,
    ssh_public_key: str,
    fence_check: Callable[[], None] | None = None,
    cleanup_fence_check: Callable[[], None] | None = None,
    controller_epoch: str | None = None,
) -> dict[str, str]:
    lane_id = item.get("id") or item["shard"]["id"]
    resource_run_id = (
        f"{run_id}-{controller_epoch}" if controller_epoch else run_id
    )
    resources = {
        "vm": resource_name("dl4j", resource_run_id, lane_id, 64),
        "nic": resource_name("dl4j-nic", resource_run_id, lane_id, 80),
        "publicIp": resource_name("dl4j-pip", resource_run_id, lane_id, 80),
        "disk": resource_name("dl4j-osdisk", resource_run_id, lane_id, 80),
    }
    try:
        return _create_lane_vm_resources(
            context,
            group,
            location,
            run_id,
            item,
            identity,
            subnet_id,
            worker_url,
            ssh_public_key,
            fence_check,
            controller_epoch,
        )
    except Exception as exc:
        cleanup_errors = delete_lane_resources(
            context,
            group,
            resources,
            fence_check=cleanup_fence_check or fence_check,
        )
        if cleanup_errors:
            primary_failure = str(exc) or repr(exc)
            raise RuntimeError(
                f"Azure lane provisioning failed for {lane_id}: {primary_failure}; "
                f"partial-resource cleanup also failed: {'; '.join(cleanup_errors)}"
            ) from exc
        raise


def decode_log_payload(payload: bytes) -> str:
    """Decode UTF-8 logs and Windows PowerShell 5 UTF-16 redirection output."""
    if payload.startswith(b"\xff\xfe"):
        return payload.decode("utf-16-le", "replace").lstrip("\ufeff")
    if payload.startswith(b"\xfe\xff"):
        return payload.decode("utf-16-be", "replace").lstrip("\ufeff")
    if len(payload) >= 4 and len(payload) % 2 == 0:
        pairs = len(payload) // 2
        odd_nulls = payload[1::2].count(0)
        even_nulls = payload[0::2].count(0)
        if odd_nulls >= max(2, pairs // 4) and odd_nulls > even_nulls * 2:
            return payload.decode("utf-16-le", "replace")
        if even_nulls >= max(2, pairs // 4) and even_nulls > odd_nulls * 2:
            return payload.decode("utf-16-be", "replace")
    return payload.decode("utf-8", "replace")


def stream_blob_log(
    container: Any,
    name: str,
    offset: int,
    *,
    label: str | None = None,
) -> int:
    """Print new log bytes without racing an actively appended Azure Blob.

    Azure's SDK protects a multi-request download with the Blob ETag. A live
    append between those requests raises ResourceModifiedError/ConditionNotMet.
    Fixed-size range reads stay within one request, and any remaining ETag race
    is a transient observability event rather than a build failure.
    """
    conflict_attempts = 0
    while True:
        try:
            payload = container.download_blob(
                name,
                offset=offset,
                length=LOG_STREAM_CHUNK_BYTES,
                max_concurrency=1,
            ).readall()
        except Exception as exc:
            error_code = str(getattr(exc, "error_code", "") or "")
            transient_conflict = (
                exc.__class__.__name__ == "ResourceModifiedError"
                or error_code == "ConditionNotMet"
            )
            range_exhausted = getattr(exc, "status_code", None) == 416
            if transient_conflict:
                conflict_attempts += 1
                if conflict_attempts < LOG_STREAM_CONFLICT_RETRIES:
                    time.sleep(0.2 * conflict_attempts)
                    continue
                return offset
            if is_not_found(exc) or range_exhausted:
                return offset
            raise
        conflict_attempts = 0
        if not payload:
            return offset
        output = decode_log_payload(payload)
        if label:
            output = "".join(
                f"[{label}] {line}" for line in output.splitlines(keepends=True)
            )
        sys.stdout.write(output)
        sys.stdout.flush()
        offset += len(payload)
        if len(payload) < LOG_STREAM_CHUNK_BYTES:
            return offset


def print_retained_shard_log(
    container: Any,
    prefix: str,
    shard_id: str,
    name: str = "build.log",
) -> int:
    """Print a retained shard transcript before surfacing a controller failure."""
    offset = stream_blob_log(
        container,
        f"{prefix}/{shard_id}/{name}",
        0,
        label=f"{shard_id}/{name}",
    )
    if offset == 0:
        print(
            f"[{shard_id}/{name}] retained Azure transcript is not available",
            flush=True,
        )
    return offset


class LaneWaitError(RuntimeError):
    def __init__(self, message: str, completed_results: dict[str, dict[str, Any]]) -> None:
        super().__init__(message)
        self.completed_results = copy.deepcopy(completed_results)


def wait_for_lane(
    context: dict[str, Any],
    group: str,
    vm_name: str,
    artifact_container: Any,
    control_container: Any,
    plan: dict[str, Any],
    prefix: str,
    lane_id: str,
    shard_ids: list[str],
    timeout_hours: int,
    expected_statuses: dict[str, dict[str, Any]],
    controller_lease: ControllerLease | None = None,
    result_callback: Callable[[str, dict[str, Any]], None] | None = None,
    abort_event: threading.Event | None = None,
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_hours * 3600
    lane_log_offset = 0
    pending = set(shard_ids)
    results: dict[str, dict[str, Any]] = {}
    stopped_at: float | None = None
    while time.monotonic() < deadline:
        if abort_event is not None and abort_event.is_set():
            raise LaneWaitError(
                f"Azure lane {lane_id} was cancelled by its controller",
                results,
            )
        if controller_lease is not None:
            controller_lease.assert_healthy()
        lane_log_offset = stream_blob_log(
            artifact_container,
            f"{prefix}/lanes/{lane_id}/live.log",
            lane_log_offset,
        )
        failed_statuses: list[tuple[str, dict[str, Any]]] = []
        for shard_id in shard_ids:
            if shard_id not in pending:
                continue
            status = get_json(
                artifact_container, f"{prefix}/{shard_id}/status.json"
            )
            if status is None:
                continue
            expected = expected_statuses.get(shard_id)
            if expected is None:
                raise RuntimeError(
                    f"Azure lane {lane_id} has no status identity for shard {shard_id}"
                )
            if not shard_status_matches(status, expected):
                print(
                    f"[{utc_now()}] ignoring stale Azure status for shard "
                    f"{shard_id}: checkpoint identity does not match",
                    flush=True,
                )
                continue
            if int(status.get("exitCode", 1)) != 0:
                failed_statuses.append((shard_id, status))
                continue
            results[shard_id] = status
            pending.remove(shard_id)
            if result_callback is not None:
                result_callback(shard_id, status)
        if failed_statuses:
            shard_id, status = failed_statuses[0]
            print_retained_shard_log(artifact_container, prefix, shard_id)
            raise LaneWaitError(
                f"Azure lane {lane_id} shard {shard_id} failed: {status}",
                results,
            )
        if not pending:
            return results
        expected_epoch = (
            controller_epoch(controller_lease)
            if controller_lease is not None
            else None
        )
        switch_object = (
            controller_lease.name
            if controller_lease is not None
            else kill_switch_blob(plan)
        )
        if kill_switch_enabled(
            control_container,
            plan,
            expected_epoch,
            object_name=switch_object,
        ):
            raise RuntimeError("Azure release run kill switch was enabled")
        try:
            view = context["compute"].virtual_machines.instance_view(group, vm_name)
            codes = [
                str(object_value(item, "code", ""))
                for item in (object_value(view, "statuses", []) or [])
            ]
            stopped = any(
                code in {"PowerState/stopped", "PowerState/deallocated"}
                for code in codes
            )
            if stopped:
                stopped_at = stopped_at or time.monotonic()
                if time.monotonic() - stopped_at > 300:
                    raise RuntimeError(
                        f"Azure VM {vm_name} stopped without a retained status.json"
                    )
            else:
                stopped_at = None
        except Exception as exc:
            if is_not_found(exc):
                raise LaneWaitError(
                    f"Azure VM {vm_name} disappeared without retained status.json",
                    results,
                ) from exc
            raise
        print(
            f"[{utc_now()}] waiting for Azure lane {lane_id}; "
            f"pending shards {', '.join(sorted(pending))}",
            flush=True,
        )
        time.sleep(15)
    raise TimeoutError(f"Azure lane {lane_id} exceeded {timeout_hours} hours")


def delete_lane_resources(
    context: dict[str, Any],
    group: str,
    resources: dict[str, str],
    fence_check: Callable[[], None] | None = None,
) -> list[str]:
    errors: list[str] = []
    check = fence_check or (lambda: None)
    operations = [
        ("VM", lambda: context["compute"].virtual_machines.begin_delete(group, resources["vm"])),
        ("NIC", lambda: context["network"].network_interfaces.begin_delete(group, resources["nic"])),
        ("public IP", lambda: context["network"].public_ip_addresses.begin_delete(group, resources["publicIp"])),
    ]
    disk_name = resources.get("disk")
    if disk_name:
        operations.append(
            ("OS disk", lambda: context["compute"].disks.begin_delete(group, disk_name))
        )
    for label, create_operation in operations:
        for attempt in range(RESOURCE_CLEANUP_ATTEMPTS):
            try:
                fenced_azure_operation(create_operation, check, timeout=1800)
                break
            except Exception as exc:
                check()
                if is_not_found(exc):
                    break
                retryable_network_cleanup = (
                    label in {"NIC", "public IP"}
                    and is_transient_delete_error(exc)
                    and attempt < RESOURCE_CLEANUP_ATTEMPTS - 1
                )
                if retryable_network_cleanup:
                    time.sleep(RESOURCE_CLEANUP_RETRY_SECONDS)
                    check()
                    continue
                errors.append(f"{label} cleanup: {exc}")
                break
    return errors


def _run_parallel_lane(
    args: argparse.Namespace,
    controller_lease: ControllerLease,
    data: dict[str, Any],
    service: Any,
    account_key: str,
    artifact_container: Any,
    control_container: Any,
    identity: Any,
    subnet_id: str,
    ssh_key: str,
    prefix: str,
    lane: dict[str, Any],
    executions_by_id: dict[str, dict[str, Any]],
    events: queue.Queue[dict[str, Any]],
    abort_event: threading.Event | None = None,
) -> dict[str, dict[str, Any]]:
    """Run one persistent lane. Mutable run-manifest state stays in the caller."""
    context = data["context"]
    plan = data["plan"]
    lane_id = lane["id"]
    epoch = controller_epoch(controller_lease)
    resources: dict[str, str] | None = None
    cleanup_errors: list[str] = []
    primary_error: Exception | None = None
    results: dict[str, dict[str, Any]] = {}

    def lane_fence_check() -> None:
        if abort_event is not None and abort_event.is_set():
            raise RuntimeError(f"Azure lane {lane_id} was cancelled by its controller")
        controller_lease.check()

    def report_shard_result(shard_id: str, status: dict[str, Any]) -> None:
        events.put({
            "laneId": lane_id,
            "status": "shard-succeeded",
            "executionId": shard_id,
            "result": status,
        })

    events.put({"laneId": lane_id, "status": "provisioning"})
    try:
        lane_fence_check()
        shards = [
            with_shard_contract_digest(
                executions_by_id[execution_id]["shard"]
            )
            for execution_id in lane["executionIds"]
        ]
        config = {
            "provider": "azure",
            "subscription": context["subscription"],
            "location": data["location"],
            "resourceGroup": data["resourceGroup"],
            "storageAccount": data["storageAccount"],
            "bucket": (
                f"{data['storageAccount']}/{artifact_container_name(plan)}"
            ),
            "killSwitchBucket": (
                f"{data['storageAccount']}/{control_container_name(plan)}"
            ),
            "artifactPrefix": plan["artifactPrefix"],
            "runId": args.run_id,
            "releaseVersion": args.version,
            "snapshotVersion": args.snapshot_version,
            "commit": args.commit,
            "repository": args.repository,
            "killSwitchObject": kill_switch_blob(plan),
            "runKillSwitchObject": run_kill_switch_blob(plan, args.run_id),
            "managedIdentityClientId": identity.client_id,
            "compilerCache": compiler_cache_config(
                context,
                data["storageAccount"],
                account_key,
                plan,
                args.timeout_hours,
            ),
            "controllerEpoch": epoch,
            "laneId": lane_id,
            "shards": shards,
        }
        worker_path = ROOT / "release/azure" / lane["worker"]
        payload = render_worker(worker_path, config)
        bootstrap_blob = (
            f"{prefix}/lanes/{lane_id}/bootstrap/{epoch}/{worker_path.name}"
        )
        lane_fence_check()
        artifact_container.upload_blob(
            bootstrap_blob,
            payload,
            overwrite=True,
            content_settings=context["modules"]["ContentSettings"](
                content_type="text/plain"
            ),
        )
        lane_fence_check()
        worker_url = worker_sas_url(
            context,
            service,
            data["storageAccount"],
            account_key,
            artifact_container_name(plan),
            bootstrap_blob,
            args.timeout_hours,
        )
        lane_fence_check()
        resources = create_lane_vm(
            context,
            data["resourceGroup"],
            data["location"],
            args.run_id,
            lane,
            identity,
            subnet_id,
            worker_url,
            ssh_key,
            fence_check=lane_fence_check,
            cleanup_fence_check=controller_lease.check,
            controller_epoch=epoch,
        )
        events.put({
            "laneId": lane_id,
            "status": "running",
            "resources": resources,
        })
        results = wait_for_lane(
            context,
            data["resourceGroup"],
            resources["vm"],
            artifact_container,
            control_container,
            plan,
            prefix,
            lane_id,
            list(lane["executionIds"]),
            args.timeout_hours,
            {
                shard["id"]: shard_status_identity(config, shard)
                for shard in shards
            },
            controller_lease,
            result_callback=report_shard_result,
            abort_event=abort_event,
        )
    except Exception as exc:
        primary_error = exc
        results = copy.deepcopy(getattr(exc, "completed_results", results))
        if abort_event is not None:
            abort_event.set()
    finally:
        if resources:
            try:
                cleanup_errors = delete_lane_resources(
                    context,
                    data["resourceGroup"],
                    resources,
                    fence_check=controller_lease.check,
                )
            except Exception as exc:
                cleanup_errors = [f"lane cleanup: {exc}"]
        if cleanup_errors:
            cleanup_errors = list(cleanup_errors)
    if primary_error is not None:
        event = {
            "laneId": lane_id,
            "status": "failed",
            "failure": str(primary_error),
            "results": results,
        }
        if cleanup_errors:
            event["cleanupErrors"] = cleanup_errors
        events.put(event)
        if cleanup_errors:
            raise RuntimeError(
                f"{primary_error}; cleanup failed: {'; '.join(cleanup_errors)}"
            ) from primary_error
        raise primary_error
    if cleanup_errors:
        error = RuntimeError(
            f"Azure lane {lane_id} succeeded but resource cleanup failed: "
            + "; ".join(cleanup_errors)
        )
        events.put({
            "laneId": lane_id,
            "status": "failed",
            "failure": str(error),
            "cleanupErrors": cleanup_errors,
            "results": results,
        })
        raise error
    events.put({
        "laneId": lane_id,
        "status": "succeeded",
        "results": results,
    })
    return results


def _start_under_controller_lease(
    args: argparse.Namespace,
    controller_lease: ControllerLease,
    data: dict[str, Any],
    account: Any,
    service: Any,
    account_key: str,
) -> None:
    context = data["context"]
    plan = data["plan"]
    commit = args.commit
    run_id = args.run_id
    group = data["resourceGroup"]
    account_name = data["storageAccount"]
    artifact_container = service.get_container_client(artifact_container_name(plan))
    control_container = service.get_container_client(control_container_name(plan))
    prefix = f"{plan['artifactPrefix'].strip('/')}/{run_id}"
    run_blob = f"{prefix}/run.json"
    epoch = controller_epoch(controller_lease)
    prepare_emergency_kill_switch(
        control_container,
        plan,
        context["modules"],
        args.reset_kill_switch,
    )
    controller_lease.external_check = lambda: assert_emergency_kill_switch_disabled(
        control_container, plan
    )
    controller_lease.check()
    if get_json(artifact_container, run_blob) is not None:
        raise RuntimeError(f"Azure release run {run_id!r} already exists")
    run_manifest = {
        "schemaVersion": 1,
        "provider": "azure",
        "runId": run_id,
        "subscription": context["subscription"],
        "location": data["location"],
        "resourceGroup": group,
        "storageAccount": account_name,
        "container": artifact_container_name(plan),
        "releaseVersion": args.version,
        "snapshotVersion": args.snapshot_version,
        "commit": commit,
        "sourceBranch": args.branch,
        "repository": args.repository,
        "createdAt": utc_now(),
        "controllerEpoch": epoch,
        "status": "initializing",
        "managedIdentity": None,
        "compilerCache": compiler_cache_metadata(plan, account_name),
        "parallel": True,
        "lanes": [],
        "executions": [],
        "unsupportedWorkflows": plan.get("unsupportedWorkflows", {}),
    }
    for lane in data["lanes"]:
        run_manifest["lanes"].append({
            "id": lane["id"],
            "os": lane["os"],
            "architecture": lane["architecture"],
            "image": lane["image"],
            "executionIds": lane["executionIds"],
            "selectedMachine": lane["selectedMachine"],
            "rootVolumeGiB": lane["rootVolumeGiB"],
            "zone": lane.get("zone"),
            "status": "pending",
        })
    for item in data["executions"]:
        run_manifest["executions"].append({
            "id": item["id"],
            "laneId": item["laneId"],
            "shard": item["shard"],
            "selectedMachine": item["selectedMachine"],
            "rootVolumeGiB": item["rootVolumeGiB"],
            "zone": item.get("zone"),
            "status": "pending",
        })
    controller_lease.check()
    put_json(
        artifact_container,
        run_blob,
        run_manifest,
        context["modules"],
        controller_lease=controller_lease,
        create_only=True,
    )
    controller_lease.check()
    run_switch = run_kill_switch_blob(plan, run_id)
    # The create-only run manifest above proves this is a new run. Initialize its
    # leased switch automatically; --reset-kill-switch is reserved for clearing a
    # prior forced global emergency stop.
    set_kill_switch(
        control_container,
        plan,
        False,
        context["modules"],
        "start",
        controller_lease=controller_lease,
        object_name=run_switch,
    )
    identity, identity_metadata = ensure_identity(
        context,
        group,
        data["location"],
        run_id,
        account.id,
        fence_check=controller_lease.check,
        controller_epoch=epoch,
    )
    subnet_id, _ = ensure_network(
        context,
        group,
        data["location"],
        fence_check=controller_lease.check,
    )
    ssh_key = resolve_ssh_public_key(args.ssh_public_key)
    run_manifest["managedIdentity"] = identity_metadata
    run_manifest["status"] = "running"
    controller_lease.check()
    put_json(
        artifact_container,
        run_blob,
        run_manifest,
        context["modules"],
        controller_lease=controller_lease,
    )
    controller_lease.check()
    print(json.dumps({
        "runId": run_id,
        "statusCommand": (
            f"python3 release/azure/release.py --subscription {context['subscription']} "
            f"--location {data['location']} status --run-id {run_id}"
        ),
        "logsCommand": (
            f"python3 release/azure/release.py --subscription {context['subscription']} "
            f"--location {data['location']} logs --run-id {run_id} --follow"
        ),
        "stopCommand": (
            f"python3 release/azure/release.py --subscription {context['subscription']} "
            f"--location {data['location']} stop-everything --wait"
        ),
    }, indent=2), flush=True)
    try:
        lane_records = {
            lane["id"]: lane for lane in run_manifest["lanes"]
        }
        execution_records = {
            execution["id"]: execution
            for execution in run_manifest["executions"]
        }
        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        failures: dict[str, str] = {}
        cancellation_enabled = False

        def record_failure(lane_id: str, message: str) -> None:
            message = str(message)
            previous = failures.get(lane_id)
            if not previous:
                failures[lane_id] = message
            elif message in previous:
                return
            elif previous in message:
                failures[lane_id] = message
            else:
                failures[lane_id] = f"{previous}; {message}"

        def apply_event(event: dict[str, Any]) -> None:
            nonlocal cancellation_enabled
            lane_id = event["laneId"]
            lane = lane_records[lane_id]
            status_value = event["status"]
            if "resources" in event:
                lane["resources"] = event["resources"]
            if "cleanupErrors" in event:
                lane["cleanupErrors"] = event["cleanupErrors"]
            if "failure" in event:
                lane["failure"] = event["failure"]
            for execution_id, result in event.get("results", {}).items():
                execution = execution_records[execution_id]
                execution["status"] = "succeeded"
                execution["result"] = result
            if status_value == "shard-succeeded":
                execution_id = event["executionId"]
                execution = execution_records[execution_id]
                execution["status"] = "succeeded"
                execution["result"] = event["result"]
                completed = lane.setdefault("completedExecutionIds", [])
                if execution_id not in completed:
                    completed.append(execution_id)
            elif status_value == "cleanup-failed":
                lane["status"] = "failed"
            else:
                lane["status"] = status_value
            for execution_id in lane["executionIds"]:
                execution = execution_records[execution_id]
                if status_value in {"provisioning", "running"}:
                    if execution["status"] in {
                        "pending", "provisioning", "running"
                    }:
                        execution["status"] = status_value
                    if "resources" in event:
                        execution["resources"] = event["resources"]
                elif status_value == "succeeded":
                    execution["status"] = "succeeded"
                    execution["result"] = event["results"][execution_id]
                elif status_value in {"failed", "cleanup-failed"}:
                    if execution.get("status") != "succeeded":
                        execution["status"] = "failed"
                        execution["failure"] = event.get(
                            "failure", lane.get("failure", "lane cleanup failed")
                        )
            if status_value in {"failed", "cleanup-failed"}:
                abort_event.set()
                failure_message = event.get(
                    "failure", lane.get("failure", "lane failed")
                )
                if event.get("cleanupErrors"):
                    failure_message += "; resource cleanup failed: " + "; ".join(
                        event["cleanupErrors"]
                    )
                record_failure(lane_id, failure_message)
                if not cancellation_enabled:
                    controller_lease.check()
                    set_kill_switch(
                        control_container,
                        plan,
                        True,
                        context["modules"],
                        f"parallel-lane-failure:{lane_id}",
                        controller_lease=controller_lease,
                        object_name=run_kill_switch_blob(plan, run_id),
                    )
                    cancellation_enabled = True
            controller_lease.check()
            put_json(
                artifact_container,
                run_blob,
                run_manifest,
                context["modules"],
                controller_lease=controller_lease,
            )
            controller_lease.check()

        abort_event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(data["lanes"]),
            thread_name_prefix="azure-release-lane",
        )
        future_lanes: dict[concurrent.futures.Future[Any], str] = {}
        try:
            future_lanes = {
                executor.submit(
                    _run_parallel_lane,
                    args,
                    controller_lease,
                    data,
                    service,
                    account_key,
                    artifact_container,
                    control_container,
                    identity,
                    subnet_id,
                    ssh_key,
                    prefix,
                    lane,
                    execution_records,
                    event_queue,
                    abort_event,
                ): lane["id"]
                for lane in data["lanes"]
            }
            pending = set(future_lanes)
            while pending:
                try:
                    apply_event(event_queue.get(timeout=1))
                except queue.Empty:
                    pass
                while True:
                    try:
                        apply_event(event_queue.get_nowait())
                    except queue.Empty:
                        break
                for future in list(pending):
                    if not future.done():
                        continue
                    pending.remove(future)
                    lane_id = future_lanes[future]
                    try:
                        future.result()
                    except Exception as exc:
                        record_failure(lane_id, str(exc))
            while True:
                try:
                    apply_event(event_queue.get_nowait())
                except queue.Empty:
                    break
        except BaseException:
            abort_event.set()
            # A local event makes every lane leave its polling/provisioning loop
            # even when Blob cancellation itself is unavailable. Only mutate the
            # shared switch while this controller can still prove ownership.
            try:
                controller_lease.check()
            except Exception:
                pass
            else:
                try:
                    set_kill_switch(
                        control_container,
                        plan,
                        True,
                        context["modules"],
                        "controller-abort",
                        controller_lease=controller_lease,
                        object_name=run_kill_switch_blob(plan, run_id),
                    )
                    cancellation_enabled = True
                except Exception:
                    pass
            for future in future_lanes:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if failures:
            raise RuntimeError(
                "parallel Azure lane failure: "
                + "; ".join(
                    f"{lane_id}: {message}"
                    for lane_id, message in sorted(failures.items())
                )
            )
        run_manifest["status"] = "succeeded"
        run_manifest["completedAt"] = utc_now()
    except Exception as exc:
        run_manifest["status"] = "failed"
        run_manifest["failure"] = str(exc)
        run_manifest["completedAt"] = utc_now()
        controller_lease.check()
        put_json(
            artifact_container,
            run_blob,
            run_manifest,
            context["modules"],
            controller_lease=controller_lease,
        )
        controller_lease.check()
        raise
    controller_lease.check()
    put_json(
        artifact_container,
        run_blob,
        run_manifest,
        context["modules"],
        controller_lease=controller_lease,
    )
    controller_lease.check()
    print(json.dumps(run_manifest, indent=2))


def start(args: argparse.Namespace) -> None:
    data = preflight_data(args, include_context=True)
    context = data["context"]
    plan = data["plan"]
    commit = args.commit.lower() if args.commit else resolve_commit(args.repository, args.branch)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full 40-character Git commit")
    run_id = args.run_id or run_id_for(args.version, commit)
    args.commit = commit
    args.run_id = run_id
    group = data["resourceGroup"]
    account_name = data["storageAccount"]
    # Blob locking cannot precede the lock account. These two operations are the
    # only pre-lease mutations and are idempotent bootstrap infrastructure; all
    # run state, kill-switch changes, identity/network/VM work follows the lease.
    ensure_resource_group(context, group, data["location"])
    account, service, account_key = ensure_storage(
        context, group, data["location"], account_name, plan
    )
    artifact_container = service.get_container_client(artifact_container_name(plan))
    control_container = service.get_container_client(control_container_name(plan))
    lease = ControllerLease(
        control_container, controller_lock_blob(plan, run_id)
    ).acquire()
    completed = False
    primary_error: BaseException | None = None
    cleanup_errors: list[str] = []
    identity_errors: list[str] = []
    try:
        _start_under_controller_lease(
            args, lease, data, account, service, account_key
        )
        completed = True
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        owns_fence = True
        try:
            lease.check()
        except Exception as exc:
            owns_fence = False
            cleanup_errors.append(f"controller lease before cleanup: {exc}")
        try:
            if owns_fence:
                try:
                    set_kill_switch(
                        control_container,
                        plan,
                        True,
                        context["modules"],
                        "run-complete" if completed else "controller-failure",
                        controller_lease=lease,
                        object_name=run_kill_switch_blob(plan, run_id),
                    )
                    lease.check()
                except Exception as exc:
                    cleanup_errors.append(f"restore kill switch: {exc}")
                    try:
                        lease.check()
                    except Exception:
                        owns_fence = False

            if owns_fence:
                try:
                    remaining_resources, resource_errors = (
                        reconcile_managed_run_resources(
                            context,
                            group,
                            run_id,
                            fence_check=lease.check,
                        )
                    )
                    cleanup_errors.extend(resource_errors)
                    remaining_names = [
                        f"{kind}={name}"
                        for kind, names in remaining_resources.items()
                        for name in names
                    ]
                    if resource_errors or remaining_names:
                        detail = ", ".join(remaining_names) or (
                            "resource cleanup could not be verified"
                        )
                        identity_errors.append(
                            "managed identity retained because run resources remain: "
                            + detail
                        )
                    else:
                        _, identity_errors = cleanup_managed_identities(
                            context,
                            group,
                            str(account.id),
                            run_id,
                            fence_check=lease.check,
                        )
                    cleanup_errors.extend(identity_errors)
                    lease.check()
                except Exception as exc:
                    cleanup_errors.append(f"managed resource cleanup: {exc}")
                    identity_error = (
                        "managed identity retained because run resource cleanup failed: "
                        f"{exc}"
                    )
                    identity_errors.append(identity_error)
                    cleanup_errors.append(identity_error)
                    try:
                        lease.check()
                    except Exception:
                        owns_fence = False

            if owns_fence:
                try:
                    lease.check()
                    run = load_run(artifact_container, plan, run_id)
                    run["identityCleanupStatus"] = (
                        "failed" if identity_errors else "succeeded"
                    )
                    if primary_error is not None:
                        run["status"] = "failed"
                        run["failure"] = str(primary_error)
                        run["completedAt"] = utc_now()
                    if cleanup_errors:
                        run["controllerCleanupErrors"] = cleanup_errors
                        if completed:
                            run["status"] = "failed"
                            run["failure"] = (
                                "controller cleanup failed: " + "; ".join(cleanup_errors)
                            )
                            run["completedAt"] = utc_now()
                    lease.check()
                    put_json(
                        artifact_container,
                        f"{plan['artifactPrefix'].strip('/')}/{run_id}/run.json",
                        run,
                        context["modules"],
                        controller_lease=lease,
                    )
                    lease.check()
                except Exception as exc:
                    if not is_not_found(exc):
                        cleanup_errors.append(f"run manifest cleanup status: {exc}")
        finally:
            release_errors = lease.release()
        if release_errors:
            cleanup_errors.extend(release_errors)
        if cleanup_errors:
            message = "Azure controller cleanup reported: " + "; ".join(cleanup_errors)
            if primary_error is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr)


def existing_storage(
    args: argparse.Namespace,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], str, str, Any, Any]:
    context = cloud_context(
        args.subscription, allow_wizard=not getattr(args, "no_wizard", False)
    )
    location = resolve_location(
        args.location, allow_wizard=not getattr(args, "no_wizard", False)
    )
    group = resource_group_name(location, getattr(args, "resource_group", None))
    account_name = storage_account_name(
        context["subscription"], location, getattr(args, "storage_account", None)
    )
    account = context["storage"].storage_accounts.get_properties(group, account_name)
    keys = context["storage"].storage_accounts.list_keys(group, account_name)
    values = list(object_value(keys, "keys", []) or [])
    if not values:
        raise RuntimeError(f"Azure storage account {account_name} returned no keys")
    service = context["modules"]["BlobServiceClient"](
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=str(object_value(values[0], "value")),
    )
    return context, location, group, account, service


def latest_run_id(container: Any, plan: dict[str, Any]) -> str:
    prefix = f"{plan['artifactPrefix'].strip('/')}/"
    candidates = [
        item.name[len(prefix):].split("/", 1)[0]
        for item in container.list_blobs(name_starts_with=prefix)
        if item.name.endswith("/run.json")
    ]
    if not candidates:
        raise RuntimeError("no Azure release runs were found")
    manifests: list[tuple[str, str]] = []
    for run_id in sorted(set(candidates)):
        value = get_json(container, f"{prefix}{run_id}/run.json")
        if value is not None:
            manifests.append((str(value.get("createdAt", "")), run_id))
    if not manifests:
        raise RuntimeError("no readable Azure release run manifests were found")
    return max(manifests)[1]


def load_run(container: Any, plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    value = get_json(
        container,
        f"{plan['artifactPrefix'].strip('/')}/{run_id}/run.json",
    )
    if value is None:
        raise RuntimeError(f"Azure release run {run_id!r} was not found")
    return value


def _display_time(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def vm_status_details(context: dict[str, Any], group: str, vm: Any) -> dict[str, Any]:
    """Return CloudWatch-style VM state and bounded boot-console diagnostics."""
    virtual_machines = context["compute"].virtual_machines
    hardware = object_value(vm, "hardware_profile", {}) or {}
    details: dict[str, Any] = {
        "name": vm.name,
        "location": vm.location,
        "tags": object_value(vm, "tags", {}) or {},
        "size": object_value(hardware, "vm_size"),
        "provisioningState": object_value(vm, "provisioning_state"),
    }
    try:
        view = virtual_machines.instance_view(group, vm.name)
        statuses = []
        for item in object_value(view, "statuses", []) or []:
            status = {
                "code": str(object_value(item, "code", "")),
                "displayStatus": object_value(item, "display_status"),
                "level": object_value(item, "level"),
                "message": object_value(item, "message"),
                "time": _display_time(object_value(item, "time")),
            }
            statuses.append({key: value for key, value in status.items() if value is not None})
        details["statuses"] = statuses
        details["powerState"] = next(
            (
                item["code"].split("/", 1)[1]
                for item in statuses
                if item["code"].startswith("PowerState/")
            ),
            None,
        )
    except Exception as exc:
        details["instanceViewError"] = str(exc)

    retrieve = getattr(virtual_machines, "retrieve_boot_diagnostics_data", None)
    if retrieve is not None:
        try:
            boot = retrieve(group, vm.name)
            serial_uri = object_value(boot, "serial_console_log_blob_uri")
            details["bootDiagnostics"] = {
                "serialConsoleLogAvailable": bool(serial_uri),
                "consoleScreenshotAvailable": bool(
                    object_value(boot, "console_screenshot_blob_uri")
                ),
            }
            if serial_uri:
                request = urllib.request.Request(
                    str(serial_uri), headers={"Range": "bytes=-12000"}
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    console = response.read(12000).decode("utf-8", "replace")
                details["consoleOutputTail"] = console.splitlines()
        except Exception as exc:
            details["bootDiagnosticsError"] = str(exc)
    return details


def status(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context, location, group, account, service = existing_storage(args, plan)
    container = service.get_container_client(artifact_container_name(plan))
    run_id = args.run_id or latest_run_id(container, plan)
    run = load_run(container, plan, run_id)
    machines: list[dict[str, Any]] = []
    for vm in context["compute"].virtual_machines.list(group):
        tags = object_value(vm, "tags", {}) or {}
        if tags.get(MANAGED_TAG) == "true" and tags.get(RUN_TAG) == run_id:
            machines.append(vm_status_details(context, group, vm))
    print(json.dumps({
        "subscription": context["subscription"],
        "location": location,
        "resourceGroup": group,
        "storageAccount": account.name,
        "run": run,
        "activeMachines": machines,
    }, indent=2))


def show_logs(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    _, _, _, _, service = existing_storage(args, plan)
    container = service.get_container_client(artifact_container_name(plan))
    run = load_run(container, plan, args.run_id)
    selected = set(args.shard or [])
    shards = [
        item["shard"]["id"] for item in run.get("executions", [])
        if not selected or item["id"] in selected or item["shard"]["id"] in selected
    ]
    if selected and not shards:
        raise RuntimeError(
            "none of the requested Azure shard selectors are present in run "
            f"{args.run_id!r}: {', '.join(sorted(selected))}"
        )
    lanes = [str(item["id"]) for item in run.get("lanes", [])]
    streams = (
        [("shard", shard) for shard in shards]
        if selected or not lanes
        else [("lane", lane) for lane in lanes]
    )
    offsets = {stream: 0 for stream in streams}
    prefix = f"{plan['artifactPrefix'].strip('/')}/{args.run_id}"
    while True:
        executions = {
            item["shard"]["id"]: item for item in run.get("executions", [])
        }
        run_terminal = run.get("status") in {"succeeded", "failed"}
        for stream in streams:
            kind, stream_id = stream
            if kind == "lane":
                name = "live.log"
                blob = f"{prefix}/lanes/{stream_id}/{name}"
            else:
                execution_terminal = executions.get(stream_id, {}).get("status") in {
                    "succeeded",
                    "failed",
                }
                name = "build.log" if run_terminal or execution_terminal else "live.log"
                blob = f"{prefix}/{stream_id}/{name}"
            offsets[stream] = stream_blob_log(
                container,
                blob,
                offsets[stream],
                label=f"{kind}/{stream_id}",
            )
        if not args.follow or run_terminal:
            return
        time.sleep(5)
        run = load_run(container, plan, args.run_id)


def _delete_logs_under_controller_lease(
    args: argparse.Namespace,
    plan: dict[str, Any],
    context: dict[str, Any],
    group: str,
    service: Any,
    controller_lease: ControllerLease,
) -> list[str]:
    container = service.get_container_client(artifact_container_name(plan))
    prefix = f"{plan['artifactPrefix'].strip('/')}/"
    controller_lease.check()
    if args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = sorted({
            item.name[len(prefix):].split("/", 1)[0]
            for item in container.list_blobs(name_starts_with=prefix)
            if item.name.endswith("/run.json")
        })
    controller_lease.check()

    target_runs = set(run_ids)
    for run_id in run_ids:
        controller_lease.check()
        run = load_run(container, plan, run_id)
        if run.get("status") not in {"succeeded", "failed"}:
            raise RuntimeError(
                f"refusing to delete logs for non-terminal Azure run {run_id!r}"
            )
    controller_lease.check()
    try:
        virtual_machines = context["compute"].virtual_machines.list(group)
        for vm in virtual_machines:
            controller_lease.check()
            tags = object_value(vm, "tags", {}) or {}
            if (
                tags.get(MANAGED_TAG) == "true"
                and tags.get(RUN_TAG) in target_runs
            ):
                raise RuntimeError(
                    "refusing to delete logs while a targeted run still has an Azure VM: "
                    f"{vm.name}"
                )
    except Exception as exc:
        if not is_not_found(exc):
            raise
    controller_lease.check()

    candidates: list[str] = []
    for run_id in run_ids:
        run_prefix = f"{prefix}{run_id}/"
        controller_lease.check()
        for item in container.list_blobs(name_starts_with=run_prefix):
            controller_lease.check()
            if item.name.endswith(("/build.log", "/live.log")) or "/events/" in item.name:
                candidates.append(item.name)
        controller_lease.check()

    removed: list[str] = []
    for name in candidates:
        controller_lease.check()
        try:
            container.delete_blob(name, delete_snapshots="include")
        except Exception:
            controller_lease.check()
            raise
        controller_lease.check()
        removed.append(name)
    return removed


def delete_logs(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("refusing to delete Azure logs without --yes")
    plan = load_plan(args.plan)
    context, _, group, _, service = existing_storage(args, plan)
    control_container = service.get_container_client(control_container_name(plan))
    lease = ControllerLease(
        control_container, controller_lock_blob(plan)
    ).acquire()
    primary_error: BaseException | None = None
    removed: list[str] = []
    try:
        removed = _delete_logs_under_controller_lease(
            args, plan, context, group, service, lease
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        release_errors = lease.release()
        if release_errors:
            message = "Azure log-deletion lease cleanup failed: " + "; ".join(
                release_errors
            )
            if primary_error is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr)
    print(json.dumps({"deletedAzureBlobLogs": removed}, indent=2))


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_attested_file(
    shard_manifest: dict[str, Any], file_name: str, path: Path
) -> None:
    entries = [
        item for item in shard_manifest.get("files", [])
        if isinstance(item, dict) and item.get("path") == file_name
    ]
    if len(entries) != 1:
        raise RuntimeError(
            f"shard manifest must attest exactly one {file_name!r} entry"
        )
    entry = entries[0]
    if int(entry.get("size", -1)) != path.stat().st_size:
        raise RuntimeError(f"shard manifest size mismatch for {file_name!r}")
    if entry.get("sha256") != file_digest(path):
        raise RuntimeError(f"shard manifest SHA-256 mismatch for {file_name!r}")


def github_release_exists(release_tag: str, github_repository: str) -> bool:
    return subprocess.run(
        ["gh", "release", "view", release_tag, "--repo", github_repository],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def github_release_asset_names(
    release_tag: str, github_repository: str
) -> set[str]:
    result = subprocess.run(
        [
            "gh", "release", "view", release_tag,
            "--repo", github_repository,
            "--json", "assets",
            "--jq", ".assets[].name",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def download_github_release_manifest(
    release_tag: str,
    github_repository: str,
    directory: Path,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "gh", "release", "download", release_tag,
            "--repo", github_repository,
            "--pattern", "release-build-manifest.json",
            "--dir", str(directory), "--clobber",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    path = directory / "release-build-manifest.json"
    if result.returncode != 0 or not path.is_file():
        if required:
            raise RuntimeError(
                "existing GitHub release is missing a downloadable release-build-manifest.json"
            )
        return None, None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing GitHub release manifest is unreadable or invalid") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("existing GitHub release manifest must be a JSON object")
    return manifest, file_digest(path)


def assert_github_manifest_unchanged(
    release_tag: str,
    github_repository: str,
    expected_release_exists: bool,
    expected_manifest_digest: str | None,
    directory: Path,
) -> None:
    current_exists = github_release_exists(release_tag, github_repository)
    if current_exists != expected_release_exists:
        raise RuntimeError(
            "GitHub release changed while collecting; rerun collect against the new state"
        )
    if not current_exists:
        return
    _, current_digest = download_github_release_manifest(
        release_tag,
        github_repository,
        directory,
        required=True,
    )
    if current_digest != expected_manifest_digest:
        raise RuntimeError(
            "GitHub release manifest changed concurrently; rerun collect to merge it"
        )


def download_existing_release_assets(
    manifest: dict[str, Any],
    directory: Path,
    release_tag: str,
    github_repository: str,
) -> dict[str, Path]:
    verified_assets = merge_release_assets(manifest.get("assets", []), [])
    outputs: dict[str, Path] = {}
    for name, asset in sorted(verified_assets.items()):
        if Path(name).name != name or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", name
        ):
            raise RuntimeError(f"unsafe existing release asset name: {name!r}")
        subprocess.run(
            [
                "gh", "release", "download", release_tag,
                "--repo", github_repository,
                "--pattern", name,
                "--dir", str(directory),
                "--clobber",
            ],
            check=True,
        )
        output = directory / name
        if not output.is_file():
            raise RuntimeError(f"existing release asset {name!r} was not downloaded")
        if output.stat().st_size != int(asset.get("size", -1)):
            raise RuntimeError(f"existing release asset {name!r} has a size mismatch")
        if file_digest(output) != asset.get("sha256"):
            raise RuntimeError(f"existing release asset {name!r} has a SHA-256 mismatch")
        outputs[name] = output
    return outputs


def download_existing_maven_archives(
    manifest: dict[str, Any],
    directory: Path,
    release_tag: str,
    github_repository: str,
) -> dict[str, Path]:
    outputs = download_existing_release_assets(
        manifest, directory, release_tag, github_repository
    )
    return {
        name: path for name, path in outputs.items()
        if name.startswith("maven-repository-") and name.endswith(".tar.gz")
    }


def attested_shard_variants(
    planned: dict[str, Any], shard_manifest: dict[str, Any]
) -> set[str]:
    planned_variants = {
        item["name"]: item for item in planned["build"]["variants"]
    }
    declared: set[str] | None = None
    if "variants" in shard_manifest:
        raw = shard_manifest["variants"]
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise RuntimeError("shard manifest variants must be a string list")
        declared = set(raw)
        if len(declared) != len(raw):
            raise RuntimeError("shard manifest variants contain duplicates")
    file_names = {
        Path(str(item.get("path", ""))).name
        for item in shard_manifest.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    platform = planned["build"]["javacppPlatform"]
    inferred = set()
    for name, variant in planned_variants.items():
        suffix = variant.get("classifierSuffix", variant.get("suffix", ""))
        classifier = f"{platform}{suffix}"
        if any(file_name.endswith(f"-{classifier}.jar") for file_name in file_names):
            inferred.add(name)
    if not inferred:
        raise RuntimeError("shard manifest has no exact classifier JARs to attest variants")
    if declared is not None and declared != inferred:
        missing = sorted(declared - inferred)
        undeclared = sorted(inferred - declared)
        details = []
        if missing:
            details.append(f"declared without classifier JAR: {', '.join(missing)}")
        if undeclared:
            details.append(f"classifier JAR without declaration: {', '.join(undeclared)}")
        raise RuntimeError(
            "shard manifest variants do not match classifier files"
            + (f" ({'; '.join(details)})" if details else "")
        )
    return inferred


def validate_existing_release_shards(
    manifest: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    plan: dict[str, Any],
    release_version: str,
    commit: str,
) -> set[str]:
    raw_shards = manifest.get("shards", [])
    if not isinstance(raw_shards, list) or any(not isinstance(item, str) for item in raw_shards):
        raise RuntimeError("existing release manifest shards must be a string list")
    shards = set(raw_shards)
    if len(shards) != len(raw_shards):
        raise RuntimeError("existing release manifest contains duplicate shards")
    if set(paths) != set(assets):
        raise RuntimeError("existing release assets were not all downloaded and verified")
    for name, asset in assets.items():
        if asset.get("shard") not in shards:
            raise RuntimeError(
                f"existing release asset {name!r} references an unclaimed shard"
            )
    by_id = {item["id"]: item for item in plan["shards"]}
    expected_workloads: set[str] = set()
    for shard_id in sorted(shards):
        parent, variant = _selector_parts(shard_id)
        planned = by_id.get(parent)
        if planned is None or (
            variant is not None
            and variant not in {item["name"] for item in planned["build"]["variants"]}
        ):
            raise RuntimeError(f"existing release manifest has unknown shard {shard_id!r}")
        expected_workloads.update(planned.get("workloads", []))
        shard_asset_name = f"{shard_id}-shard-manifest.json"
        shard_asset = assets.get(shard_asset_name)
        shard_path = paths.get(shard_asset_name)
        if shard_asset is None or shard_path is None or shard_asset.get("shard") != shard_id:
            raise RuntimeError(
                f"existing release shard {shard_id!r} lacks its verified shard manifest asset"
            )
        try:
            shard_manifest = json.loads(shard_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"existing shard manifest {shard_asset_name!r} is invalid"
            ) from exc
        identity = (
            shard_manifest.get("shard"),
            shard_manifest.get("commit"),
            shard_manifest.get("releaseVersion"),
        )
        if identity != (shard_id, commit, release_version):
            raise RuntimeError(
                f"existing shard manifest {shard_asset_name!r} has a different identity"
            )
        if manifest.get("runId") and shard_manifest.get("runId") != manifest.get("runId"):
            raise RuntimeError(
                f"existing shard manifest {shard_asset_name!r} has a different runId"
            )
        if set(shard_manifest.get("workloads", [])) != set(planned.get("workloads", [])):
            raise RuntimeError(
                f"existing shard manifest {shard_asset_name!r} has unexpected workloads"
            )
        expected_platform_identity = (
            planned["os"],
            planned["build"]["javacppPlatform"],
            planned["build"]["backend"],
        )
        actual_platform_identity = (
            shard_manifest.get("os"),
            shard_manifest.get("platform"),
            shard_manifest.get("backend"),
        )
        if actual_platform_identity != expected_platform_identity:
            raise RuntimeError(
                f"existing shard manifest {shard_asset_name!r} has unexpected platform identity"
            )
        planned_variants = {
            item["name"] for item in planned["build"]["variants"]
        }
        actual_variants = attested_shard_variants(planned, shard_manifest)
        invalid_variants = (
            not actual_variants
            or not actual_variants <= planned_variants
            or (variant is not None and actual_variants != {variant})
        )
        if invalid_variants:
            raise RuntimeError(
                f"existing shard manifest {shard_asset_name!r} has unexpected variants"
            )
        for workload in planned.get("workloads", []):
            prefix = "maven-repository" if workload == "maven" else "sdk-assets"
            archive_name = f"{prefix}-{shard_id}.tar.gz"
            archive_asset = assets.get(archive_name)
            if (
                archive_asset is None
                or archive_name not in paths
                or archive_asset.get("shard") != shard_id
            ):
                raise RuntimeError(
                    f"existing release shard {shard_id!r} lacks verified {workload} output"
                )
    if set(manifest.get("workloads", [])) != expected_workloads:
        raise RuntimeError("existing release manifest workloads do not match verified shards")
    return shards


def verified_release_matrix_coverage(
    manifest: dict[str, Any],
    shards: set[str],
    paths: dict[str, Path],
    plan: dict[str, Any],
) -> set[str]:
    by_id = {item["id"]: item for item in plan["shards"]}
    covered: set[str] = set()
    for shard_id in sorted(shards):
        parent, selected_variant = _selector_parts(shard_id)
        shard_manifest = json.loads(
            paths[f"{shard_id}-shard-manifest.json"].read_text(encoding="utf-8")
        )
        variants = attested_shard_variants(by_id[parent], shard_manifest)
        if selected_variant is not None and variants != {selected_variant}:
            raise RuntimeError(
                f"existing shard {shard_id!r} attests unexpected variants"
            )
        covered.update(f"{parent}--{variant}" for variant in variants)
    declared = manifest.get("matrixEntries")
    if declared is not None:
        if (
            not isinstance(declared, list)
            or any(not isinstance(item, str) for item in declared)
            or len(set(declared)) != len(declared)
            or set(declared) != covered
        ):
            raise RuntimeError(
                "existing release manifest matrixEntries do not match verified shard variants"
            )
    return covered


def _collect_under_controller_lease(
    args: argparse.Namespace,
    plan: dict[str, Any],
    context: dict[str, Any],
    location: str,
    group: str,
    account: Any,
    service: Any,
    collector_lease: ControllerLease,
) -> None:
    container = service.get_container_client(artifact_container_name(plan))
    collector_lease.check()
    run = load_run(container, plan, args.run_id)
    if run.get("commit") != args.commit or run.get("releaseVersion") != args.version:
        raise RuntimeError("collect identity does not match run.json")
    executions = run.get("executions", [])
    if args.shard:
        selected = set(args.shard)
        executions = [
            item for item in executions
            if item["id"] in selected or item["shard"]["id"] in selected
        ]
    if not executions:
        raise RuntimeError("no selected execution outputs")
    prefix = f"{plan['artifactPrefix'].strip('/')}/{args.run_id}"
    with tempfile.TemporaryDirectory(prefix="dl4j-azure-collect-") as temp:
        directory = Path(temp)
        aws_plan = json.loads(
            (ROOT / "release/aws/release-plan.json").read_text(encoding="utf-8")
        )
        existing_manifest = None
        existing_manifest_digest = None
        existing_assets: dict[str, dict[str, Any]] = {}
        existing_asset_paths: dict[str, Path] = {}
        verified_existing_shards: set[str] = set()
        verified_existing_matrix: set[str] = set()
        maven_archives: dict[str, Path] = {}
        release_exists = False
        if not args.no_github:
            release_exists = github_release_exists(
                args.release_tag, args.github_repository
            )
            if release_exists:
                existing_directory = directory / "existing-release"
                existing_manifest, existing_manifest_digest = (
                    download_github_release_manifest(
                        args.release_tag,
                        args.github_repository,
                        existing_directory,
                        required=True,
                    )
                )
                if (
                    existing_manifest.get("releaseVersion") != args.version
                    or existing_manifest.get("commit") != args.commit
                    or existing_manifest.get("releaseTag") != args.release_tag
                ):
                    raise RuntimeError(
                        "existing GitHub release manifest has a different immutable identity"
                    )
                existing_assets = merge_release_assets(
                    existing_manifest.get("assets", []), []
                )
                existing_asset_paths = download_existing_release_assets(
                    existing_manifest,
                    existing_directory,
                    args.release_tag,
                    args.github_repository,
                )
                verified_existing_shards = validate_existing_release_shards(
                    existing_manifest,
                    existing_assets,
                    existing_asset_paths,
                    aws_plan,
                    args.version,
                    args.commit,
                )
                verified_existing_matrix = verified_release_matrix_coverage(
                    existing_manifest,
                    verified_existing_shards,
                    existing_asset_paths,
                    aws_plan,
                )
                maven_archives.update({
                    name: path for name, path in existing_asset_paths.items()
                    if name.startswith("maven-repository-")
                    and name.endswith(".tar.gz")
                })
        assets: list[dict[str, Any]] = []
        for item in executions:
            shard = item["shard"]["id"]
            status_value = get_json(container, f"{prefix}/{shard}/status.json")
            if not status_value or int(status_value.get("exitCode", 1)) != 0:
                raise RuntimeError(f"shard {shard} is incomplete or failed: {status_value}")
            manifest_value = get_json(
                container, f"{prefix}/{shard}/shard-manifest.json"
            )
            expected_variants = {
                variant["name"] for variant in item["shard"]["build"]["variants"]
            }
            if (
                not manifest_value
                or manifest_value.get("runId") != args.run_id
                or manifest_value.get("shard") != shard
                or manifest_value.get("commit") != args.commit
                or manifest_value.get("releaseVersion") != args.version
                or set(manifest_value.get("workloads", []))
                != set(item["shard"]["workloads"])
                or set(manifest_value.get("variants", [])) != expected_variants
            ):
                raise RuntimeError(f"shard {shard} manifest identity mismatch")
            for workload in item["shard"]["workloads"]:
                source_name = (
                    "maven-repository.tar.gz"
                    if workload == "maven"
                    else "sdk-assets.tar.gz"
                )
                output_name = (
                    f"{source_name.removesuffix('.tar.gz')}-{shard}.tar.gz"
                )
                output = directory / output_name
                output.write_bytes(
                    container.download_blob(f"{prefix}/{shard}/{source_name}").readall()
                )
                verify_attested_file(manifest_value, source_name, output)
                if workload == "maven":
                    maven_archives[output_name] = output
                assets.append({
                    "fileName": output_name,
                    "sha256": file_digest(output),
                    "size": output.stat().st_size,
                    "shard": shard,
                    "provider": "azure",
                    "sourceObject": f"{prefix}/{shard}/{source_name}",
                })
            manifest_path = directory / f"{shard}-shard-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            assets.append({
                "fileName": manifest_path.name,
                "sha256": file_digest(manifest_path),
                "size": manifest_path.stat().st_size,
                "shard": shard,
                "provider": "azure",
                "sourceObject": f"{prefix}/{shard}/shard-manifest.json",
            })
        combined_assets = merge_release_assets(
            (existing_manifest or {}).get("assets", []), assets
        )
        repository_dir = directory / "azure-maven-repository"
        repository_manifest = directory / "azure-maven-repository-manifest.json"
        central_tool = ROOT / "release/central/repository.py"
        command = [
            sys.executable,
            str(central_tool),
            "materialize-test-repository",
            "--output", str(repository_dir),
            "--manifest", str(repository_manifest),
            "--release-version", args.version,
            "--commit", args.commit,
        ]
        for archive in sorted(maven_archives.values()):
            command.extend(["--input", str(archive)])
        subprocess.run(command, check=True)
        repository_prefix = f"{prefix}/maven2"
        for path in sorted(repository_dir.rglob("*")):
            if path.is_file():
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                container.upload_blob(
                    f"{repository_prefix}/{path.relative_to(repository_dir).as_posix()}",
                    path.read_bytes(),
                    overwrite=True,
                    content_settings=context["modules"]["ContentSettings"](
                        content_type=content_type
                    ),
                )
        container.upload_blob(
            f"{repository_prefix}/.dl4j/manifest.json",
            repository_manifest.read_bytes(),
            overwrite=True,
            content_settings=context["modules"]["ContentSettings"](
                content_type="application/json"
            ),
        )
        current_shards = sorted(item["shard"]["id"] for item in executions)
        expected_matrix = matrix_coverage(
            aws_plan, [item["id"] for item in aws_plan["shards"]]
        )
        current_matrix = execution_matrix_coverage(executions)
        current_missing_matrix = sorted(expected_matrix - current_matrix)
        combined_shards = sorted(set(current_shards) | verified_existing_shards)
        combined_matrix = current_matrix | verified_existing_matrix
        missing_matrix = sorted(expected_matrix - combined_matrix)
        put_json(container, f"{repository_prefix}/.dl4j/complete.json", {
            "schemaVersion": 1,
            "layout": "maven2",
            "ready": True,
            "provider": "azure",
            "runId": args.run_id,
            "releaseVersion": args.version,
            "commit": args.commit,
            "azureMatrixComplete": len(executions) == len(run.get("executions", [])),
            "azureMissingMatrixEntries": current_missing_matrix,
            "matrixEntries": sorted(combined_matrix),
            "completeMatrix": not missing_matrix,
            "missingMatrixEntries": missing_matrix,
            "missingWorkflows": (
                sorted(plan.get("unsupportedWorkflows", {}))
                if any(
                    value.startswith("macos-14-arm64-cpu--")
                    for value in missing_matrix
                )
                else []
            ),
            "manifestSha256": file_digest(repository_manifest),
        }, context["modules"])
        current_workloads = {
            workload
            for item in executions
            for workload in item["shard"].get("workloads", [])
        }
        aws_shards_by_id = {item["id"]: item for item in aws_plan["shards"]}
        verified_existing_workloads = {
            workload
            for shard_id in verified_existing_shards
            for workload in aws_shards_by_id[
                _selector_parts(shard_id)[0]
            ].get("workloads", [])
        }
        combined_workloads = sorted(current_workloads | verified_existing_workloads)
        manifest = {
            "schemaVersion": 1,
            "provider": merged_release_provider(existing_manifest),
            "runId": args.run_id,
            "releaseTag": args.release_tag,
            "releaseVersion": args.version,
            "commit": args.commit,
            "subscription": context["subscription"],
            "location": location,
            "resourceGroup": group,
            "storageAccount": account.name,
            "container": artifact_container_name(plan),
            "workloads": combined_workloads,
            "shards": combined_shards,
            "matrixEntries": sorted(combined_matrix),
            "completeMatrix": not missing_matrix,
            "missingMatrixEntries": missing_matrix,
            "missingWorkflows": (
                sorted(plan.get("unsupportedWorkflows", {}))
                if any(
                    value.startswith("macos-14-arm64-cpu--")
                    for value in missing_matrix
                )
                else []
            ),
            "testMavenRepository": {
                "uri": (
                    f"https://{account.name}.blob.core.windows.net/"
                    f"{artifact_container_name(plan)}/{repository_prefix}"
                ),
                "layout": "maven2",
                "ready": True,
                "completeMatrix": not missing_matrix,
                "missingMatrixEntries": missing_matrix,
            },
            "assets": sorted(
                combined_assets.values(), key=lambda value: value["fileName"]
            ),
        }
        manifest_path = directory / "release-build-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum = directory / "release-build-manifest.json.sha256"
        checksum.write_text(
            f"{file_digest(manifest_path)}  {manifest_path.name}\n",
            encoding="ascii",
        )
        if not args.no_github:
            collector_lease.check()
            assert_github_manifest_unchanged(
                args.release_tag,
                args.github_repository,
                release_exists,
                existing_manifest_digest,
                directory / "pre-upload-manifest-check",
            )
            if not release_exists:
                subprocess.run(
                    [
                        "gh", "release", "create", args.release_tag,
                        "--repo", args.github_repository,
                        "--target", args.commit, "--draft",
                        "--title", f"DL4J {args.version} external build",
                    ],
                    check=True,
                )
            asset_files = [
                str(directory / item["fileName"])
                for item in assets
                if item["fileName"] not in existing_assets
            ]
            if asset_files:
                subprocess.run(
                    [
                        "gh", "release", "upload", args.release_tag,
                        "--repo", args.github_repository, *asset_files,
                    ],
                    check=True,
                )
            collector_lease.check()
            if release_exists:
                assert_github_manifest_unchanged(
                    args.release_tag,
                    args.github_repository,
                    True,
                    existing_manifest_digest,
                    directory / "final-manifest-check",
                )
            elif "release-build-manifest.json" in github_release_asset_names(
                args.release_tag, args.github_repository
            ):
                raise RuntimeError(
                    "GitHub release manifest appeared concurrently; rerun collect to merge it"
                )
            subprocess.run(
                [
                    "gh", "release", "upload", args.release_tag,
                    "--repo", args.github_repository, "--clobber",
                    str(manifest_path), str(checksum),
                ],
                check=True,
            )
            collector_lease.check()
            _, uploaded_digest = download_github_release_manifest(
                args.release_tag,
                args.github_repository,
                directory / "uploaded-manifest-check",
                required=True,
            )
            if uploaded_digest != file_digest(manifest_path):
                raise RuntimeError(
                    "uploaded GitHub release manifest does not match the collected manifest"
                )
        print(json.dumps(manifest, indent=2))


def collect(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context, location, group, account, service = existing_storage(args, plan)
    control_container = service.get_container_client(control_container_name(plan))
    lease = ControllerLease(control_container, controller_lock_blob(plan)).acquire()
    primary_error: BaseException | None = None
    try:
        _collect_under_controller_lease(
            args, plan, context, location, group, account, service, lease
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        release_errors = lease.release()
        if release_errors:
            message = "Azure collector lease cleanup failed: " + "; ".join(release_errors)
            if primary_error is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr)


def managed_tags(value: Any) -> bool:
    return (object_value(value, "tags", {}) or {}).get(MANAGED_TAG) == "true"


def fence_release_controller(
    context: dict[str, Any],
    plan: dict[str, Any],
    group: str,
    account_name: str,
) -> tuple[str, Any, Any, ControllerLeaseGroup]:
    account = context["storage"].storage_accounts.get_properties(group, account_name)
    keys = context["storage"].storage_accounts.list_keys(group, account_name)
    values = list(object_value(keys, "keys", []) or [])
    if not values:
        raise RuntimeError(f"Azure storage account {account_name} returned no keys")
    service = context["modules"]["BlobServiceClient"](
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=str(object_value(values[0], "value")),
    )
    control_container = service.get_container_client(control_container_name(plan))
    artifact_container = service.get_container_client(artifact_container_name(plan))

    def break_lease(name: str) -> None:
        try:
            lock_blob = control_container.get_blob_client(name)
            context["modules"]["BlobLeaseClient"](lock_blob).break_lease(
                lease_break_period=0
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            error_code = str(getattr(exc, "error_code", "") or "")
            no_lease = status_code == 409 and error_code in {
                "LeaseAlreadyBroken",
                "LeaseNotPresentWithLeaseOperation",
            }
            if status_code != 404 and not no_lease:
                raise RuntimeError(
                    f"could not break Azure release controller lease {name!r}"
                ) from exc

    global_lock = controller_lock_blob(plan)
    break_lease(global_lock)
    fence_lease = ControllerLease(control_container, global_lock).acquire()
    run_fences: list[ControllerLease] = []
    try:
        fence_lease.check()
        set_kill_switch(
            control_container,
            plan,
            True,
            context["modules"],
            "stop-everything fenced",
            controller_lease=fence_lease,
            force=True,
        )
        # The forced switch prevents new controllers from mutating resources.
        # Break and hold every already-visible run lease before deletion starts.
        for item in control_container.list_blobs(
            name_starts_with=run_controller_prefix(plan)
        ):
            fence_lease.check()
            name = str(object_value(item, "name", ""))
            if not name.endswith("/kill-switch.json"):
                continue
            break_lease(name)
            run_fences.append(ControllerLease(control_container, name).acquire())
        group_lease = ControllerLeaseGroup(fence_lease, run_fences)
        group_lease.check()
    except BaseException:
        for lease in reversed(run_fences):
            lease.release()
        fence_lease.release()
        raise
    return str(account.id), control_container, artifact_container, group_lease


def _stop_fenced_resources(
    args: argparse.Namespace,
    plan: dict[str, Any],
    context: dict[str, Any],
    location: str,
    group: str,
    storage_scope: str,
    control_container: Any,
    artifact_container: Any,
    fence_lease: ControllerLease,
) -> None:
    errors: list[str] = []
    fence_lease.check()

    deleted: dict[str, list[str]] = {
        "virtualMachines": [],
        "networkInterfaces": [],
        "publicIps": [],
        "disks": [],
        "managedIdentities": [],
    }

    def delete_phase(
        label: str,
        values: Iterable[Any],
        begin_delete: Any,
        output_key: str,
    ) -> None:
        pending: list[tuple[str, Any]] = []
        fence_lease.check()
        for value in values:
            fence_lease.check()
            managed_disk_name = (
                label == "OS disk"
                and str(getattr(value, "name", "")).startswith("dl4j-osdisk-")
            )
            if not managed_tags(value) and not managed_disk_name:
                continue
            try:
                operation = begin_delete(value.name)
            except Exception as exc:
                if not is_not_found(exc):
                    errors.append(f"{label} {value.name}: {exc}")
                continue
            fence_lease.check()
            pending.append((f"{label} {value.name}", operation))
            deleted[output_key].append(str(value.name))
        for item_label, operation in pending:
            fence_lease.check()
            try:
                wait_operation(operation)
            except Exception as exc:
                if not is_not_found(exc):
                    errors.append(f"{item_label}: {exc}")
            fence_lease.check()

    try:
        delete_phase(
            "VM",
            context["compute"].virtual_machines.list(group),
            lambda name: context["compute"].virtual_machines.begin_delete(group, name),
            "virtualMachines",
        )
    except Exception as exc:
        if not is_not_found(exc):
            errors.append(f"VM discovery: {exc}")
    fence_lease.check()
    try:
        delete_phase(
            "NIC",
            context["network"].network_interfaces.list(group),
            lambda name: context["network"].network_interfaces.begin_delete(group, name),
            "networkInterfaces",
        )
    except Exception as exc:
        if not is_not_found(exc):
            errors.append(f"NIC discovery: {exc}")
    fence_lease.check()
    try:
        delete_phase(
            "public IP",
            context["network"].public_ip_addresses.list(group),
            lambda name: context["network"].public_ip_addresses.begin_delete(group, name),
            "publicIps",
        )
    except Exception as exc:
        if not is_not_found(exc):
            errors.append(f"public IP discovery: {exc}")
    fence_lease.check()
    try:
        delete_phase(
            "OS disk",
            context["compute"].disks.list_by_resource_group(group),
            lambda name: context["compute"].disks.begin_delete(group, name),
            "disks",
        )
    except Exception as exc:
        if not is_not_found(exc):
            errors.append(f"OS disk discovery: {exc}")
    fence_lease.check()

    identity_names, identity_errors = cleanup_managed_identities(
        context, group, storage_scope, fence_check=fence_lease.check
    )
    deleted["managedIdentities"].extend(identity_names)
    errors.extend(identity_errors)
    fence_lease.check()

    deleted_logs: list[str] = []
    purged: list[str] = []
    if artifact_container is not None and (args.purge_logs or args.purge_storage):
        prefix = f"{plan['artifactPrefix'].strip('/')}/"
        for item in artifact_container.list_blobs(name_starts_with=prefix):
            fence_lease.check()
            is_log = item.name.endswith(("/build.log", "/live.log")) or "/events/" in item.name
            if args.purge_storage or (args.purge_logs and is_log):
                try:
                    artifact_container.delete_blob(
                        item.name, delete_snapshots="include"
                    )
                    (deleted_logs if is_log else purged).append(item.name)
                except Exception as exc:
                    errors.append(f"delete blob {item.name}: {exc}")
                fence_lease.check()
    fence_lease.check()
    if control_container is not None:
        try:
            set_kill_switch(
                control_container,
                plan,
                True,
                context["modules"],
                "stop-everything completed",
                controller_lease=fence_lease,
                force=True,
            )
        except Exception as exc:
            errors.append(f"restore kill switch: {exc}")
    fence_lease.check()
    print(json.dumps({
        "subscription": context["subscription"],
        "location": location,
        "resourceGroup": group,
        "killSwitch": control_container is not None,
        "deleted": deleted,
        "deletedAzureBlobLogs": deleted_logs,
        "deletedStorageObjects": purged,
        "errors": errors,
    }, indent=2))
    if errors:
        raise RuntimeError(
            "emergency shutdown completed best-effort but could not verify every target"
        )


def stop_everything(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context = cloud_context(
        args.subscription, allow_wizard=not getattr(args, "no_wizard", False)
    )
    location = resolve_location(
        args.location, allow_wizard=not getattr(args, "no_wizard", False)
    )
    group = resource_group_name(location, args.resource_group)
    account_name = storage_account_name(
        context["subscription"], location, args.storage_account
    )
    try:
        storage_scope, control_container, artifact_container, fence_lease = (
            fence_release_controller(context, plan, group, account_name)
        )
    except Exception as exc:
        raise RuntimeError(
            "refusing Azure resource deletion because the controller could not be fenced"
        ) from exc
    primary_error: BaseException | None = None
    try:
        _stop_fenced_resources(
            args,
            plan,
            context,
            location,
            group,
            storage_scope,
            control_container,
            artifact_container,
            fence_lease,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        release_errors = fence_lease.release()
        if release_errors:
            message = "Azure emergency fence cleanup failed: " + "; ".join(
                release_errors
            )
            if primary_error is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr)


def add_selection_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--shard", action="append", help="lane or lane--variant; repeatable")
    command.add_argument(
        "--exclude-shard",
        action="append",
        help="exclude a lane or lane--variant; repeatable",
    )
    command.add_argument(
        "--machine-type",
        help="force one verified Azure VM size for every selected lane",
    )
    command.add_argument(
        "--lane-machine",
        action="append",
        metavar="LANE=AZURE_VM_SIZE",
        help="force a verified size for one compatibility lane; repeatable",
    )
    command.add_argument("--build-threads", type=int)
    command.add_argument(
        "--max-cores",
        type=int,
        help="limit the selected size of each VM to this many vCPUs",
    )
    command.add_argument(
        "--max-total-cores",
        type=int,
        help="cap aggregate vCPUs across all simultaneously running lanes",
    )
    command.add_argument("--zone", help="force availability zone 1, 2, or 3")


def add_storage_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--resource-group")
    command.add_argument("--storage-account")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    result.add_argument(
        "--subscription", help="defaults to AZURE_SUBSCRIPTION_ID"
    )
    result.add_argument(
        "--location", help="defaults to AZURE_LOCATION/AZURE_DEFAULTS_LOCATION"
    )
    result.add_argument(
        "--no-wizard",
        action="store_true",
        help="fail instead of prompting when Azure configuration is incomplete",
    )
    sub = result.add_subparsers(dest="command", required=True)

    setup = sub.add_parser(
        "configure", help="validate or interactively complete Azure configuration"
    )
    setup.set_defaults(func=configure_environment)

    check = sub.add_parser("preflight")
    add_selection_options(check)
    add_storage_options(check)
    check.add_argument("--root-volume-gib", type=int)
    check.set_defaults(func=preflight)

    launch = sub.add_parser("start")
    launch.add_argument("--version", required=True)
    launch.add_argument("--snapshot-version", default="1.0.0-SNAPSHOT")
    source = launch.add_mutually_exclusive_group(required=True)
    source.add_argument("--commit")
    source.add_argument("--branch")
    launch.add_argument("--repository", default=DEFAULT_REPOSITORY)
    launch.add_argument("--run-id")
    launch.add_argument("--root-volume-gib", type=int)
    launch.add_argument("--timeout-hours", type=int, default=12)
    launch.add_argument("--reset-kill-switch", action="store_true")
    launch.add_argument(
        "--ssh-public-key",
        help="public key text or path; an existing local key/ephemeral key is otherwise used",
    )
    add_selection_options(launch)
    add_storage_options(launch)
    launch.set_defaults(func=start)

    show = sub.add_parser("status")
    show.add_argument("--run-id")
    add_storage_options(show)
    show.set_defaults(func=status)

    logs = sub.add_parser("logs")
    logs.add_argument("--run-id", required=True)
    logs.add_argument("--shard", action="append")
    logs.add_argument("--follow", action="store_true")
    add_storage_options(logs)
    logs.set_defaults(func=show_logs)

    remove = sub.add_parser("delete-logs")
    target = remove.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--all-runs", action="store_true")
    remove.add_argument("--yes", action="store_true")
    add_storage_options(remove)
    remove.set_defaults(func=delete_logs)

    gather = sub.add_parser("collect")
    gather.add_argument("--run-id", required=True)
    gather.add_argument("--release-tag", required=True)
    gather.add_argument("--version", required=True)
    gather.add_argument("--commit", required=True)
    gather.add_argument(
        "--github-repository", default="deeplearning4j/deeplearning4j"
    )
    gather.add_argument("--shard", action="append")
    gather.add_argument("--no-github", action="store_true")
    add_storage_options(gather)
    gather.set_defaults(func=collect)

    stop = sub.add_parser("stop-everything")
    stop.add_argument("--wait", action="store_true")
    stop.add_argument("--purge-storage", action="store_true")
    stop.add_argument("--purge-logs", action="store_true")
    add_storage_options(stop)
    stop.set_defaults(func=stop_everything)
    return result


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "max_cores") and args.max_cores is not None and args.max_cores < 1:
        raise SystemExit("--max-cores must be positive")
    if (
        hasattr(args, "max_total_cores")
        and args.max_total_cores is not None
        and args.max_total_cores < 1
    ):
        raise SystemExit("--max-total-cores must be positive")
    if (
        hasattr(args, "build_threads")
        and args.build_threads is not None
        and args.build_threads < 1
    ):
        raise SystemExit("--build-threads must be positive")
    if hasattr(args, "timeout_hours") and args.timeout_hours < 1:
        raise SystemExit("--timeout-hours must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
