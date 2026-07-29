#!/usr/bin/env python3
"""Run the DL4J release matrix on Google Cloud.

The controller uses Application Default Credentials (ADC), provisions one
serial build lane at a time, and reuses that lane for all of its classifier
variants so ccache/sccache are effective.  Compile-only accelerator lanes use
ordinary Compute Engine hosts.  Real Cloud TPU allocation is opt-in through
`tpu-smoke`.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
from typing import Any, Iterable

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
MANAGED_LABEL = "dl4j-release-managed"
PROJECT_LABEL = "dl4j-project"
RUN_LABEL = "dl4j-run"
SHARD_LABEL = "dl4j-shard"
BUCKET_MANAGED_LABEL = "dl4j_release_managed"
DEFAULT_REPOSITORY = "https://github.com/deeplearning4j/deeplearning4j.git"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "release/gcp/release-plan.json"
BUILD_DRIVER = ROOT / "release/aws/build-platform.py"
CLOUD_IO = ROOT / "release/gcp/cloud-io.py"
GCP_PROJECT_PATTERN = re.compile(r"^(?:[a-z][a-z0-9-]{4,28}[a-z0-9]|[0-9]{6,})$")
GCP_REGION_PATTERN = re.compile(r"^[a-z]+(?:-[a-z0-9]+)*[0-9]$")
GCP_CREDENTIAL_ERROR_NAMES = {
    "ClientCertError", "DefaultCredentialsError", "MalformedError", "OAuthError",
    "RefreshError", "UserAccessTokenError",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != 1:
        raise ValueError(f"unsupported GCP release plan schema: {value.get('schemaVersion')}")
    shards = value.get("shards") or []
    ids = [item.get("id") for item in shards]
    if not shards or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("release plan requires unique, non-empty shard ids")
    for shard in shards:
        if shard.get("os") not in {"linux", "windows"}:
            raise ValueError(f"GCP shard {shard['id']} has unsupported OS {shard.get('os')!r}")
        if not shard.get("build", {}).get("variants"):
            raise ValueError(f"shard {shard['id']} has no build variants")
        if not set(shard.get("workloads", [])) <= {"maven", "sdk"}:
            raise ValueError(f"shard {shard['id']} has an unknown workload")
    return value


def normalize_label(value: str, maximum: int = 63) -> str:
    result = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    if not result or not result[0].isalpha():
        result = "r-" + result
    return result[:maximum].rstrip("-_")


def resource_name(prefix: str, run_id: str, shard: str) -> str:
    digest = hashlib.sha1(f"{run_id}/{shard}".encode()).hexdigest()[:8]
    stem = normalize_label(f"{prefix}-{run_id}-{shard}", 52).replace("_", "-")
    return f"{stem}-{digest}"[:63].rstrip("-")


def log_id(run_id: str, shard: str) -> str:
    return normalize_label(f"deeplearning4j-release-{run_id}-{shard}", 200)


def kill_switch_object(plan: dict[str, Any]) -> str:
    return f"{plan.get('artifactPrefix', 'deeplearning4j/releases').strip('/')}/control/kill-switch.json"


def release_bucket_name(project: str, region: str, override: str | None = None) -> str:
    if override:
        return override.removeprefix("gs://").strip("/")
    return normalize_label(f"dl4j-release-{project}-{region}", 63).replace("_", "-")


def control_bucket_name(project: str) -> str:
    return normalize_label(f"dl4j-release-{project}-control", 63).replace("_", "-")


def google_modules():
    try:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession, Request
        from google.cloud import cloudquotas_v1, compute_v1, logging_v2, storage, tpu_v2
    except ImportError as exc:
        raise SystemExit(
            "Google Cloud dependencies are missing. Run: "
            "python3 -m pip install -r release/gcp/requirements.txt"
        ) from exc
    return google.auth, AuthorizedSession, Request, cloudquotas_v1, compute_v1, logging_v2, storage, tpu_v2


def interactive_wizard_enabled(allow_wizard: bool) -> bool:
    """Never make a redirected or CI controller wait for input."""
    ci = any(
        os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no"}
        for name in ("CI", "GITHUB_ACTIONS")
    )
    return bool(allow_wizard and not ci and getattr(sys.stdin, "isatty", lambda: False)())


def prompt_value(label: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            print(f"{label}{suffix}: ", end="", file=sys.stderr, flush=True)
            value = input().strip()
        except EOFError as exc:
            raise SystemExit("Google Cloud configuration wizard lost its interactive input") from exc
        if not value and default is not None:
            value = default
        if value:
            return value
        print(f"{label} is required.", file=sys.stderr)


def gcp_configuration_error(problem: str) -> SystemExit:
    return SystemExit(
        f"Google Cloud release configuration is incomplete: {problem}. Configure Application "
        "Default Credentials plus GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION, or run "
        "`python3 release/gcp/release.py configure` in an interactive terminal."
    )


def gcp_credential_exception(exc: BaseException) -> bool:
    return bool({item.__name__ for item in exc.__class__.__mro__} & GCP_CREDENTIAL_ERROR_NAMES)


def safe_gcp_credential_problem(exc: BaseException) -> str:
    return f"ADC validation returned {exc.__class__.__name__}"


def load_gcp_credentials(google_auth: Any, request_factory: Any) -> tuple[Any | None, str | None, str | None]:
    try:
        credentials, project = google_auth.default(scopes=SCOPES)
        if not getattr(credentials, "valid", True):
            credentials.refresh(request_factory())
        return credentials, project, None
    except Exception as exc:
        if gcp_credential_exception(exc):
            return None, None, safe_gcp_credential_problem(exc)
        raise


def configured_gcp_project(project_override: str | None = None) -> str | None:
    return (
        project_override
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("CLOUDSDK_CORE_PROJECT")
    )


def valid_gcp_project(project: str | None) -> bool:
    return bool(project and GCP_PROJECT_PATTERN.fullmatch(project))


def valid_gcp_region(region: str | None) -> bool:
    return bool(region and GCP_REGION_PATTERN.fullmatch(region))


def prompt_gcp_project(project_override: str | None = None) -> str:
    current = configured_gcp_project(project_override)
    while True:
        project = prompt_value("Google Cloud project ID", default=current)
        if valid_gcp_project(project):
            os.environ["GOOGLE_CLOUD_PROJECT"] = project
            return project
        print("Enter a project ID (for example deeplearning4j-release) or numeric project number.", file=sys.stderr)


def configure_gcp_credentials(google_auth: Any, request_factory: Any,
                              project_override: str | None, initial_problem: str) -> tuple[Any, str | None]:
    """Interactively select an official ADC source; credential contents are never printed or copied."""
    print(f"Google Cloud credential setup is required ({initial_problem}).", file=sys.stderr)
    project = configured_gcp_project(project_override)
    if not valid_gcp_project(project):
        project = prompt_gcp_project(project_override)
    current_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    default_source = "file" if current_file else "gcloud"

    while True:
        source = prompt_value("ADC source (file/gcloud)", default=default_source).lower()
        if source in {"file", "f"}:
            path_text = prompt_value("GOOGLE_APPLICATION_CREDENTIALS file", default=current_file)
            path = Path(path_text).expanduser().resolve()
            if not path.is_file():
                print("That ADC JSON file does not exist.", file=sys.stderr)
                continue
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
        elif source in {"gcloud", "login", "g"}:
            executable = shutil.which("gcloud")
            if not executable:
                print(
                    "gcloud is not installed; choose `file` or install the Google Cloud CLI.",
                    file=sys.stderr,
                )
                continue
            old_file = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            command = [executable, "auth", "application-default", "login"]
            if project:
                command.extend(["--project", project])
            if subprocess.run(command, check=False).returncode != 0:
                if old_file:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_file
                print("gcloud did not create Application Default Credentials.", file=sys.stderr)
                continue
        else:
            print("Choose `file` or `gcloud`.", file=sys.stderr)
            continue

        credentials, adc_project, problem = load_gcp_credentials(google_auth, request_factory)
        if problem is None:
            print("Google Application Default Credentials validated for this invocation.", file=sys.stderr)
            return credentials, adc_project
        print(f"That ADC source is not usable: {problem}.", file=sys.stderr)


def cloud_context(project_override: str | None = None, *, allow_wizard: bool = True):
    (
        google_auth, authorized_session, request_factory, cloudquotas_v1,
        compute_v1, logging_v2, storage, tpu_v2,
    ) = google_modules()
    credentials, adc_project, credential_problem = load_gcp_credentials(google_auth, request_factory)
    if credential_problem:
        if not interactive_wizard_enabled(allow_wizard):
            raise gcp_configuration_error(credential_problem)
        print("Google Cloud release environment wizard", file=sys.stderr)
        credentials, adc_project = configure_gcp_credentials(
            google_auth, request_factory, project_override, credential_problem
        )
    project = (
        project_override
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("CLOUDSDK_CORE_PROJECT")
        or adc_project
    )
    if not valid_gcp_project(project):
        problem = "no valid Google Cloud project ID was resolved"
        if not interactive_wizard_enabled(allow_wizard):
            raise gcp_configuration_error(problem)
        print("Google Cloud release environment wizard", file=sys.stderr)
        project = prompt_gcp_project(project_override)
    return {
        "project": project,
        "credentials": credentials,
        "AuthorizedSession": authorized_session,
        "quotas": cloudquotas_v1,
        "compute": compute_v1,
        "logging": logging_v2,
        "storage": storage,
        "tpu": tpu_v2,
    }


def resolve_region(value: str | None, *, allow_wizard: bool = True) -> str:
    region = value or os.environ.get("GOOGLE_CLOUD_REGION") or os.environ.get("CLOUDSDK_COMPUTE_REGION")
    if not valid_gcp_region(region):
        problem = "no valid Compute Engine region was resolved"
        if not interactive_wizard_enabled(allow_wizard):
            raise gcp_configuration_error(problem)
        print("Google Cloud release environment wizard", file=sys.stderr)
        while True:
            region = prompt_value("Google Cloud region", default=region)
            if valid_gcp_region(region):
                os.environ["GOOGLE_CLOUD_REGION"] = region
                break
            print("Enter a region such as us-central1, not a zone such as us-central1-a.", file=sys.stderr)
    return region


def configure_environment(args: argparse.Namespace) -> None:
    context = cloud_context(args.project, allow_wizard=not getattr(args, "no_wizard", False))
    region = resolve_region(args.region, allow_wizard=not getattr(args, "no_wizard", False))
    credentials = context["credentials"]
    credential_source = (
        "GOOGLE_APPLICATION_CREDENTIALS"
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        else credentials.__class__.__name__
    )
    print(json.dumps({
        "configured": True,
        "project": context["project"],
        "region": region,
        "credentialSource": credential_source,
        "nonSecretEnvironmentForFutureCommands": {
            "GOOGLE_CLOUD_PROJECT": context["project"],
            "GOOGLE_CLOUD_REGION": region,
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
    return normalize_label(f"{version}-{suffix}", 63)


def _selector_parts(selector: str) -> tuple[str, str | None]:
    if "--" not in selector:
        return selector, None
    return tuple(selector.rsplit("--", 1))  # type: ignore[return-value]


def selected_executions(
    plan: dict[str, Any], selectors: list[str] | None = None, exclusions: list[str] | None = None
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
                variants = [item for item in shard["build"]["variants"] if item["name"] == variant]
                if not variants:
                    raise ValueError(f"unknown variant selector: {selector}")
                shard["build"]["variants"] = variants
                shard["parentShard"] = parent
                shard["id"] = selector
            selected.append(shard)
    result: list[dict[str, Any]] = []
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
    for shard in selected:
        parent = shard.get("parentShard", shard["id"])
        if parent in excluded_lanes:
            continue
        blocked = excluded_variants.get(parent, set())
        shard["build"]["variants"] = [item for item in shard["build"]["variants"] if item["name"] not in blocked]
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
            covered.update(f"{parent}--{item['name']}" for item in shard["build"]["variants"])
        elif any(item["name"] == variant for item in shard["build"]["variants"]):
            covered.add(f"{parent}--{variant}")
    return covered


def merged_release_provider(existing_manifest: dict[str, Any] | None) -> str:
    if not existing_manifest:
        return "gcp"
    return "gcp" if existing_manifest.get("provider") == "gcp" else "hybrid"


def candidate_names(plan: dict[str, Any], shard: dict[str, Any], override: str | None = None) -> list[str]:
    if override:
        return [override]
    explicit = shard.get("machineCandidates")
    if explicit:
        return list(explicit)
    key = "armMachineCandidates" if shard.get("machineClass") == "arm" else "x86MachineCandidates"
    return list(plan["defaults"][key])


def zone_region(zone: Any) -> str:
    return str(getattr(zone, "region", "")).rstrip("/").split("/")[-1]


def available_zones(zones_client: Any, project: str, region: str, requested: str | None = None) -> list[str]:
    zones = [
        item.name for item in zones_client.list(project=project)
        if zone_region(item) == region and str(getattr(item, "status", "UP")) == "UP"
    ]
    if requested:
        if requested not in zones:
            raise RuntimeError(f"zone {requested} is not an UP Compute Engine zone in region {region}")
        return [requested]
    if not zones:
        raise RuntimeError(f"no UP Compute Engine zones found in region {region}")
    return sorted(zones)


def parse_named_resource(value: str, collection: str, default_project: str) -> tuple[str, str]:
    parts = value.strip("/").split("/")
    if "projects" in parts and collection in parts:
        return parts[parts.index("projects") + 1], parts[parts.index(collection) + 1]
    return default_project, value


def resolve_network(
    compute: Any, credentials: Any, project: str, region: str,
    network_value: str | None, subnetwork_value: str | None,
) -> dict[str, str | None]:
    network_project, network_name = parse_named_resource(network_value or "default", "networks", project)
    network = compute.NetworksClient(credentials=credentials).get(project=network_project, network=network_name)
    subnetwork_link = None
    if subnetwork_value:
        subnet_project, subnet_name = parse_named_resource(subnetwork_value, "subnetworks", network_project)
        subnetwork = compute.SubnetworksClient(credentials=credentials).get(
            project=subnet_project, region=region, subnetwork=subnet_name
        )
        if str(subnetwork.network).rstrip("/") != str(network.self_link).rstrip("/"):
            raise RuntimeError(f"subnetwork {subnetwork.name} does not belong to network {network.name}")
        subnetwork_link = subnetwork.self_link
    elif not bool(getattr(network, "auto_create_subnetworks", False)):
        raise RuntimeError(f"network {network.name} is custom mode; pass --subnetwork in region {region}")
    return {"network": network.self_link, "subnetwork": subnetwork_link}


def not_found(exc: Exception) -> bool:
    try:
        from google.api_core.exceptions import NotFound
    except ImportError:
        return exc.__class__.__name__ == "NotFound"
    return isinstance(exc, NotFound) or exc.__class__.__name__ == "NotFound"


def choose_machine_live(
    machine_client: Any,
    project: str,
    zones: list[str],
    candidates: list[str],
    max_cores: int | None,
) -> dict[str, Any]:
    offerings: dict[str, list[str]] = {}
    definitions: dict[str, Any] = {}
    errors: list[str] = []
    for name in candidates:
        for zone in zones:
            try:
                value = machine_client.get(project=project, zone=zone, machine_type=name)
            except Exception as exc:
                if not_found(exc):
                    continue
                errors.append(f"{zone}/{name}: {exc}")
                continue
            cpus = int(value.guest_cpus)
            offerings.setdefault(name, []).append(zone)
            definitions[name] = value
    alternatives = []
    for name in candidates:
        if name not in definitions:
            continue
        value = definitions[name]
        cpus = int(value.guest_cpus)
        if max_cores is not None and cpus > max_cores:
            continue
        alternatives.append({
            "machineType": name,
            "zone": offerings[name][0],
            "vcpus": cpus,
            "memoryGiB": round(int(value.memory_mb) / 1024, 2),
            "zones": offerings[name],
        })
    alternatives.sort(key=lambda item: (-item["vcpus"], candidates.index(item["machineType"])))
    if alternatives:
        return {**alternatives[0], "offerings": offerings, "launchAlternatives": alternatives}
    constraint = f" within --max-cores={max_cores}" if max_cores is not None else ""
    details = f"; API errors: {errors[:3]}" if errors else ""
    raise RuntimeError(f"none of the configured machine candidates exist in the selected region{constraint}{details}")


def resolve_image(images_client: Any, shard: dict[str, Any]) -> dict[str, str]:
    image = images_client.get_from_family(project=shard["imageProject"], family=shard["imageFamily"])
    status = str(getattr(image, "status", "READY"))
    if status != "READY":
        raise RuntimeError(f"image family {shard['imageFamily']} resolved to non-READY image {image.name}")
    expected = "ARM64" if shard["architecture"] == "arm64" else "X86_64"
    actual = str(getattr(image, "architecture", expected)).upper()
    if actual and actual != expected:
        raise RuntimeError(f"image {image.name} architecture is {actual}; expected {expected}")
    return {"name": image.name, "selfLink": image.self_link, "architecture": actual}


def adapt_build_resources(shard: dict[str, Any], vcpus: int, memory_gib: float, threads_override: int | None) -> None:
    desired_threads = int(shard["build"].get("buildThreads", 32))
    threads = threads_override or min(desired_threads, max(1, vcpus // 2))
    if threads < 1 or threads > vcpus:
        raise ValueError(f"build threads must be between 1 and selected vCPUs ({vcpus})")
    desired_heap = int(shard["build"].get("mavenHeapGiB", 24))
    heap = min(desired_heap, max(4, int(memory_gib // 4)))
    shard["build"]["buildThreads"] = threads
    shard["build"]["mavenHeapGiB"] = heap


def quota_family(machine_type: str) -> str:
    if machine_type.startswith("c4a-"):
        return "C4A"
    if machine_type.startswith("c4d-"):
        return "C4D"
    if machine_type.startswith("c4-"):
        return "C4"
    if machine_type.startswith("t2a-"):
        return "T2A"
    return "GENERAL"


def cloud_family_limits(quota_infos: Iterable[Any], region: str) -> dict[str, float]:
    limits: dict[str, float] = {}
    for info in quota_infos:
        metric = str(getattr(info, "metric", "")).lower()
        if not metric.endswith("/cpus_per_vm_family"):
            continue
        for dimensions_info in (getattr(info, "dimensions_infos", None) or []):
            dimensions = {str(key).lower(): str(value) for key, value in dict(dimensions_info.dimensions).items()}
            locations = set(getattr(dimensions_info, "applicable_locations", None) or [])
            dimension_region = dimensions.get("region")
            if dimension_region and dimension_region != region:
                continue
            if not dimension_region and locations and region not in locations:
                continue
            family = dimensions.get("vm_family", "").upper()
            if family:
                limits[family] = float(dimensions_info.details.value)
    return limits


def quota_report(
    region_resource: Any, schedule: list[dict[str, Any]], family_limits: dict[str, float] | None = None
) -> dict[str, Any]:
    legacy_quotas = {
        str(item.metric): {"limit": float(item.limit), "usage": float(item.usage), "remaining": float(item.limit) - float(item.usage)}
        for item in (getattr(region_resource, "quotas", None) or [])
    }
    family_limits = family_limits or {}
    required: dict[str, int] = {}
    for item in schedule:
        family = quota_family(item["machineType"])
        required[family] = max(required.get(family, 0), int(item["vcpus"]))
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for family, cores in required.items():
        if family in family_limits:
            available = {"limit": family_limits[family], "usage": None, "remaining": None, "source": "Cloud Quotas API"}
            checks[family] = {"required": cores, "enforcedMetric": "CPUS_PER_VM_FAMILY", "quota": available}
            if family_limits[family] < cores:
                failures.append(
                    f"{family} serial lane needs {cores} vCPUs but its CPUS_PER_VM_FAMILY limit is {family_limits[family]:g}"
                )
            continue
        selected_metric = "CPUS" if family == "GENERAL" and "CPUS" in legacy_quotas else None
        available = legacy_quotas.get(selected_metric) if selected_metric else None
        checks[family] = {"required": cores, "enforcedMetric": selected_metric, "quota": available}
        if available is None:
            failures.append(
                f"no readable CPUS_PER_VM_FAMILY quota was returned for {family}; enable cloudquotas.googleapis.com "
                "and grant roles/cloudquotas.viewer"
            )
        elif available["remaining"] < cores:
            failures.append(
                f"{family} serial lane needs {cores} vCPUs but {selected_metric} has "
                f"{available['remaining']:g} remaining ({available['usage']:g}/{available['limit']:g} used)"
            )
    return {
        "checks": checks,
        "failures": failures,
        "allRegionalCpuQuotas": {key: value for key, value in legacy_quotas.items() if "CPU" in key},
        "cloudVmFamilyLimits": family_limits,
    }


def preflight_data(args: argparse.Namespace, *, include_clients: bool = False) -> dict[str, Any]:
    plan = load_plan(args.plan)
    executions = selected_executions(plan, args.shard, getattr(args, "exclude_shard", None))
    context = cloud_context(args.project, allow_wizard=not getattr(args, "no_wizard", False))
    project = context["project"]
    region = resolve_region(args.region, allow_wizard=not getattr(args, "no_wizard", False))
    compute = context["compute"]
    zones_client = compute.ZonesClient(credentials=context["credentials"])
    machine_client = compute.MachineTypesClient(credentials=context["credentials"])
    images_client = compute.ImagesClient(credentials=context["credentials"])
    regions_client = compute.RegionsClient(credentials=context["credentials"])
    projects_client = compute.ProjectsClient(credentials=context["credentials"])
    quotas_client = context["quotas"].CloudQuotasClient(credentials=context["credentials"])
    project_number = str(projects_client.get(project=project).id)
    quota_infos = quotas_client.list_quota_infos(
        parent=f"projects/{project_number}/locations/global/services/compute.googleapis.com"
    )
    family_limits = cloud_family_limits(quota_infos, region)
    zones = available_zones(zones_client, project, region, getattr(args, "zone", None))
    network = resolve_network(
        compute, context["credentials"], project, region,
        getattr(args, "network", None), getattr(args, "subnetwork", None),
    )
    machine_cache: dict[tuple[str, tuple[str, ...], int | None], dict[str, Any]] = {}
    image_cache: dict[tuple[str, str], dict[str, str]] = {}
    schedule: list[dict[str, Any]] = []
    for execution in executions:
        candidates = candidate_names(plan, execution, getattr(args, "machine_type", None))
        family = quota_family(candidates[0])
        constraints = [value for value in (getattr(args, "max_cores", None), family_limits.get(family)) if value is not None]
        effective_max_cores = int(min(constraints)) if constraints else None
        cache_key = (execution.get("machineClass", "x86"), tuple(candidates), effective_max_cores)
        if cache_key not in machine_cache:
            machine_cache[cache_key] = choose_machine_live(
                machine_client, project, zones, candidates, effective_max_cores
            )
        machine = copy.deepcopy(machine_cache[cache_key])
        image_key = (execution["imageProject"], execution["imageFamily"])
        if image_key not in image_cache:
            image_cache[image_key] = resolve_image(images_client, execution)
        adapt_build_resources(execution, machine["vcpus"], machine["memoryGiB"], getattr(args, "build_threads", None))
        schedule.append({
            "id": execution["id"],
            "parentShard": execution.get("parentShard", execution["id"]),
            "machineType": machine["machineType"],
            "zone": machine["zone"],
            "vcpus": machine["vcpus"],
            "memoryGiB": machine["memoryGiB"],
            "buildThreads": execution["build"]["buildThreads"],
            "mavenHeapGiB": execution["build"]["mavenHeapGiB"],
            "image": image_cache[image_key],
            "offerings": machine["offerings"],
            "launchAlternatives": machine["launchAlternatives"],
            "shard": execution,
        })
    quota = quota_report(
        regions_client.get(project=project, region=region), schedule, family_limits
    )
    result = {
        "project": project,
        "projectNumber": project_number,
        "region": region,
        "executions": len(schedule),
        "serialLanes": True,
        "peakVcpusByMachineQuota": {metric: value["required"] for metric, value in quota["checks"].items()},
        "quota": quota,
        "network": network,
        "schedule": schedule,
        "unsupportedWorkflows": plan.get("unsupportedWorkflows", {}),
    }
    if include_clients:
        result["_context"] = context
        result["_plan"] = plan
    return result


def printable_preflight(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def preflight(args: argparse.Namespace) -> None:
    value = preflight_data(args)
    print(json.dumps(printable_preflight(value), indent=2))
    if value["quota"]["failures"]:
        raise SystemExit("Preflight failed: " + "; ".join(value["quota"]["failures"]))
    if getattr(args, "include_tpu_smoke", False):
        validate_tpu_configuration(args, load_plan(args.plan), cloud_context(args.project, allow_wizard=not getattr(args, "no_wizard", False)))


def ensure_bucket(context: dict[str, Any], project: str, region: str, name: str):
    storage_client = context["storage"].Client(project=project, credentials=context["credentials"])
    bucket = storage_client.lookup_bucket(name)
    if bucket is None:
        bucket = storage_client.bucket(name)
        bucket.storage_class = "STANDARD"
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.iam_configuration.public_access_prevention = "enforced"
        bucket = storage_client.create_bucket(bucket, location=region)
    location = str(bucket.location or "").lower()
    if location and location != region.lower():
        raise RuntimeError(f"bucket gs://{name} is in {bucket.location}, not requested region {region}")
    bucket.versioning_enabled = False
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    bucket.iam_configuration.public_access_prevention = "enforced"
    bucket.labels = {**dict(bucket.labels or {}), BUCKET_MANAGED_LABEL: "true"}
    bucket.patch()
    # New buckets otherwise retain deleted objects for seven days. This release
    # bucket intentionally supports immediate, auditable purge of logs/artifacts.
    session = context["AuthorizedSession"](context["credentials"])
    response = session.patch(
        f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(name, safe='')}",
        json={"softDeletePolicy": {"retentionDurationSeconds": "0"}},
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"failed to disable soft delete on gs://{name}: {response.status_code} {response.text}")
    bucket.reload()
    policy = getattr(bucket, "soft_delete_policy", None)
    retention = getattr(policy, "retention_duration_seconds", 0) if policy else 0
    if retention not in (None, 0):
        raise RuntimeError(f"gs://{name} still has soft delete retention {retention}; logs would not be immediately purgeable")
    return storage_client, bucket


def ensure_control_bucket(context: dict[str, Any], project: str, region: str):
    name = control_bucket_name(project)
    client = context["storage"].Client(project=project, credentials=context["credentials"])
    existing = client.lookup_bucket(name)
    location = str(existing.location).lower() if existing and existing.location else region
    return ensure_bucket(context, project, location, name)


def is_managed_bucket(bucket: Any) -> bool:
    return dict(getattr(bucket, "labels", None) or {}).get(BUCKET_MANAGED_LABEL) == "true"


def worker_service_account_email(service_account: str | None, project_number: str) -> str:
    return service_account or f"{project_number}-compute@developer.gserviceaccount.com"


def ensure_worker_bucket_access(
    bucket: Any, service_account: str, role: str = "roles/storage.objectAdmin"
) -> None:
    member = f"serviceAccount:{service_account}"
    policy = bucket.get_iam_policy(requested_policy_version=3)
    changed = False
    for binding in policy.bindings:
        if binding.get("role") != role:
            continue
        members = set(binding.get("members", []))
        if member not in members:
            binding["members"] = sorted(members | {member})
            changed = True
        break
    else:
        policy.bindings.append({"role": role, "members": [member]})
        changed = True
    if changed:
        bucket.set_iam_policy(policy)


def get_json(bucket: Any, name: str) -> dict[str, Any] | None:
    blob = bucket.blob(name)
    try:
        return json.loads(blob.download_as_text())
    except Exception as exc:
        if not_found(exc):
            return None
        raise


def put_json(bucket: Any, name: str, value: dict[str, Any]) -> None:
    bucket.blob(name).upload_from_string(json.dumps(value, indent=2, sort_keys=True) + "\n", content_type="application/json")


def set_kill_switch(bucket: Any, plan: dict[str, Any], enabled: bool, reason: str = "controller") -> None:
    put_json(bucket, kill_switch_object(plan), {"enabled": enabled, "reason": reason, "updatedAt": utc_now()})


def render_worker(worker_path: Path, config: dict[str, Any], *, tpu: bool = False) -> str:
    worker = worker_path.read_text(encoding="utf-8")
    worker = worker.replace(
        "__DL4J_WORKER_CONFIG_B64__",
        base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode(),
    )
    worker = worker.replace("__DL4J_CLOUD_IO_B64__", base64.b64encode(CLOUD_IO.read_bytes()).decode())
    if not tpu:
        worker = worker.replace("__DL4J_BUILD_DRIVER_B64__", base64.b64encode(BUILD_DRIVER.read_bytes()).decode())
    unresolved = re.findall(r"__DL4J_[A-Z0-9_]+__", worker)
    if unresolved:
        raise RuntimeError(f"unresolved worker placeholders: {unresolved}")
    return worker


def instance_resource(context: dict[str, Any], args: argparse.Namespace, item: dict[str, Any], startup_script: str):
    compute = context["compute"]
    project = context["project"]
    zone = item["zone"]
    shard = item["shard"]
    interface = compute.NetworkInterface(
        network=item["network"]["network"],
        nic_type="GVNIC",
        access_configs=[compute.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT", network_tier="PREMIUM")],
    )
    if item["network"]["subnetwork"]:
        interface.subnetwork = item["network"]["subnetwork"]
    init = compute.AttachedDiskInitializeParams(
        source_image=item["image"]["selfLink"],
        disk_size_gb=int(args.root_volume_gib or item["planDefaults"]["rootVolumeGiB"]),
        disk_type=f"zones/{zone}/diskTypes/hyperdisk-balanced",
    )
    disk = compute.AttachedDisk(boot=True, auto_delete=True, type_="PERSISTENT", initialize_params=init)
    service_account = compute.ServiceAccount(email=args.service_account or "default", scopes=SCOPES)
    metadata_key = "windows-startup-script-ps1" if shard["os"] == "windows" else "startup-script"
    metadata = compute.Metadata(items=[compute.Items(key=metadata_key, value=startup_script), compute.Items(key="enable-oslogin", value="TRUE")])
    return compute.Instance(
        name=item["instanceName"],
        machine_type=f"zones/{zone}/machineTypes/{item['machineType']}",
        disks=[disk],
        network_interfaces=[interface],
        service_accounts=[service_account],
        metadata=metadata,
        labels={
            MANAGED_LABEL: "true",
            PROJECT_LABEL: normalize_label(item["planProjectLabel"]),
            RUN_LABEL: normalize_label(item["runId"]),
            SHARD_LABEL: normalize_label(shard["id"]),
        },
        deletion_protection=False,
    )


def emit_controller_event(logging_client: Any, log_name: str, run_id: str, shard: str, **event: Any) -> None:
    payload = {"timestamp": utc_now(), "runId": run_id, "shard": shard, **event}
    print("[dl4j-controller] " + " ".join(f"{key}={value}" for key, value in payload.items()), flush=True)
    try:
        logging_client.logger(log_name).log_struct(payload, labels={"dl4j_run_id": run_id, "dl4j_shard": shard})
    except Exception as exc:
        print(f"[{shard}] controller Cloud Logging publish failed: {exc}", flush=True)


def stream_log_entries(logging_client: Any, project: str, log_name: str, run_id: str, shard: str, seen: set[str]) -> None:
    encoded = urllib.parse.quote(log_name, safe="")
    filter_ = (
        f'logName="projects/{project}/logs/{encoded}" '
        f'AND labels.dl4j_run_id="{run_id}" AND labels.dl4j_shard="{shard}"'
    )
    try:
        entries = logging_client.list_entries(filter_=filter_, order_by="timestamp asc", page_size=1000)
        for entry in entries:
            text = getattr(entry, "payload", None)
            if isinstance(text, dict):
                text = json.dumps(text, sort_keys=True)
            text = str(text)
            key = f"{getattr(entry, 'timestamp', '')}/{text}"
            if key not in seen:
                seen.add(key)
                print(f"[{shard}/cloud-logging] {text}", flush=True)
    except Exception as exc:
        print(f"[{shard}] Cloud Logging read failed: {exc}", flush=True)


def print_retained_log(bucket: Any, prefix: str, shard: str) -> None:
    blob = bucket.blob(f"{prefix}/{shard}/build.log")
    try:
        text = blob.download_as_text()
    except Exception as exc:
        print(f"[{shard}] retained GCS build log unavailable: {exc}", flush=True)
        return
    for line in text.splitlines():
        print(f"[{shard}/gcs-build-log] {line}", flush=True)


def wait_for_lane(
    context: dict[str, Any], bucket: Any, plan: dict[str, Any], item: dict[str, Any], timeout_seconds: int
) -> dict[str, Any]:
    compute = context["compute"]
    instances = compute.InstancesClient(credentials=context["credentials"])
    logging_client = context["logging"].Client(project=context["project"], credentials=context["credentials"])
    project = context["project"]
    run_id = item["runId"]
    shard = item["shard"]["id"]
    prefix = f"{plan['artifactPrefix'].strip('/')}/{run_id}"
    status_name = f"{prefix}/{shard}/status.json"
    start_time = time.monotonic()
    serial_start = 0
    seen_logs: set[str] = set()
    last_state = None
    last_heartbeat = 0.0
    while True:
        elapsed = int(time.monotonic() - start_time)
        status = get_json(bucket, status_name)
        if status is not None:
            if int(status.get("exitCode", 1)) == 0:
                emit_controller_event(logging_client, item["logId"], run_id, shard, phase="lane", status="complete", elapsedSeconds=elapsed)
                return status
            print_retained_log(bucket, prefix, shard)
            raise RuntimeError(f"lane {shard} failed with exit code {status.get('exitCode')}; retained GCS build log printed above")
        state = "not-found"
        try:
            instance = instances.get(project=project, zone=item["zone"], instance=item["instanceName"])
            state = str(instance.status)
            # `start` exists on the request message but is not exposed as a
            # flattened keyword by the generated Python client.
            serial = instances.get_serial_port_output(request={
                "project": project, "zone": item["zone"], "instance": item["instanceName"],
                "port": 1, "start": serial_start,
            })
            contents = getattr(serial, "contents", "") or ""
            if contents:
                for line in contents.splitlines():
                    print(f"[{item['instanceName']}/console] {line}", flush=True)
            serial_start = int(getattr(serial, "next_", serial_start) or serial_start)
        except Exception as exc:
            if not not_found(exc):
                print(f"[{shard}] instance/serial status read failed: {exc}", flush=True)
        if state != last_state:
            emit_controller_event(logging_client, item["logId"], run_id, shard, phase="compute-instance", status="changed", instanceState=state, elapsedSeconds=elapsed)
            last_state = state
        stream_log_entries(logging_client, project, item["logId"], run_id, shard, seen_logs)
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            emit_controller_event(logging_client, item["logId"], run_id, shard, phase="controller-heartbeat", status="waiting", instanceState=state, elapsedSeconds=elapsed)
            last_heartbeat = now
        if state in {"TERMINATED", "not-found"} and elapsed > 90:
            time.sleep(5)
            status = get_json(bucket, status_name)
            if status is None:
                print_retained_log(bucket, prefix, shard)
                raise RuntimeError(f"lane {shard} stopped without status.json; bootstrap or service-account permissions failed")
        if elapsed >= timeout_seconds:
            print_retained_log(bucket, prefix, shard)
            raise TimeoutError(f"lane {shard} exceeded timeout of {timeout_seconds} seconds")
        time.sleep(10)


def delete_instance(context: dict[str, Any], project: str, zone: str, name: str, wait: bool = True) -> None:
    client = context["compute"].InstancesClient(credentials=context["credentials"])
    try:
        operation = client.delete(project=project, zone=zone, instance=name)
        if wait:
            operation.result(timeout=900)
    except Exception as exc:
        if not_found(exc):
            return
        raise


def retryable_capacity_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(token in text for token in (
        "resourceexhausted", "resource exhausted", "resource_pool_exhausted",
        "zone_resource_pool_exhausted", "insufficient capacity", "stockout",
        "quota_exceeded", "quota exceeded",
    ))


def start(args: argparse.Namespace) -> None:
    value = preflight_data(args, include_clients=True)
    print(json.dumps(printable_preflight(value), indent=2))
    if value["quota"]["failures"]:
        raise SystemExit("Preflight failed: " + "; ".join(value["quota"]["failures"]))
    context = value["_context"]
    plan = value["_plan"]
    project = value["project"]
    region = value["region"]
    commit = args.commit.lower() if args.commit else resolve_commit(args.repository, args.branch)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("--commit must be a full 40-character Git SHA")
    run_id = args.run_id or run_id_for(args.version, commit)
    bucket_name = release_bucket_name(project, region, args.bucket)
    storage_client, bucket = ensure_bucket(context, project, region, bucket_name)
    _, control_bucket = ensure_control_bucket(context, project, region)
    worker_account = worker_service_account_email(args.service_account, value["projectNumber"])
    ensure_worker_bucket_access(bucket, worker_account)
    ensure_worker_bucket_access(control_bucket, worker_account, "roles/storage.objectViewer")
    existing_kill = get_json(control_bucket, kill_switch_object(plan))
    if existing_kill is None:
        set_kill_switch(control_bucket, plan, False, "controller initialization")
        existing_kill = {"enabled": False}
    legacy_kill = get_json(bucket, kill_switch_object(plan))
    if any(value and value.get("enabled") for value in (existing_kill, legacy_kill)) and not args.reset_kill_switch:
        raise SystemExit("global GCP release kill switch is enabled; pass --reset-kill-switch to explicitly clear it")
    if args.reset_kill_switch:
        set_kill_switch(control_bucket, plan, False, "start --reset-kill-switch")
        set_kill_switch(bucket, plan, False, "start --reset-kill-switch")
    logging_client = context["logging"].Client(project=project, credentials=context["credentials"])
    schedule = value["schedule"]
    for item in schedule:
        item["runId"] = run_id
        item["instanceName"] = resource_name("dl4j", run_id, item["id"])
        item["logId"] = log_id(run_id, item["id"])
        item["planDefaults"] = plan["defaults"]
        item["planProjectLabel"] = plan["projectLabel"]
        item["network"] = value["network"]
    run_manifest = {
        "schemaVersion": 1,
        "provider": "gcp",
        "runId": run_id,
        "project": project,
        "region": region,
        "bucket": bucket_name,
        "controlBucket": control_bucket.name,
        "sourceBranch": args.branch,
        "commit": commit,
        "releaseVersion": args.version,
        "snapshotVersion": args.snapshot_version,
        "createdAt": utc_now(),
        "status": "running",
        "serialLanes": True,
        "completeDl4jMatrix": False,
        "unsupportedWorkflows": plan.get("unsupportedWorkflows", {}),
        "workerServiceAccount": worker_account,
        "executions": schedule,
    }
    run_key = f"{plan['artifactPrefix'].strip('/')}/{run_id}/run.json"
    put_json(bucket, run_key, run_manifest)
    print(json.dumps({
        "event": "run-created", "runId": run_id, "project": project, "region": region, "bucket": bucket_name,
        "sourceBranch": args.branch, "resolvedCommit": commit,
        "logsCommand": f"python3 release/gcp/release.py --region {region} logs --run-id {run_id} --follow",
        "statusCommand": f"python3 release/gcp/release.py --region {region} status --run-id {run_id}",
        "shutdownCommand": f"python3 release/gcp/release.py --region {region} stop-everything --wait",
    }, indent=2))
    active: tuple[str, str] | None = None
    try:
        for item in schedule:
            client = context["compute"].InstancesClient(credentials=context["credentials"])
            launched_item = None
            capacity_errors: list[str] = []
            for alternative in item["launchAlternatives"]:
                for zone in alternative["zones"]:
                    candidate = copy.deepcopy(item)
                    candidate.update({
                        "machineType": alternative["machineType"], "zone": zone,
                        "vcpus": alternative["vcpus"], "memoryGiB": alternative["memoryGiB"],
                    })
                    adapt_build_resources(
                        candidate["shard"], candidate["vcpus"], candidate["memoryGiB"], args.build_threads
                    )
                    candidate["buildThreads"] = candidate["shard"]["build"]["buildThreads"]
                    candidate["mavenHeapGiB"] = candidate["shard"]["build"]["mavenHeapGiB"]
                    shard = candidate["shard"]
                    config = {
                        "provider": "gcp", "project": project, "region": region, "bucket": bucket_name,
                        "artifactPrefix": plan["artifactPrefix"], "runId": run_id, "releaseVersion": args.version,
                        "snapshotVersion": args.snapshot_version, "commit": commit, "repository": args.repository,
                        "killSwitchBucket": control_bucket.name, "killSwitchObject": kill_switch_object(plan),
                        "logId": candidate["logId"], "shard": shard,
                    }
                    startup = render_worker(ROOT / "release/gcp" / shard["worker"], config)
                    resource = instance_resource(context, args, candidate, startup)
                    active = (zone, candidate["instanceName"])
                    emit_controller_event(
                        logging_client, candidate["logId"], run_id, shard["id"], phase="provision",
                        status="attempt", machineType=candidate["machineType"], zone=zone,
                    )
                    try:
                        operation = client.insert(project=project, zone=zone, instance_resource=resource)
                        operation.result(timeout=900)
                    except Exception as exc:
                        cleanup_error = None
                        try:
                            delete_instance(context, project, zone, candidate["instanceName"], wait=True)
                        except Exception as delete_exc:
                            cleanup_error = delete_exc
                        active = None
                        if cleanup_error is not None:
                            raise RuntimeError(
                                f"launch failed and cleanup of {zone}/{candidate['instanceName']} also failed: "
                                f"{cleanup_error}"
                            ) from exc
                        if not retryable_capacity_error(exc):
                            raise
                        capacity_errors.append(f"{zone}/{candidate['machineType']}: {exc}")
                        emit_controller_event(
                            logging_client, candidate["logId"], run_id, shard["id"], phase="provision",
                            status="capacity-retry", machineType=candidate["machineType"], zone=zone,
                        )
                        continue
                    launched_item = candidate
                    break
                if launched_item:
                    break
            if launched_item is None:
                raise RuntimeError(f"all verified zone/machine launch alternatives exhausted: {capacity_errors}")
            item.clear()
            item.update(launched_item)
            run_manifest["executions"] = schedule
            put_json(bucket, run_key, run_manifest)
            shard = item["shard"]
            emit_controller_event(logging_client, item["logId"], run_id, shard["id"], phase="provision", status="complete", instance=item["instanceName"])
            wait_for_lane(context, bucket, plan, item, args.timeout_hours * 3600)
            delete_instance(context, project, item["zone"], item["instanceName"])
            active = None
        run_manifest["status"] = "successful"
        run_manifest["completedAt"] = utc_now()
        put_json(bucket, run_key, run_manifest)
        print(json.dumps({"event": "run-complete", "runId": run_id, "status": "successful"}, indent=2))
    except BaseException:
        run_manifest["status"] = "failed"
        run_manifest["completedAt"] = utc_now()
        put_json(bucket, run_key, run_manifest)
        if active:
            delete_instance(context, project, active[0], active[1], wait=False)
        raise


def load_run(bucket: Any, plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    value = get_json(bucket, f"{plan['artifactPrefix'].strip('/')}/{run_id}/run.json")
    if value is None:
        raise SystemExit(f"run {run_id} was not found in gs://{bucket.name}")
    return value


def context_bucket(args: argparse.Namespace, plan: dict[str, Any]):
    context = cloud_context(args.project, allow_wizard=not getattr(args, "no_wizard", False))
    project = context["project"]
    region = resolve_region(args.region, allow_wizard=not getattr(args, "no_wizard", False))
    name = release_bucket_name(project, region, getattr(args, "bucket", None))
    storage_client = context["storage"].Client(project=project, credentials=context["credentials"])
    bucket = storage_client.lookup_bucket(name)
    if bucket is None:
        raise SystemExit(f"managed release bucket gs://{name} does not exist")
    return context, project, region, bucket


def managed_instances(context: dict[str, Any], project: str) -> list[tuple[str, Any]]:
    client = context["compute"].InstancesClient(credentials=context["credentials"])
    result: list[tuple[str, Any]] = []
    for scope, scoped in client.aggregated_list(project=project):
        zone = scope.rstrip("/").split("/")[-1]
        for instance in (getattr(scoped, "instances", None) or []):
            labels = dict(getattr(instance, "labels", None) or {})
            if labels.get(MANAGED_LABEL) == "true":
                result.append((zone, instance))
    return result


def status(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context, project, region, bucket = context_bucket(args, plan)
    instances = []
    for zone, instance in managed_instances(context, project):
        labels = dict(instance.labels or {})
        if args.run_id and labels.get(RUN_LABEL) != normalize_label(args.run_id):
            continue
        instances.append({"name": instance.name, "zone": zone, "status": str(instance.status), "machineType": str(instance.machine_type).split("/")[-1], "labels": labels})
    run = load_run(bucket, plan, args.run_id) if args.run_id else None
    print(json.dumps({"project": project, "region": region, "run": run, "managedInstances": instances}, indent=2))


def log_names_for_run(run: dict[str, Any], shards: list[str] | None = None) -> list[tuple[str, str]]:
    selected = set(shards or [])
    values = []
    for item in run.get("executions", []):
        shard = item["shard"]["id"]
        if selected and shard not in selected and item["id"] not in selected:
            continue
        values.append((shard, item["logId"]))
    smoke = run.get("tpuSmoke")
    if smoke and (not selected or "tpu-smoke" in selected):
        values.append(("tpu-smoke", smoke["logId"]))
    return values


def show_logs(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context, project, _, bucket = context_bucket(args, plan)
    run = load_run(bucket, plan, args.run_id)
    logging_client = context["logging"].Client(project=project, credentials=context["credentials"])
    seen: dict[str, set[str]] = {}
    while True:
        for shard, name in log_names_for_run(run, args.shard):
            stream_log_entries(logging_client, project, name, args.run_id, shard, seen.setdefault(name, set()))
        if not args.follow:
            return
        time.sleep(5)


def delete_log_material(context: dict[str, Any], project: str, bucket: Any, plan: dict[str, Any], run_ids: list[str]) -> dict[str, Any]:
    logging_client = context["logging"].Client(project=project, credentials=context["credentials"])
    deleted_logs: list[str] = []
    deleted_blobs: list[str] = []
    for run_id in run_ids:
        run = load_run(bucket, plan, run_id)
        for _, name in log_names_for_run(run):
            try:
                logging_client.logger(name).delete()
                deleted_logs.append(name)
            except Exception as exc:
                if not not_found(exc):
                    raise
        prefix = f"{plan['artifactPrefix'].strip('/')}/{run_id}/"
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith(("/build.log", "/tpu-smoke.log")):
                blob.delete()
                deleted_blobs.append(blob.name)
    return {"deletedCloudLogs": deleted_logs, "deletedGcsLogObjects": deleted_blobs}


def delete_logs(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("delete-logs is destructive; pass --yes after reviewing the target")
    plan = load_plan(args.plan)
    context, project, _, bucket = context_bucket(args, plan)
    if args.all_runs:
        prefix = f"{plan['artifactPrefix'].strip('/')}/"
        run_ids = sorted({blob.name[len(prefix):].split("/", 1)[0] for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith("/run.json")})
    else:
        run_ids = [args.run_id]
    result = delete_log_material(context, project, bucket, plan, run_ids)
    print(json.dumps({"bucket": bucket.name, "runIds": run_ids, **result}, indent=2))


def discover_tpu_zones(client: Any, project: str) -> list[str]:
    zones: set[str] = set()
    request = {"name": f"projects/{project}", "page_size": 100}
    while True:
        response = client.list_locations(request=request)
        zones.update(
            str(getattr(location, "location_id", "")) or str(location.name).rstrip("/").split("/")[-1]
            for location in response.locations
            if str(getattr(location, "location_id", "")) or getattr(location, "name", None)
        )
        token = str(getattr(response, "next_page_token", "") or "")
        if not token:
            return sorted(zones)
        request["page_token"] = token


def list_tpu_nodes(
    context: dict[str, Any], project: str, extra_zones: Iterable[str] = ()
) -> tuple[list[Any], list[str]]:
    client = context["tpu"].TpuClient(credentials=context["credentials"])
    result = []
    errors: list[str] = []
    try:
        zones = set(discover_tpu_zones(client, project)) | set(extra_zones)
    except Exception as exc:
        zones = set(extra_zones)
        errors.append(f"location discovery: {exc}")
    for zone in sorted(zones):
        try:
            for node in client.list_nodes(parent=f"projects/{project}/locations/{zone}"):
                if dict(getattr(node, "labels", None) or {}).get(MANAGED_LABEL) == "true":
                    result.append(node)
        except Exception as exc:
            if not not_found(exc):
                errors.append(f"{zone}: {exc}")
    return result, errors


def validate_tpu_configuration(args: argparse.Namespace, plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    settings = plan["tpuSmoke"]
    zone = getattr(args, "tpu_zone", None) or settings["defaultZone"]
    accelerator = getattr(args, "accelerator_type", None) or settings["acceleratorType"]
    runtime = getattr(args, "runtime_version", None) or settings["runtimeVersion"]
    parent = f"projects/{context['project']}/locations/{zone}"
    client = context["tpu"].TpuClient(credentials=context["credentials"])
    accelerator_values = list(client.list_accelerator_types(parent=parent))
    runtime_values = list(client.list_runtime_versions(parent=parent))
    available_accelerators = {str(getattr(item, "type_", "")) or str(item.name).split("/")[-1] for item in accelerator_values}
    available_runtimes = {str(getattr(item, "version", "")) or str(item.name).split("/")[-1] for item in runtime_values}
    if accelerator not in available_accelerators:
        raise RuntimeError(f"Cloud TPU accelerator {accelerator} is not offered in {zone}; available: {sorted(available_accelerators)}")
    if runtime not in available_runtimes:
        raise RuntimeError(f"Cloud TPU runtime {runtime} is not offered in {zone}; available: {sorted(available_runtimes)}")
    result = {"zone": zone, "acceleratorType": accelerator, "runtimeVersion": runtime}
    print(json.dumps({"tpuSmokePreflight": result}, indent=2))
    return result


def tpu_smoke(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context = cloud_context(args.project, allow_wizard=not getattr(args, "no_wizard", False))
    project = context["project"]
    region = resolve_region(args.region, allow_wizard=not getattr(args, "no_wizard", False))
    tpu = validate_tpu_configuration(args, plan, context)
    network = resolve_network(
        context["compute"], context["credentials"], project, region, args.network, args.subnetwork
    )
    commit = args.commit.lower() if args.commit else resolve_commit(args.repository, args.branch)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit("--commit must be a full 40-character Git SHA")
    run_id = args.run_id or run_id_for("tpu-smoke", commit)
    bucket_name = release_bucket_name(project, region, args.bucket)
    _, bucket = ensure_bucket(context, project, region, bucket_name)
    _, control_bucket = ensure_control_bucket(context, project, region)
    projects_client = context["compute"].ProjectsClient(credentials=context["credentials"])
    project_number = str(projects_client.get(project=project).id)
    worker_account = worker_service_account_email(args.service_account, project_number)
    ensure_worker_bucket_access(bucket, worker_account)
    ensure_worker_bucket_access(control_bucket, worker_account, "roles/storage.objectViewer")
    existing = get_json(control_bucket, kill_switch_object(plan))
    if existing is None:
        set_kill_switch(control_bucket, plan, False, "controller initialization")
        existing = {"enabled": False}
    legacy = get_json(bucket, kill_switch_object(plan))
    if any(value and value.get("enabled") for value in (existing, legacy)) and not args.reset_kill_switch:
        raise SystemExit("global GCP release kill switch is enabled; pass --reset-kill-switch")
    if args.reset_kill_switch:
        set_kill_switch(control_bucket, plan, False, "tpu-smoke --reset-kill-switch")
        set_kill_switch(bucket, plan, False, "tpu-smoke --reset-kill-switch")
    name = resource_name("dl4j-tpu", run_id, "smoke")
    log_name = log_id(run_id, "tpu-smoke")
    config = {
        "provider": "gcp", "project": project, "region": region, "bucket": bucket_name,
        "artifactPrefix": plan["artifactPrefix"], "runId": run_id, "commit": commit,
        "repository": args.repository, "killSwitchBucket": control_bucket.name,
        "killSwitchObject": kill_switch_object(plan), "logId": log_name,
    }
    startup = render_worker(ROOT / "release/gcp/tpu-worker.sh", config, tpu=True)
    types = context["tpu"]
    node_kwargs: dict[str, Any] = {
        "accelerator_type": tpu["acceleratorType"],
        "runtime_version": tpu["runtimeVersion"],
        "metadata": {"startup-script": startup},
        "labels": {MANAGED_LABEL: "true", PROJECT_LABEL: normalize_label(plan["projectLabel"]), RUN_LABEL: normalize_label(run_id), SHARD_LABEL: "tpu-smoke"},
        "network_config": types.NetworkConfig(
            enable_external_ips=True, network=network["network"], subnetwork=network["subnetwork"] or ""
        ),
        "scheduling_config": types.SchedulingConfig(preemptible=args.spot),
    }
    node_kwargs["service_account"] = types.ServiceAccount(email=worker_account, scope=SCOPES)
    node = types.Node(**node_kwargs)
    client = types.TpuClient(credentials=context["credentials"])
    parent = f"projects/{project}/locations/{tpu['zone']}"
    run_key = f"{plan['artifactPrefix'].strip('/')}/{run_id}/run.json"
    run = get_json(bucket, run_key) or {
        "schemaVersion": 1, "provider": "gcp", "runId": run_id, "project": project, "region": region,
        "bucket": bucket_name, "commit": commit, "createdAt": utc_now(), "executions": [], "status": "tpu-smoke-running",
    }
    run["controlBucket"] = control_bucket.name
    run["workerServiceAccount"] = worker_account
    run["tpuSmoke"] = {**tpu, "name": name, "logId": log_name, "status": "running"}
    put_json(bucket, run_key, run)
    print(json.dumps({"event": "tpu-smoke-created", "runId": run_id, "name": name, **tpu,
                      "shutdownCommand": f"python3 release/gcp/release.py --region {region} stop-everything --wait"}, indent=2))
    try:
        operation = client.create_node(parent=parent, node=node, node_id=name)
        operation.result(timeout=1800)
        fake_item = {"runId": run_id, "shard": {"id": "tpu-smoke"}, "logId": log_name, "zone": tpu["zone"], "instanceName": name}
        deadline = time.monotonic() + args.timeout_hours * 3600
        status_name = f"{plan['artifactPrefix'].strip('/')}/{run_id}/tpu-smoke/status.json"
        logging_client = context["logging"].Client(project=project, credentials=context["credentials"])
        seen: set[str] = set()
        while time.monotonic() < deadline:
            status_value = get_json(bucket, status_name)
            stream_log_entries(logging_client, project, log_name, run_id, "tpu-smoke", seen)
            if status_value is not None:
                if int(status_value.get("exitCode", 1)) != 0:
                    raise RuntimeError(f"TPU smoke failed: {status_value}")
                run["tpuSmoke"]["status"] = "successful"
                run["status"] = "tpu-smoke-successful"
                put_json(bucket, run_key, run)
                print(json.dumps(status_value, indent=2))
                return
            time.sleep(10)
        raise TimeoutError("Cloud TPU smoke test timed out")
    except BaseException as exc:
        state = "timed-out" if isinstance(exc, TimeoutError) else "failed"
        run["tpuSmoke"].update({"status": state, "error": f"{exc.__class__.__name__}: {exc}"})
        run["status"] = f"tpu-smoke-{state}"
        run["completedAt"] = utc_now()
        try:
            put_json(bucket, run_key, run)
        except Exception as state_exc:
            print(f"failed to persist TPU smoke failure state: {state_exc}", file=sys.stderr)
        raise
    finally:
        try:
            client.delete_node(name=f"{parent}/nodes/{name}").result(timeout=1800)
        except Exception as exc:
            if not not_found(exc):
                print(f"TPU node cleanup failed; run stop-everything: {exc}", file=sys.stderr)


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        stream.extractall(destination)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context, project, region, bucket = context_bucket(args, plan)
    run = load_run(bucket, plan, args.run_id)
    if run.get("commit") != args.commit or run.get("releaseVersion") != args.version:
        raise RuntimeError("collect identity does not match run.json")
    executions = run.get("executions", [])
    if args.shard:
        selected = set(args.shard)
        executions = [item for item in executions if item["id"] in selected or item["shard"]["id"] in selected]
    if not executions:
        raise RuntimeError("no selected execution outputs")
    prefix = f"{plan['artifactPrefix'].strip('/')}/{args.run_id}"
    with tempfile.TemporaryDirectory(prefix="dl4j-gcp-collect-") as temp:
        directory = Path(temp)
        existing_manifest = None
        release_exists = False
        if not args.no_github:
            release_exists = subprocess.run(
                ["gh", "release", "view", args.release_tag, "--repo", args.github_repository],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
            if release_exists:
                existing_directory = directory / "existing-release"
                existing_directory.mkdir()
                download = subprocess.run(
                    ["gh", "release", "download", args.release_tag, "--repo", args.github_repository,
                     "--pattern", "release-build-manifest.json", "--dir", str(existing_directory), "--clobber"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                existing_path = existing_directory / "release-build-manifest.json"
                if download.returncode == 0 and existing_path.exists():
                    existing_manifest = json.loads(existing_path.read_text(encoding="utf-8"))
                    if (
                        existing_manifest.get("releaseVersion") != args.version
                        or existing_manifest.get("commit") != args.commit
                        or existing_manifest.get("releaseTag") != args.release_tag
                    ):
                        raise RuntimeError("existing GitHub release manifest has a different immutable identity")
        assets: list[dict[str, Any]] = []
        maven_archives: list[Path] = []
        for item in executions:
            shard = item["shard"]["id"]
            status_value = get_json(bucket, f"{prefix}/{shard}/status.json")
            if not status_value or int(status_value.get("exitCode", 1)) != 0:
                raise RuntimeError(f"shard {shard} is incomplete or failed: {status_value}")
            manifest_value = get_json(bucket, f"{prefix}/{shard}/shard-manifest.json")
            if not manifest_value or manifest_value.get("commit") != args.commit or manifest_value.get("releaseVersion") != args.version:
                raise RuntimeError(f"shard {shard} manifest identity mismatch")
            for workload in item["shard"]["workloads"]:
                source_name = "maven-repository.tar.gz" if workload == "maven" else "sdk-assets.tar.gz"
                blob = bucket.blob(f"{prefix}/{shard}/{source_name}")
                output_name = f"{source_name.removesuffix('.tar.gz')}-{shard}.tar.gz"
                output = directory / output_name
                blob.download_to_filename(str(output))
                if workload == "maven":
                    maven_archives.append(output)
                assets.append({"fileName": output_name, "sha256": file_digest(output), "size": output.stat().st_size, "shard": shard, "sourceObject": blob.name})
            manifest_path = directory / f"{shard}-shard-manifest.json"
            manifest_path.write_text(json.dumps(manifest_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            assets.append({"fileName": manifest_path.name, "sha256": file_digest(manifest_path), "size": manifest_path.stat().st_size, "shard": shard, "sourceObject": f"{prefix}/{shard}/shard-manifest.json"})
        repository_dir = directory / "gcs-maven-repository"
        repository_manifest = directory / "gcs-maven-repository-manifest.json"
        central_tool = ROOT / "release/central/repository.py"
        command = [sys.executable, str(central_tool), "materialize-test-repository", "--output", str(repository_dir), "--manifest", str(repository_manifest), "--release-version", args.version, "--commit", args.commit]
        for archive in sorted(maven_archives):
            command.extend(["--input", str(archive)])
        subprocess.run(command, check=True)
        repository_prefix = f"{prefix}/maven2"
        for path in sorted(repository_dir.rglob("*")):
            if path.is_file():
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                bucket.blob(f"{repository_prefix}/{path.relative_to(repository_dir).as_posix()}").upload_from_filename(str(path), content_type=content_type)
        bucket.blob(f"{repository_prefix}/.dl4j/manifest.json").upload_from_filename(str(repository_manifest), content_type="application/json")
        current_shards = sorted(item["shard"]["id"] for item in executions)
        aws_plan = json.loads((ROOT / "release/aws/release-plan.json").read_text(encoding="utf-8"))
        expected_matrix = matrix_coverage(aws_plan, [item["id"] for item in aws_plan["shards"]])
        current_matrix = matrix_coverage(aws_plan, current_shards)
        current_missing_matrix = sorted(expected_matrix - current_matrix)
        put_json(bucket, f"{repository_prefix}/.dl4j/complete.json", {
            "schemaVersion": 1, "layout": "maven2", "ready": True, "provider": "gcp", "runId": args.run_id,
            "releaseVersion": args.version, "commit": args.commit, "gcpMatrixComplete": len(executions) == len(run.get("executions", [])),
            "completeMatrix": not current_missing_matrix, "missingMatrixEntries": current_missing_matrix,
            "missingWorkflows": sorted(plan.get("unsupportedWorkflows", {})) if any(value.startswith("macos-14-arm64-cpu--") for value in current_missing_matrix) else [],
            "manifestSha256": file_digest(repository_manifest),
        })
        combined_shards = sorted(set(current_shards) | set((existing_manifest or {}).get("shards", [])))
        covered_matrix = matrix_coverage(aws_plan, combined_shards)
        missing_matrix = sorted(expected_matrix - covered_matrix)
        combined_assets = {item["fileName"]: item for item in (existing_manifest or {}).get("assets", [])}
        combined_assets.update({item["fileName"]: item for item in assets})
        current_workloads = {
            workload
            for item in executions
            for workload in item["shard"].get("workloads", [])
        }
        combined_workloads = sorted(current_workloads | set((existing_manifest or {}).get("workloads", [])))
        provider = merged_release_provider(existing_manifest)
        manifest = {
            "schemaVersion": 1, "provider": provider, "runId": args.run_id, "releaseTag": args.release_tag,
            "releaseVersion": args.version, "commit": args.commit, "project": project, "region": region,
            "bucket": bucket.name, "workloads": combined_workloads, "shards": combined_shards,
            "completeMatrix": not missing_matrix,
            "missingMatrixEntries": missing_matrix,
            "missingWorkflows": sorted(plan.get("unsupportedWorkflows", {})) if any(value.startswith("macos-14-arm64-cpu--") for value in missing_matrix) else [],
            "testMavenRepository": {
                "uri": f"gs://{bucket.name}/{repository_prefix}", "layout": "maven2", "ready": True,
                "completeMatrix": not current_missing_matrix, "missingMatrixEntries": current_missing_matrix,
            },
            "assets": sorted(combined_assets.values(), key=lambda value: value["fileName"]),
        }
        manifest_path = directory / "release-build-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksum = directory / "release-build-manifest.json.sha256"
        checksum.write_text(f"{file_digest(manifest_path)}  {manifest_path.name}\n", encoding="ascii")
        if not args.no_github:
            if not release_exists:
                subprocess.run(["gh", "release", "create", args.release_tag, "--repo", args.github_repository, "--target", args.commit, "--draft", "--title", f"DL4J {args.version} external build"], check=True)
            files = [str(directory / item["fileName"]) for item in assets] + [str(manifest_path), str(checksum)]
            subprocess.run(["gh", "release", "upload", args.release_tag, "--repo", args.github_repository, "--clobber", *files], check=True)
        print(json.dumps(manifest, indent=2))


def stop_everything(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    context = cloud_context(args.project, allow_wizard=not getattr(args, "no_wizard", False))
    project = context["project"]
    region = resolve_region(args.region, allow_wizard=not getattr(args, "no_wizard", False))
    bucket_name = release_bucket_name(project, region, args.bucket)
    shutdown_errors: list[str] = []
    try:
        storage_client = context["storage"].Client(project=project, credentials=context["credentials"])
    except Exception as exc:
        storage_client = None
        shutdown_errors.append(f"Cloud Storage client: {exc}")
    managed_buckets: dict[str, Any] = {}
    kill_switch_set = False

    # Signal the project-wide switch first, but never let a Storage failure prevent
    # direct Compute Engine and Cloud TPU deletion below.
    try:
        _, control_bucket = ensure_control_bucket(context, project, region)
        set_kill_switch(control_bucket, plan, True, "stop-everything")
        managed_buckets[control_bucket.name] = control_bucket
        kill_switch_set = True
    except Exception as exc:
        shutdown_errors.append(f"control kill switch: {exc}")

    deleted_instances: list[str] = []
    operations: list[tuple[str, Any]] = []
    try:
        instances = managed_instances(context, project)
    except Exception as exc:
        instances = []
        shutdown_errors.append(f"Compute Engine discovery: {exc}")
    try:
        compute_client = context["compute"].InstancesClient(credentials=context["credentials"])
    except Exception as exc:
        compute_client = None
        shutdown_errors.append(f"Compute Engine client: {exc}")
    for zone, instance in instances:
        target = f"{zone}/{instance.name}"
        if compute_client is None:
            shutdown_errors.append(f"Compute Engine delete {target}: client unavailable")
            continue
        try:
            operations.append((target, compute_client.delete(project=project, zone=zone, instance=instance.name)))
            deleted_instances.append(target)
        except Exception as exc:
            if not not_found(exc):
                shutdown_errors.append(f"Compute Engine delete {target}: {exc}")

    tpu_zones = set(plan["tpuSmoke"]["zones"])
    if args.tpu_zone:
        tpu_zones.add(args.tpu_zone)
    try:
        nodes, tpu_scan_errors = list_tpu_nodes(context, project, tpu_zones)
    except Exception as exc:
        nodes, tpu_scan_errors = [], [f"Cloud TPU discovery: {exc}"]
    try:
        tpu_client = context["tpu"].TpuClient(credentials=context["credentials"])
    except Exception as exc:
        tpu_client = None
        shutdown_errors.append(f"Cloud TPU client: {exc}")
    deleted_tpus: list[str] = []
    for node in nodes:
        if tpu_client is None:
            shutdown_errors.append(f"Cloud TPU delete {node.name}: client unavailable")
            continue
        try:
            operations.append((node.name, tpu_client.delete_node(name=node.name)))
            deleted_tpus.append(node.name)
        except Exception as exc:
            if not not_found(exc):
                shutdown_errors.append(f"Cloud TPU delete {node.name}: {exc}")

    # Signal compatibility switches in every existing regional bucket after
    # deletion requests are in flight. This also discovers buckets for purging.
    discovered_buckets: dict[str, Any] = {}
    if storage_client is not None:
        try:
            primary = storage_client.lookup_bucket(bucket_name)
            if primary is not None and is_managed_bucket(primary):
                discovered_buckets[primary.name] = primary
        except Exception as exc:
            shutdown_errors.append(f"primary bucket lookup {bucket_name}: {exc}")
    managed_prefix = release_bucket_name(project, "", None).rstrip("-") + "-"
    if storage_client is not None:
        try:
            for candidate in storage_client.list_buckets(project=project, prefix=managed_prefix):
                if is_managed_bucket(candidate):
                    discovered_buckets[candidate.name] = candidate
        except Exception as exc:
            shutdown_errors.append(f"managed bucket discovery: {exc}")
    for name, candidate in sorted(discovered_buckets.items()):
        if name in managed_buckets:
            continue
        try:
            set_kill_switch(candidate, plan, True, "stop-everything")
            managed_buckets[name] = candidate
            if name == control_bucket_name(project):
                kill_switch_set = True
        except Exception as exc:
            shutdown_errors.append(f"kill switch gs://{name}: {exc}")

    if args.wait:
        for target, operation in operations:
            try:
                operation.result(timeout=1800)
            except Exception as exc:
                if not not_found(exc):
                    shutdown_errors.append(f"wait for deletion {target}: {exc}")

    deleted_logs = {"deletedCloudLogs": [], "deletedGcsLogObjects": []}
    if args.purge_logs:
        prefix = f"{plan['artifactPrefix'].strip('/')}/"
        for managed_bucket in managed_buckets.values():
            try:
                run_ids = sorted({
                    blob.name[len(prefix):].split("/", 1)[0]
                    for blob in managed_bucket.list_blobs(prefix=prefix)
                    if blob.name.endswith("/run.json")
                })
                removed = delete_log_material(context, project, managed_bucket, plan, run_ids)
                deleted_logs["deletedCloudLogs"].extend(removed["deletedCloudLogs"])
                deleted_logs["deletedGcsLogObjects"].extend(removed["deletedGcsLogObjects"])
            except Exception as exc:
                shutdown_errors.append(f"purge logs gs://{managed_bucket.name}: {exc}")
    purged: list[str] = []
    if args.purge_storage:
        control_object = kill_switch_object(plan)
        for managed_bucket in managed_buckets.values():
            try:
                blobs = list(managed_bucket.list_blobs())
            except Exception as exc:
                shutdown_errors.append(f"list purge objects gs://{managed_bucket.name}: {exc}")
                continue
            for blob in blobs:
                if blob.name == control_object:
                    continue
                try:
                    blob.delete()
                    purged.append(f"{managed_bucket.name}/{blob.name}")
                except Exception as exc:
                    shutdown_errors.append(f"purge gs://{managed_bucket.name}/{blob.name}: {exc}")
            try:
                set_kill_switch(managed_bucket, plan, True, "stop-everything --purge-storage")
            except Exception as exc:
                shutdown_errors.append(f"restore kill switch gs://{managed_bucket.name}: {exc}")
    print(json.dumps({
        "project": project, "region": region, "killSwitch": kill_switch_set,
        "killSwitchBuckets": sorted(managed_buckets),
        "deletedInstances": deleted_instances, "deletedTpuNodes": deleted_tpus,
        "tpuScanErrors": tpu_scan_errors, "shutdownErrors": shutdown_errors,
        "storagePurged": args.purge_storage, "deletedStorageObjects": len(purged), **deleted_logs,
    }, indent=2))
    if shutdown_errors or tpu_scan_errors:
        raise RuntimeError("emergency shutdown completed best-effort but could not verify every target; see errors above")


def add_selection_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--shard", action="append", help="lane or lane--variant; repeatable")
    command.add_argument("--exclude-shard", action="append", help="exclude a lane or lane--variant; repeatable")
    command.add_argument("--machine-type", help="override candidate list; still verified in the selected zone")
    command.add_argument("--build-threads", type=int)
    command.add_argument("--max-cores", type=int, help="greedily choose the largest verified machine no larger than this vCPU limit")
    command.add_argument("--zone", help="force one Compute Engine zone; otherwise each lane chooses a verified zone in the region")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    result.add_argument("--project", help="defaults to GOOGLE_CLOUD_PROJECT/ADC project")
    result.add_argument("--region", help="defaults to GOOGLE_CLOUD_REGION or CLOUDSDK_COMPUTE_REGION")
    result.add_argument(
        "--no-wizard", action="store_true",
        help="fail instead of prompting when ADC, project, or region configuration is incomplete",
    )
    sub = result.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("configure", help="validate or interactively complete Google Cloud configuration")
    setup.set_defaults(func=configure_environment)

    check = sub.add_parser("preflight")
    add_selection_options(check)
    check.add_argument("--include-tpu-smoke", action="store_true")
    check.add_argument("--tpu-zone")
    check.add_argument("--accelerator-type")
    check.add_argument("--runtime-version")
    check.add_argument("--network")
    check.add_argument("--subnetwork")
    check.set_defaults(func=preflight)

    launch = sub.add_parser("start")
    launch.add_argument("--version", required=True)
    launch.add_argument("--snapshot-version", default="1.0.0-SNAPSHOT")
    source = launch.add_mutually_exclusive_group(required=True)
    source.add_argument("--commit")
    source.add_argument("--branch")
    launch.add_argument("--repository", default=DEFAULT_REPOSITORY)
    launch.add_argument("--run-id")
    launch.add_argument("--bucket")
    launch.add_argument("--network")
    launch.add_argument("--subnetwork")
    launch.add_argument("--service-account")
    launch.add_argument("--root-volume-gib", type=int)
    launch.add_argument("--timeout-hours", type=int, default=12)
    launch.add_argument("--reset-kill-switch", action="store_true")
    add_selection_options(launch)
    launch.set_defaults(func=start)

    show = sub.add_parser("status")
    show.add_argument("--run-id")
    show.add_argument("--bucket")
    show.set_defaults(func=status)

    logs = sub.add_parser("logs")
    logs.add_argument("--run-id", required=True)
    logs.add_argument("--bucket")
    logs.add_argument("--shard", action="append")
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=show_logs)

    remove = sub.add_parser("delete-logs")
    target = remove.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--all-runs", action="store_true")
    remove.add_argument("--bucket")
    remove.add_argument("--yes", action="store_true")
    remove.set_defaults(func=delete_logs)

    gather = sub.add_parser("collect")
    gather.add_argument("--run-id", required=True)
    gather.add_argument("--bucket")
    gather.add_argument("--release-tag", required=True)
    gather.add_argument("--version", required=True)
    gather.add_argument("--commit", required=True)
    gather.add_argument("--github-repository", default="deeplearning4j/deeplearning4j")
    gather.add_argument("--shard", action="append")
    gather.add_argument("--no-github", action="store_true")
    gather.set_defaults(func=collect)

    smoke = sub.add_parser("tpu-smoke")
    source = smoke.add_mutually_exclusive_group(required=True)
    source.add_argument("--commit")
    source.add_argument("--branch")
    smoke.add_argument("--repository", default=DEFAULT_REPOSITORY)
    smoke.add_argument("--run-id")
    smoke.add_argument("--bucket")
    smoke.add_argument("--tpu-zone")
    smoke.add_argument("--accelerator-type")
    smoke.add_argument("--runtime-version")
    smoke.add_argument("--service-account")
    smoke.add_argument("--network")
    smoke.add_argument("--subnetwork")
    smoke.add_argument("--spot", action="store_true")
    smoke.add_argument("--timeout-hours", type=int, default=2)
    smoke.add_argument("--reset-kill-switch", action="store_true")
    smoke.set_defaults(func=tpu_smoke)

    stop = sub.add_parser("stop-everything")
    stop.add_argument("--bucket")
    stop.add_argument("--tpu-zone")
    stop.add_argument("--wait", action="store_true")
    stop.add_argument("--purge-storage", action="store_true")
    stop.add_argument("--purge-logs", action="store_true")
    stop.set_defaults(func=stop_everything)
    return result


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "max_cores") and args.max_cores is not None and args.max_cores < 1:
        raise SystemExit("--max-cores must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
