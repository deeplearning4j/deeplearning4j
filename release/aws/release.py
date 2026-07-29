#!/usr/bin/env python3
"""Provision and stop ephemeral DL4J release builders using the standard AWS SDK chain."""

from __future__ import annotations

import argparse
import copy
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

MANAGED_TAG = "DL4JReleaseManaged"
PROJECT_TAG = "DL4JReleaseProject"
RUN_TAG = "DL4JReleaseRun"
SHARD_TAG = "DL4JReleaseShard"
ACTIVE_STATES = ("pending", "running", "stopping", "stopped")
GPU_INSTANCE_PREFIXES = ("g", "p", "inf", "trn")


def _boto3():
    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except ImportError as exc:
        raise SystemExit("boto3 is required: python3 -m pip install boto3") from exc
    return boto3, ClientError


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schemaVersion") != 2:
        raise ValueError("unsupported release-plan schemaVersion")
    ids: set[str] = set()
    defaults = plan.get("defaults", {})
    for shard in plan.get("shards", []):
        shard_id = shard.get("id")
        if not shard_id or shard_id in ids:
            raise ValueError(f"missing or duplicate shard id: {shard_id!r}")
        ids.add(shard_id)
        workloads = set(shard.get("workloads", []))
        if not workloads or not workloads <= {"maven", "sdk"}:
            raise ValueError(f"invalid workloads for {shard_id}: {sorted(workloads)}")
        instance_type = shard.get("instanceType", defaults.get("instanceType", ""))
        family = instance_type.split(".", 1)[0]
        if family.startswith(GPU_INSTANCE_PREFIXES):
            raise ValueError(f"GPU/accelerator instance type is forbidden for compile-only shard {shard_id}: {instance_type}")
        if shard.get("os") not in {"linux", "windows", "macos"}:
            raise ValueError(f"invalid operating system for {shard_id}: {shard.get('os')}")
        query = shard.get("amiQuery")
        if not isinstance(query, dict) or not query.get("owners") or not query.get("name") or not query.get("architecture"):
            raise ValueError(f"shard {shard_id} must define a complete amiQuery")
        if shard.get("amiSsmParameter") and not query:
            raise ValueError(f"shard {shard_id} cannot trust an SSM AMI without independent query criteria")
        if not shard.get("worker") or not shard.get("build", {}).get("variants"):
            raise ValueError(f"shard {shard_id} has no worker or build variants")
        if shard.get("os") == "macos" and not shard.get("dedicatedHost"):
            raise ValueError(f"macOS shard {shard_id} must use an EC2 Mac dedicated host")
    if not ids:
        raise ValueError("release plan contains no shards")
    return plan


def execution_shards(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create reusable platform/toolchain lanes; variants run serially on one host."""
    executions: list[dict[str, Any]] = []
    for original in plan["shards"]:
        shard = copy.deepcopy(original)
        shard["parentShard"] = original["id"]
        if shard["build"].get("prebuildCrossPlatform") and shard["build"].get("javacppPlatform") == "linux-x86_64":
            shard["artifactRules"] = {**original["artifactRules"], "mode": "all"}
        executions.append(shard)
    return executions


def resolve_branch(repository: str, branch: str) -> str:
    """Resolve a remote branch once so every matrix worker builds the same commit."""
    if not branch or branch.startswith("-") or any(character.isspace() for character in branch):
        raise SystemExit(f"Invalid branch name: {branch!r}")
    validation = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        text=True, capture_output=True, check=False,
    )
    if validation.returncode != 0:
        raise SystemExit(f"Invalid branch name {branch!r}: {validation.stderr.strip()}")
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", repository, f"refs/heads/{branch}"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or "branch was not found"
        raise SystemExit(f"Unable to resolve branch {branch!r} from {repository}: {detail}")
    matches = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(matches) != 1 or len(matches[0]) != 2 or matches[0][1] != f"refs/heads/{branch}":
        raise SystemExit(f"Remote returned an ambiguous result for branch {branch!r}: {result.stdout.strip()}")
    commit = matches[0][0].lower()
    if len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit):
        raise SystemExit(f"Remote returned an invalid commit for branch {branch!r}: {commit!r}")
    return commit


def session_clients(region: str | None = None):
    boto3, _ = _boto3()
    requested_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    session = boto3.Session(region_name=requested_region)
    resolved_region = session.region_name
    if not resolved_region:
        raise SystemExit("Set AWS_REGION or AWS_DEFAULT_REGION")
    return session, resolved_region, session.client("ec2"), session.client("ssm"), session.client("s3"), session.client("sts"), session.client("iam")


def kill_parameter_name(plan: dict[str, Any]) -> str:
    return str(plan.get("killSwitchParameter", "/deeplearning4j/release/kill-switch"))


def log_group_name(plan: dict[str, Any]) -> str:
    return str(plan.get("logGroupName", "/deeplearning4j/releases"))


def ensure_log_group(logs_client, plan: dict[str, Any]) -> str:
    _, ClientError = _boto3()
    group = log_group_name(plan)
    try:
        logs_client.create_log_group(
            logGroupName=group,
            tags={MANAGED_TAG: "true", PROJECT_TAG: str(plan.get("projectTag", "deeplearning4j-release"))},
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise
    logs_client.put_retention_policy(
        logGroupName=group,
        retentionInDays=int(plan.get("logRetentionDays", 30)),
    )
    return group


def ensure_log_stream(logs_client, group: str, stream: str) -> None:
    _, ClientError = _boto3()
    try:
        logs_client.create_log_stream(logGroupName=group, logStreamName=stream)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceAlreadyExistsException":
            raise


def emit_cloudwatch_event(logs_client, group: str, stream: str, message: str) -> bool:
    """Publish a controller lifecycle event without depending on worker bootstrap."""
    try:
        logs_client.put_log_events(
            logGroupName=group,
            logStreamName=stream,
            logEvents=[{"timestamp": int(time.time() * 1000), "message": f"[dl4j-controller] {message}"}],
        )
        return True
    except Exception as exc:
        print(f"[{stream}] controller CloudWatch publish failed: {exc}", flush=True)
        return False


def list_log_streams(logs_client, group: str, prefix: str | None = None) -> list[str]:
    _, ClientError = _boto3()
    request: dict[str, Any] = {"logGroupName": group}
    if prefix:
        request["logStreamNamePrefix"] = prefix
    streams: list[str] = []
    while True:
        try:
            response = logs_client.describe_log_streams(**request)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return []
            raise
        streams.extend(item["logStreamName"] for item in response.get("logStreams", []))
        token = response.get("nextToken")
        if not token:
            return streams
        request["nextToken"] = token


def delete_log_streams(logs_client, group: str, prefix: str | None = None, selected: set[str] | None = None) -> list[str]:
    deleted = []
    for stream in list_log_streams(logs_client, group, prefix):
        if selected and stream.rsplit("/", 1)[-1] not in selected:
            continue
        logs_client.delete_log_stream(logGroupName=group, logStreamName=stream)
        deleted.append(stream)
    return deleted


def set_kill_switch(ssm, plan: dict[str, Any], enabled: bool) -> None:
    ssm.put_parameter(
        Name=kill_parameter_name(plan),
        Description="Global emergency stop for all managed DL4J release builders",
        Type="String",
        Value="true" if enabled else "false",
        Overwrite=True,
    )
    ssm.add_tags_to_resource(
        ResourceType="Parameter",
        ResourceId=kill_parameter_name(plan),
        Tags=[{"Key": "DL4JReleaseKillSwitch", "Value": "true"}],
    )


def kill_switch_enabled(ssm, plan: dict[str, Any]) -> bool:
    _, ClientError = _boto3()
    try:
        result = ssm.get_parameter(Name=kill_parameter_name(plan))
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ParameterNotFound":
            return False
        raise
    return result["Parameter"]["Value"].strip().lower() == "true"


def default_network(ec2, instance_types: list[str] | None = None) -> tuple[str, str, str]:
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        raise SystemExit("No default VPC found; pass --subnet-id and --security-group-id")
    vpc_id = vpcs[0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    if not subnets:
        raise SystemExit(f"Default VPC {vpc_id} has no subnets")
    if instance_types:
        subnet_zones = sorted({item["AvailabilityZone"] for item in subnets})
        response = ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": sorted(set(instance_types))},
                     {"Name": "location", "Values": subnet_zones}],
        )
        zones_by_type = {instance_type: set() for instance_type in instance_types}
        for offering in response["InstanceTypeOfferings"]:
            zones_by_type[offering["InstanceType"]].add(offering["Location"])
        common_zones = set(subnet_zones)
        for zones in zones_by_type.values():
            common_zones &= zones
        subnets = [item for item in subnets if item["AvailabilityZone"] in common_zones]
        if not subnets:
            details = {key: sorted(value) for key, value in zones_by_type.items()}
            raise SystemExit(f"Default VPC has no subnet AZ common to selected instance types: {details}")
    subnets.sort(key=lambda item: item.get("AvailableIpAddressCount", 0), reverse=True)
    groups = ec2.describe_security_groups(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "group-name", "Values": ["default"]}]
    )["SecurityGroups"]
    return subnets[0]["SubnetId"], groups[0]["GroupId"], subnets[0]["AvailabilityZone"]


def ensure_bucket(s3, sts, region: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    account = sts.get_caller_identity()["Account"]
    bucket = f"dl4j-release-{account}-{region}".lower()
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        request: dict[str, Any] = {"Bucket": bucket}
        if region != "us-east-1":
            request["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**request)
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]},
        )
    return bucket


def ensure_instance_profile(iam, bucket: str, parameter: str, log_group: str) -> str:
    _, ClientError = _boto3()
    role = "DL4JReleaseBuilderRole"
    profile = "DL4JReleaseBuilderProfile"
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
    try:
        iam.get_role(RoleName=role)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust), Description="Ephemeral DL4J release builders")
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:PutObject", "s3:AbortMultipartUpload", "s3:ListBucket"], "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
            {"Effect": "Allow", "Action": "ssm:GetParameter", "Resource": "*", "Condition": {"StringEquals": {"ssm:ResourceTag/DL4JReleaseKillSwitch": "true"}}},
            {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": f"arn:aws:logs:*:*:log-group:{log_group}:log-stream:*"},
        ],
    }
    iam.put_role_policy(RoleName=role, PolicyName="DL4JReleaseBuilder", PolicyDocument=json.dumps(policy))
    profile_data = None
    try:
        profile_data = iam.get_instance_profile(InstanceProfileName=profile)["InstanceProfile"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise
        iam.create_instance_profile(InstanceProfileName=profile)
    attached = {item["RoleName"] for item in (profile_data or {}).get("Roles", [])}
    if role not in attached:
        iam.add_role_to_instance_profile(InstanceProfileName=profile, RoleName=role)
        time.sleep(10)
    return profile


def _validate_ami(shard: dict[str, Any], query: dict[str, Any], image: dict[str, Any]) -> None:
    expected_architecture = query["architecture"]
    if image.get("Architecture") != expected_architecture:
        raise RuntimeError(
            f"AMI {image.get('ImageId')} for {shard['id']} has architecture "
            f"{image.get('Architecture')!r}, expected {expected_architecture!r}"
        )
    if image.get("State") != "available":
        raise RuntimeError(f"AMI {image.get('ImageId')} for {shard['id']} is not available")
    if image.get("RootDeviceType") != query.get("rootDeviceType", "ebs"):
        raise RuntimeError(f"AMI {image.get('ImageId')} for {shard['id']} has an unexpected root device type")
    if image.get("VirtualizationType") != query.get("virtualizationType", "hvm"):
        raise RuntimeError(f"AMI {image.get('ImageId')} for {shard['id']} has an unexpected virtualization type")
    if query.get("platform") and image.get("Platform") != query["platform"]:
        raise RuntimeError(f"AMI {image.get('ImageId')} for {shard['id']} has an unexpected platform")
    owner_ids = set(query.get("ownerIds", []))
    if owner_ids and image.get("OwnerId") not in owner_ids:
        raise RuntimeError(
            f"AMI {image.get('ImageId')} for {shard['id']} is owned by "
            f"{image.get('OwnerId')}, expected one of {sorted(owner_ids)}"
        )


def resolve_ami(ec2, ssm, shard: dict[str, Any]) -> str:
    """Resolve and independently verify an AMI; never trust a parameter value alone."""
    query = shard.get("amiQuery")
    parameter = shard.get("amiSsmParameter")
    if not query:
        raise ValueError(f"shard {shard['id']} must define amiQuery verification criteria")
    images: list[dict[str, Any]]
    if parameter:
        image_id = ssm.get_parameter(Name=parameter)["Parameter"]["Value"]
        images = ec2.describe_images(ImageIds=[image_id])["Images"]
    else:
        filters = [
            {"Name": "name", "Values": [query["name"]]},
            {"Name": "architecture", "Values": [query["architecture"]]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "root-device-type", "Values": [query.get("rootDeviceType", "ebs")]},
            {"Name": "virtualization-type", "Values": [query.get("virtualizationType", "hvm")]},
        ]
        if query.get("platform"):
            filters.append({"Name": "platform", "Values": [query["platform"]]})
        images = ec2.describe_images(Owners=query["owners"], Filters=filters)["Images"]
    if not images:
        raise RuntimeError(f"no AMI matched verified query for shard {shard['id']}: {query}")
    image = max(images, key=lambda item: item["CreationDate"])
    _validate_ami(shard, query, image)
    return image["ImageId"]


def selected_executions(plan: dict[str, Any], selected_ids: list[str] | None) -> list[dict[str, Any]]:
    selected = set(selected_ids or [])
    lanes = execution_shards(plan)
    if not selected:
        return lanes
    executions: list[dict[str, Any]] = []
    matched: set[str] = set()
    for lane in lanes:
        if lane["id"] in selected:
            executions.append(lane)
            matched.add(lane["id"])
            continue
        prefix = f"{lane['id']}--"
        for requested in sorted(item for item in selected if item.startswith(prefix)):
            variant_name = requested[len(prefix):]
            variants = [item for item in lane["build"]["variants"] if item["name"] == variant_name]
            if not variants:
                continue
            execution = copy.deepcopy(lane)
            execution["id"] = requested
            execution["build"]["variants"] = variants
            if execution["build"].get("buildAot"):
                execution["build"]["buildAot"] = variant_name == "base"
            if execution["build"].get("prebuildCrossPlatform"):
                execution["build"]["prebuildCrossPlatform"] = variant_name == "base"
            executions.append(execution)
            matched.add(requested)
    unmatched = sorted(selected - matched)
    if unmatched:
        raise SystemExit(f"No executions matched --shard {unmatched}")
    if not executions:
        raise SystemExit(f"No executions matched --shard {sorted(selected)}")
    return executions


def apply_execution_overrides(executions: list[dict[str, Any]], instance_type: str | None,
                              build_threads: int | None) -> None:
    """Apply explicit smoke/capacity overrides without modifying the release plan."""
    if instance_type:
        family = instance_type.split(".", 1)[0]
        if "." not in instance_type or family.startswith(GPU_INSTANCE_PREFIXES):
            raise SystemExit(f"invalid CPU compile instance type override: {instance_type}")
        if any(item.get("dedicatedHost") for item in executions):
            raise SystemExit("--instance-type cannot override an EC2 Mac dedicated-host shard")
        for item in executions:
            item["instanceType"] = instance_type
    if build_threads is not None:
        if build_threads < 1:
            raise SystemExit("--build-threads must be at least 1")
        for item in executions:
            item["build"]["buildThreads"] = build_threads


def apply_plan_defaults(plan: dict[str, Any], executions: list[dict[str, Any]]) -> None:
    defaults = plan.get("defaults", {})
    for item in executions:
        item["build"] = {**defaults, **item["build"]}


def apply_core_constraint(ec2, executions: list[dict[str, Any]], max_cores: int | None) -> list[dict[str, Any]]:
    """Greedily choose the largest compatible virtualized size within an EC2 vCPU budget."""
    if max_cores is None:
        return []
    if max_cores < 1:
        raise SystemExit("--max-cores must be at least 1")
    family_cache: dict[str, list[dict[str, Any]]] = {}
    schedule: list[dict[str, Any]] = []
    failures: list[str] = []
    for lane in executions:
        original_type = lane["instanceType"]
        if lane.get("dedicatedHost"):
            schedule.append({"lane": lane["id"], "originalInstanceType": original_type,
                             "selectedInstanceType": original_type, "constraintApplied": False,
                             "reason": "EC2 Mac dedicated-host size is fixed"})
            continue
        family = original_type.split(".", 1)[0]
        if family not in family_cache:
            request: dict[str, Any] = {"Filters": [{"Name": "instance-type", "Values": [f"{family}.*"]}]}
            descriptions: list[dict[str, Any]] = []
            while True:
                response = ec2.describe_instance_types(**request)
                descriptions.extend(response.get("InstanceTypes", []))
                token = response.get("NextToken")
                if not token:
                    break
                request["NextToken"] = token
            family_cache[family] = descriptions
        expected_architecture = lane["amiQuery"]["architecture"]
        candidates = [item for item in family_cache[family]
                      if expected_architecture in item.get("ProcessorInfo", {}).get("SupportedArchitectures", [])
                      and int(item.get("VCpuInfo", {}).get("DefaultVCpus", 0)) <= max_cores
                      and ".metal" not in item.get("InstanceType", "")]
        if candidates:
            offered_response = ec2.describe_instance_type_offerings(
                LocationType="region",
                Filters=[{"Name": "instance-type", "Values": [item["InstanceType"] for item in candidates]}],
            )
            offered_types = {item["InstanceType"] for item in offered_response.get("InstanceTypeOfferings", [])}
            candidates = [item for item in candidates if item["InstanceType"] in offered_types]
        if not candidates:
            available = sorted({int(item.get("VCpuInfo", {}).get("DefaultVCpus", 0))
                                for item in family_cache[family]
                                if expected_architecture in item.get("ProcessorInfo", {}).get("SupportedArchitectures", [])})
            failures.append(f"{lane['id']}: no {family} {expected_architecture} size <= {max_cores} vCPUs; family sizes={available}")
            continue
        selected = max(candidates, key=lambda item: (
            int(item["VCpuInfo"]["DefaultVCpus"]), int(item.get("MemoryInfo", {}).get("SizeInMiB", 0))))
        selected_cores = int(selected["VCpuInfo"]["DefaultVCpus"])
        memory_gib = int(selected.get("MemoryInfo", {}).get("SizeInMiB", 0)) // 1024
        build = lane["build"]
        original_threads = int(build.get("buildThreads", selected_cores))
        original_heap = int(build.get("mavenHeapGiB", 16))
        selected_threads = min(original_threads, selected_cores)
        selected_heap = min(original_heap, max(2, memory_gib // 2))
        lane["instanceType"] = selected["InstanceType"]
        build["buildThreads"] = selected_threads
        build["mavenHeapGiB"] = selected_heap
        schedule.append({
            "lane": lane["id"], "family": family, "originalInstanceType": original_type,
            "selectedInstanceType": selected["InstanceType"], "selectedVcpus": selected_cores,
            "memoryGiB": memory_gib, "buildThreads": selected_threads,
            "mavenHeapGiB": selected_heap, "constraintApplied": True,
        })
    if failures:
        raise SystemExit("Core constraint is infeasible:\n  " + "\n  ".join(failures))
    return schedule


def validate_launch_matrix(ec2, ssm, executions: list[dict[str, Any]], region: str,
                           availability_zone: str | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """Validate all AMIs and instance types before any mutable AWS operation."""
    instance_types = sorted({item["instanceType"] for item in executions})
    response = ec2.describe_instance_types(InstanceTypes=instance_types)
    descriptions = {item["InstanceType"]: item for item in response["InstanceTypes"]}
    missing = sorted(set(instance_types) - set(descriptions))
    if missing:
        raise RuntimeError(f"instance types do not exist in {region}: {missing}")
    for shard in executions:
        supported = set(descriptions[shard["instanceType"]]["ProcessorInfo"]["SupportedArchitectures"])
        expected = shard["amiQuery"]["architecture"]
        if expected not in supported:
            raise RuntimeError(f"{shard['instanceType']} does not support {expected} required by {shard['id']}")
    if availability_zone:
        offerings = ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": instance_types},
                     {"Name": "location", "Values": [availability_zone]}],
        )["InstanceTypeOfferings"]
        offered = {item["InstanceType"] for item in offerings}
        unavailable = sorted(set(instance_types) - offered)
        if unavailable:
            raise RuntimeError(f"instance types unavailable in {availability_zone}: {unavailable}")
    return ({item["id"]: resolve_ami(ec2, ssm, item) for item in executions}, descriptions)


def stream_lane_logs(logs_client, group: str, stream: str, token: str | None = None) -> tuple[str | None, int]:
    """Print newly available events from one CloudWatch stream and return its cursor."""
    request: dict[str, Any] = {
        "logGroupName": group,
        "logStreamName": stream,
        "startFromHead": True,
    }
    if token:
        request["nextToken"] = token
    response = logs_client.get_log_events(**request)
    events = response.get("events", [])
    for event in events:
        print(f"[{stream}] {event.get('message', '')}", flush=True)
    return response.get("nextForwardToken", token), len(events)


def stream_console_output(ec2, instance_id: str, offset: int) -> int:
    """Print new EC2 console output, which is available before CloudWatch bootstrap."""
    response = ec2.get_console_output(InstanceId=instance_id, Latest=True)
    output = response.get("Output", "") or ""
    if len(output) < offset:
        offset = 0
    if len(output) > offset:
        for line in output[offset:].splitlines():
            print(f"[{instance_id}/console] {line}", flush=True)
        offset = len(output)
    return offset


def instance_health(ec2, instance_id: str) -> tuple[str, str]:
    """Return AWS instance/system status checks even while they are initializing."""
    response = ec2.describe_instance_status(InstanceIds=[instance_id], IncludeAllInstances=True)
    statuses = response.get("InstanceStatuses", [])
    if not statuses:
        return "not-reported", "not-reported"
    status = statuses[0]
    return (
        status.get("InstanceStatus", {}).get("Status", "not-reported"),
        status.get("SystemStatus", {}).get("Status", "not-reported"),
    )


def wait_for_lane(ec2, s3, ssm, logs_client, plan: dict[str, Any], instance_id: str, bucket: str,
                  status_key: str, shard_id: str, log_group: str, log_stream: str) -> None:
    """Wait for a reusable lane while continuously reporting state and build activity."""
    started = time.monotonic()
    last_report = 0.0
    last_state = None
    last_health = None
    log_token = None
    console_offset = 0
    cloudwatch_active = False
    log_warning_reported = False
    console_warning_reported = False
    print(f"waiting for {shard_id}; live stream {log_group} / {log_stream}", flush=True)
    emit_cloudwatch_event(logs_client, log_group, log_stream,
                          f"phase=controller-wait status=started shard={shard_id} instance={instance_id}")
    while True:
        if kill_switch_enabled(ssm, plan):
            raise RuntimeError(f"global kill switch enabled while waiting for {shard_id}")
        reservations = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"]
        state = reservations[0]["Instances"][0]["State"]["Name"]
        elapsed = int(time.monotonic() - started)
        if state != last_state:
            reason = reservations[0]["Instances"][0].get("StateTransitionReason", "")
            suffix = f"; reason={reason}" if reason else ""
            message = f"phase=ec2-state status={state} shard={shard_id} instance={instance_id} elapsedSeconds={elapsed}{suffix}"
            print(f"[{shard_id}] EC2 state: {state} ({elapsed}s elapsed){suffix}", flush=True)
            emit_cloudwatch_event(logs_client, log_group, log_stream, message)
            last_state = state
        try:
            health = instance_health(ec2, instance_id)
            if health != last_health:
                print(f"[{shard_id}] AWS health: instance={health[0]} system={health[1]}", flush=True)
                emit_cloudwatch_event(
                    logs_client, log_group, log_stream,
                    f"phase=ec2-health status=changed shard={shard_id} instance={instance_id} instanceCheck={health[0]} systemCheck={health[1]}",
                )
                last_health = health
        except Exception as exc:
            print(f"[{shard_id}] AWS health unavailable: {exc}", flush=True)
        try:
            log_token, event_count = stream_lane_logs(logs_client, log_group, log_stream, log_token)
            cloudwatch_active = cloudwatch_active or event_count > 0
        except Exception as exc:
            if not log_warning_reported:
                print(f"[{shard_id}] CloudWatch stream not ready: {exc}", flush=True)
                log_warning_reported = True
        if not cloudwatch_active:
            try:
                console_offset = stream_console_output(ec2, instance_id, console_offset)
            except Exception as exc:
                if not console_warning_reported:
                    print(f"[{shard_id}] EC2 console output not ready: {exc}", flush=True)
                    console_warning_reported = True
        now = time.monotonic()
        if now - last_report >= 60:
            print(f"[{shard_id}] still {state}; {elapsed}s elapsed; waiting for build completion", flush=True)
            emit_cloudwatch_event(
                logs_client, log_group, log_stream,
                f"phase=controller-heartbeat status={state} shard={shard_id} instance={instance_id} elapsedSeconds={elapsed}",
            )
            last_report = now
        if state in {"shutting-down", "terminated"}:
            try:
                console_offset = stream_console_output(ec2, instance_id, console_offset)
            except Exception as exc:
                print(f"[{shard_id}] final EC2 console retrieval failed: {exc}", flush=True)
            break
        time.sleep(15)

    # The worker uploads status before shutting down. Drain final log events while
    # allowing a short grace period for an in-flight S3 upload.
    deadline = time.monotonic() + 300
    while True:
        try:
            log_token, _ = stream_lane_logs(logs_client, log_group, log_stream, log_token)
        except Exception:
            pass
        try:
            response = s3.get_object(Bucket=bucket, Key=status_key)
            status = json.loads(response["Body"].read())
            exit_code = int(status.get("exitCode", 1))
            if exit_code != 0:
                raise RuntimeError(f"lane {shard_id} failed with exit code {exit_code}; see live output above")
            return
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") not in {"NoSuchKey", "404"}:
                raise
            remaining = max(0, int(deadline - time.monotonic()))
            if remaining == 0:
                try:
                    stream_console_output(ec2, instance_id, console_offset)
                except Exception:
                    pass
                raise RuntimeError(
                    f"lane {shard_id} terminated without status.json; bootstrap failed before its final upload. "
                    f"Review the console and CloudWatch output above"
                ) from exc
            print(f"[{shard_id}] instance terminated; waiting for final status upload ({remaining}s remaining)", flush=True)
            time.sleep(min(15, remaining))


def render_user_data(worker: Path, values: dict[str, Any]) -> str:
    encoded = base64.b64encode(json.dumps(values, separators=(",", ":")).encode()).decode()
    driver = base64.b64encode((worker.parent / "build-platform.py").read_bytes()).decode()
    forwarder = base64.b64encode((worker.parent / "log-forwarder.py").read_bytes()).decode()
    return (worker.read_text(encoding="utf-8")
            .replace("__DL4J_WORKER_CONFIG_B64__", encoded)
            .replace("__DL4J_BUILD_DRIVER_B64__", driver)
            .replace("__DL4J_LOG_FORWARDER_B64__", forwarder))


def bootstrap_user_data(os_name: str, url: str) -> str:
    if os_name == "windows":
        return ("<powershell>\n$ErrorActionPreference='Stop'\n"
                "Write-Output '[dl4j-phase] phase=cloud-init status=started'\n"
                f"Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile C:\\dl4j-worker.ps1\n"
                "Write-Output '[dl4j-phase] phase=worker-download status=complete'\n"
                "& powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\\dl4j-worker.ps1\n</powershell>\n")
    return ("#!/usr/bin/env bash\nset -Eeuo pipefail\n"
            "printf '[dl4j-phase] phase=cloud-init status=started\\n'\n"
            f"curl --fail --location --retry 5 '{url}' -o /tmp/dl4j-worker.sh\n"
            "printf '[dl4j-phase] phase=worker-download status=complete\\n'\n"
            "chmod 700 /tmp/dl4j-worker.sh\nexec /tmp/dl4j-worker.sh\n")


def managed_hosts(ec2, run_id: str | None = None) -> list[dict[str, Any]]:
    filters = [{"Name": f"tag:{MANAGED_TAG}", "Values": ["true"]}]
    if run_id:
        filters.append({"Name": f"tag:{RUN_TAG}", "Values": [run_id]})
    return ec2.describe_hosts(Filter=filters).get("Hosts", [])


def managed_instances(ec2, run_id: str | None = None, include_terminated: bool = False) -> list[dict[str, Any]]:
    filters = [{"Name": f"tag:{MANAGED_TAG}", "Values": ["true"]}]
    if not include_terminated:
        filters.append({"Name": "instance-state-name", "Values": list(ACTIVE_STATES)})
    if run_id:
        filters.append({"Name": f"tag:{RUN_TAG}", "Values": [run_id]})
    reservations = ec2.describe_instances(Filters=filters)["Reservations"]
    return [instance for reservation in reservations for instance in reservation["Instances"]]


def preflight(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    session, region, ec2, ssm, _, sts, _ = session_clients(args.region)
    executions = selected_executions(plan, args.shard)
    apply_plan_defaults(plan, executions)
    apply_execution_overrides(executions, args.instance_type, args.build_threads)
    if args.instance_type and args.max_cores is not None:
        raise SystemExit("--instance-type and --max-cores are mutually exclusive")
    core_schedule = apply_core_constraint(ec2, executions, args.max_cores)
    instance_types = sorted({item["instanceType"] for item in executions})
    subnet_id, group_id, default_az = default_network(ec2, instance_types)
    availability_zones = [item["ZoneName"] for item in ec2.describe_availability_zones(
        Filters=[{"Name": "state", "Values": ["available"]}])["AvailabilityZones"]]
    descriptions = {item["InstanceType"]: item for item in ec2.describe_instance_types(InstanceTypes=instance_types)["InstanceTypes"]}
    offerings: dict[str, list[str]] = {}
    for instance_type in instance_types:
        response = ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}, {"Name": "location", "Values": availability_zones}],
        )
        offerings[instance_type] = sorted(item["Location"] for item in response["InstanceTypeOfferings"])
    missing = [instance_type for instance_type, zones in offerings.items() if not zones]
    if missing:
        raise RuntimeError(f"instance types unavailable in {region}: {missing}")
    amis, _ = validate_launch_matrix(ec2, ssm, executions, region)
    required_vcpus = max((descriptions[item["instanceType"]]["VCpuInfo"]["DefaultVCpus"]
                          for item in executions if not item.get("dedicatedHost")), default=0)
    quota_value = None
    quota_error = None
    try:
        quota_value = session.client("service-quotas").get_service_quota(
            ServiceCode="ec2", QuotaCode="L-1216C47A")["Quota"]["Value"]
    except Exception as exc:
        quota_error = str(exc)
    result = {
        "account": sts.get_caller_identity()["Account"], "region": region,
        "executions": len(executions), "instanceTypes": instance_types,
        "maxCoresConstraint": args.max_cores, "coreConstraintSchedule": core_schedule,
        "offeringsByAvailabilityZone": offerings, "resolvedAmis": amis,
        "defaultNetwork": {"subnetId": subnet_id, "securityGroupId": group_id, "availabilityZone": default_az},
        "peakStandardOnDemandVcpusSerialLanes": required_vcpus,
        "standardOnDemandVcpuQuota": quota_value, "quotaReadError": quota_error,
        "macDefaultAzSupportsType": all(default_az in offerings[item["instanceType"]] for item in executions if item.get("dedicatedHost")),
    }
    print(json.dumps(result, indent=2))
    if any(item.get("dedicatedHost") for item in executions) and not result["macDefaultAzSupportsType"]:
        raise SystemExit(f"Preflight failed: default subnet AZ {default_az} does not offer the selected EC2 Mac type; pass a supported --subnet-id when starting")
    if quota_value is not None and required_vcpus > quota_value:
        raise SystemExit(f"Preflight failed: largest serial lane requires {required_vcpus} standard vCPUs but quota is {quota_value:g}")


def start(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    commit = args.commit or resolve_branch(args.repository, args.branch)
    session, region, ec2, ssm, s3, sts, iam = session_clients(args.region)
    logs_client = session.client("logs")
    executions = selected_executions(plan, args.shard)
    apply_plan_defaults(plan, executions)
    apply_execution_overrides(executions, args.instance_type, args.build_threads)
    if args.instance_type and args.max_cores is not None:
        raise SystemExit("--instance-type and --max-cores are mutually exclusive")
    core_schedule = apply_core_constraint(ec2, executions, args.max_cores)
    subnet_id, group_id = (args.subnet_id, args.security_group_id)
    if not subnet_id or not group_id:
        subnet_id, group_id, availability_zone = default_network(
            ec2, sorted({item["instanceType"] for item in executions}))
    else:
        availability_zone = ec2.describe_subnets(SubnetIds=[subnet_id])["Subnets"][0]["AvailabilityZone"]
    resolved_amis, descriptions = validate_launch_matrix(ec2, ssm, executions, region, availability_zone)
    peak_vcpus = max((int(descriptions[item["instanceType"]]["VCpuInfo"]["DefaultVCpus"])
                      for item in executions if not item.get("dedicatedHost")), default=0)
    if args.max_cores is not None and peak_vcpus > args.max_cores:
        raise SystemExit(f"calculated schedule requires {peak_vcpus} vCPUs, exceeding --max-cores {args.max_cores}")
    try:
        quota_value = float(session.client("service-quotas").get_service_quota(
            ServiceCode="ec2", QuotaCode="L-1216C47A")["Quota"]["Value"])
    except Exception as exc:
        if args.max_cores is not None:
            raise SystemExit(f"cannot prove --max-cores schedule is launchable because the EC2 quota could not be read: {exc}") from exc
        quota_value = None
    if quota_value is not None and peak_vcpus > quota_value:
        raise SystemExit(f"calculated schedule requires {peak_vcpus} standard vCPUs but account quota is {quota_value:g}")
    if core_schedule:
        print(json.dumps({"maxCoresConstraint": args.max_cores, "coreConstraintSchedule": core_schedule}, indent=2))
    if kill_switch_enabled(ssm, plan) and not args.reset_kill_switch:
        raise SystemExit("Global kill switch is ON. Pass --reset-kill-switch to explicitly start a new release.")
    set_kill_switch(ssm, plan, False)
    bucket = ensure_bucket(s3, sts, region, args.bucket)
    log_group = ensure_log_group(logs_client, plan)
    profile = ensure_instance_profile(iam, bucket, kill_parameter_name(plan), log_group)
    run_id = args.run_id or f"{args.version}-{uuid.uuid4().hex[:10]}"
    print(json.dumps({
        "event": "run-created", "runId": run_id, "region": region, "bucket": bucket,
        "sourceBranch": args.branch, "resolvedCommit": commit,
        "logsCommand": f"python3 release/aws/release.py --region {region} logs --run-id {run_id} --follow",
        "statusCommand": f"python3 release/aws/release.py --region {region} status --run-id {run_id}",
        "shutdownCommand": f"python3 release/aws/release.py --region {region} stop-everything --wait",
    }, indent=2), flush=True)
    defaults = plan.get("defaults", {})
    launched: list[str] = []
    allocated_hosts: list[str] = []
    try:
        for shard in executions:
            if kill_switch_enabled(ssm, plan):
                raise RuntimeError("global kill switch enabled during provisioning")
            ami = resolved_amis[shard["id"]]
            instance_type = shard["instanceType"]
            worker = args.plan.parent / shard["worker"]
            shard["build"] = {**defaults, **shard["build"]}
            config = {
                "bucket": bucket,
                "artifactPrefix": plan.get("artifactPrefix", "deeplearning4j/releases"),
                "runId": run_id,
                "releaseVersion": args.version,
                "snapshotVersion": args.snapshot_version,
                "commit": commit,
                "sourceBranch": args.branch,
                "repository": args.repository,
                "killSwitchParameter": kill_parameter_name(plan),
                "logGroupName": log_group,
                "logStreamName": f"{run_id}/{shard['id']}",
                "shard": shard,
                "region": region,
            }
            worker_key = f"{plan.get('artifactPrefix', 'deeplearning4j/releases')}/{run_id}/bootstrap/{shard['id']}-{shard['worker']}"
            ensure_log_stream(logs_client, log_group, config["logStreamName"])
            s3.put_object(Bucket=bucket, Key=worker_key, Body=render_user_data(worker, config).encode(), ServerSideEncryption="AES256")
            worker_url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": worker_key}, ExpiresIn=86400)
            placement: dict[str, str] = {}
            if shard.get("dedicatedHost"):
                host_response = ec2.allocate_hosts(
                    AvailabilityZone=availability_zone,
                    InstanceType=instance_type,
                    Quantity=1,
                    AutoPlacement="off",
                    HostRecovery="off",
                    TagSpecifications=[{"ResourceType": "dedicated-host", "Tags": [
                        {"Key": "Name", "Value": f"dl4j-release-{shard['id']}"},
                        {"Key": MANAGED_TAG, "Value": "true"},
                        {"Key": PROJECT_TAG, "Value": str(plan.get("projectTag", "deeplearning4j-release"))},
                        {"Key": RUN_TAG, "Value": run_id},
                        {"Key": SHARD_TAG, "Value": shard["id"]},
                    ]}],
                )
                host_id = host_response["HostIds"][0]
                allocated_hosts.append(host_id)
                deadline = time.time() + 1800
                while ec2.describe_hosts(HostIds=[host_id])["Hosts"][0]["State"] != "available":
                    if kill_switch_enabled(ssm, plan):
                        raise RuntimeError("global kill switch enabled while allocating EC2 Mac host")
                    if time.time() >= deadline:
                        raise TimeoutError(f"EC2 Mac host {host_id} did not become available")
                    time.sleep(15)
                placement = {"HostId": host_id, "Tenancy": "host"}
            launch_request: dict[str, Any] = {
                "ImageId": ami,
                "InstanceType": instance_type,
                "MinCount": 1,
                "MaxCount": 1,
                "IamInstanceProfile": {"Name": profile},
                "InstanceInitiatedShutdownBehavior": "terminate",
                "UserData": bootstrap_user_data(shard["os"], worker_url),
                "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": subnet_id, "Groups": [group_id], "AssociatePublicIpAddress": True, "DeleteOnTermination": True}],
                "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": int(shard.get("rootVolumeGiB", defaults.get("rootVolumeGiB", 750))), "VolumeType": "gp3", "DeleteOnTermination": True, "Encrypted": True}}],
                "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                    {"Key": "Name", "Value": f"dl4j-release-{shard['id']}"},
                    {"Key": MANAGED_TAG, "Value": "true"},
                    {"Key": PROJECT_TAG, "Value": str(plan.get("projectTag", "deeplearning4j-release"))},
                    {"Key": RUN_TAG, "Value": run_id},
                    {"Key": SHARD_TAG, "Value": shard["id"]},
                ]}],
            }
            if placement:
                launch_request["Placement"] = placement
            response = ec2.run_instances(**launch_request)
            instance_id = response["Instances"][0]["InstanceId"]
            launched.append(instance_id)
            if kill_switch_enabled(ssm, plan):
                ec2.terminate_instances(InstanceIds=launched)
                raise RuntimeError("global kill switch enabled immediately after launch")
            print(f"launched {shard['id']}: {instance_id} ({shard['os']}, {shard['build']['backend']}, CPU compile host)")
            status_key = f"{plan.get('artifactPrefix', 'deeplearning4j/releases')}/{run_id}/{shard['id']}/status.json"
            wait_for_lane(
                ec2, s3, ssm, logs_client, plan, instance_id, bucket, status_key,
                shard["id"], log_group, config["logStreamName"],
            )
            print(f"completed {shard['id']}: {instance_id}; launching next lane")
    except Exception:
        set_kill_switch(ssm, plan, True)
        if launched:
            ec2.terminate_instances(InstanceIds=launched)
        for host_id in allocated_hosts:
            try:
                ec2.release_hosts(HostIds=[host_id])
            except Exception:
                pass
        raise
    print(json.dumps({
        "runId": run_id,
        "region": region,
        "bucket": bucket,
        "sourceBranch": args.branch,
        "resolvedCommit": commit,
        "instances": launched,
        "dedicatedHosts": allocated_hosts,
        "logGroup": log_group,
        "logsCommand": f"python3 release/aws/release.py --region {region} logs --run-id {run_id} --follow",
        "shutdownCommand": f"python3 release/aws/release.py --region {region} stop-everything --wait",
    }, indent=2))


def status(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    _, region, ec2, ssm, s3, _, _ = session_clients(args.region)
    instances = managed_instances(ec2, args.run_id, include_terminated=True)
    rows = []
    for instance in instances:
        tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
        try:
            health = instance_health(ec2, instance["InstanceId"])
        except Exception:
            health = ("unavailable", "unavailable")
        try:
            console = ec2.get_console_output(InstanceId=instance["InstanceId"], Latest=True).get("Output", "") or ""
        except Exception as exc:
            console = f"console unavailable: {exc}"
        launch_time = instance.get("LaunchTime")
        rows.append({
            "instanceId": instance["InstanceId"], "state": instance["State"]["Name"],
            "stateTransitionReason": instance.get("StateTransitionReason", ""),
            "instanceHealth": health[0], "systemHealth": health[1],
            "launchTime": launch_time.isoformat() if hasattr(launch_time, "isoformat") else str(launch_time or ""),
            "runId": tags.get(RUN_TAG), "shard": tags.get(SHARD_TAG), "type": instance["InstanceType"],
            "consoleOutputTail": console[-12000:].splitlines(),
        })
    hosts = [{"hostId": host["HostId"], "state": host["State"], "type": host.get("InstanceType")} for host in managed_hosts(ec2, args.run_id)]
    print(json.dumps({
        "region": region,
        "killSwitch": kill_switch_enabled(ssm, plan),
        "instances": rows,
        "dedicatedHosts": hosts,
        "logGroup": log_group_name(plan),
        "logsCommand": f"python3 release/aws/release.py --region {region} logs --run-id {args.run_id} --follow" if args.run_id else None,
    }, indent=2))


def show_logs(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    session, region, _, _, _, _, _ = session_clients(args.region)
    logs_client = session.client("logs")
    group = log_group_name(plan)
    selected = set(args.shard or [])
    if selected:
        execution_ids = {
            item["id"] for item in execution_shards(plan)
            if item["id"] in selected or item.get("parentShard") in selected
        }
        if not execution_ids:
            raise SystemExit(f"No executions matched --shard {sorted(selected)}")
    else:
        execution_ids = set()
    since = None
    if args.since_minutes is not None:
        since = int((time.time() - args.since_minutes * 60) * 1000)
    elif args.follow:
        since = int((time.time() - 10 * 60) * 1000)
    try:
        while True:
            streams = [name for name in list_log_streams(logs_client, group, f"{args.run_id}/")
                       if not execution_ids or name.rsplit("/", 1)[-1] in execution_ids]
            if streams:
                request: dict[str, Any] = {"logGroupName": group, "logStreamNames": streams, "interleaved": True}
                if since is not None:
                    request["startTime"] = since
                while True:
                    response = logs_client.filter_log_events(**request)
                    for event in response.get("events", []):
                        stream = event.get("logStreamName", "unknown")
                        print(f"[{stream}] {event.get('message', '')}", flush=True)
                        since = max(since or 0, int(event["timestamp"]) + 1)
                    token = response.get("nextToken")
                    if not token or token == request.get("nextToken"):
                        break
                    request["nextToken"] = token
            if not args.follow:
                return
            time.sleep(3)
    except KeyboardInterrupt:
        return


def delete_logs(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("Log deletion requires --yes")
    plan = load_plan(args.plan)
    session, region, _, _, _, _, _ = session_clients(args.region)
    logs_client = session.client("logs")
    group = log_group_name(plan)
    selected = set(args.shard or [])
    execution_ids = None
    if selected:
        execution_ids = {
            item["id"] for item in execution_shards(plan)
            if item["id"] in selected or item.get("parentShard") in selected
        }
        if not execution_ids:
            raise SystemExit(f"No executions matched --shard {sorted(selected)}")
    prefix = None if args.all_runs else f"{args.run_id}/"
    deleted = delete_log_streams(logs_client, group, prefix, execution_ids)
    print(json.dumps({"region": region, "logGroup": group, "deletedLogStreams": deleted}, indent=2))


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def publish_test_repository(
    s3, bucket: str, run_prefix: str, repository: Path, repository_manifest: Path,
    run_id: str, version: str, commit: str, complete_matrix: bool,
) -> dict[str, Any]:
    """Publish an exploded Maven 2 layout, with the readiness marker written last."""
    repository_prefix = f"{run_prefix}maven-repository/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=repository_prefix):
        stale = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if stale:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": stale, "Quiet": True})
    files = sorted(path for path in repository.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(repository).as_posix()
        s3.upload_file(
            str(path), bucket, repository_prefix + relative,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
    metadata_prefix = repository_prefix + ".dl4j/"
    s3.upload_file(
        str(repository_manifest), bucket, metadata_prefix + "repository-manifest.json",
        ExtraArgs={"ServerSideEncryption": "AES256", "ContentType": "application/json"},
    )
    manifest_checksum = Path(str(repository_manifest) + ".sha256")
    s3.upload_file(
        str(manifest_checksum), bucket, metadata_prefix + "repository-manifest.json.sha256",
        ExtraArgs={"ServerSideEncryption": "AES256", "ContentType": "text/plain"},
    )
    marker = {
        "schemaVersion": 1,
        "layout": "maven2",
        "ready": True,
        "runId": run_id,
        "releaseVersion": version,
        "commit": commit,
        "completeMatrix": complete_matrix,
        "repositoryFiles": len(files),
        "manifestSha256": file_digest(repository_manifest),
    }
    s3.put_object(
        Bucket=bucket,
        Key=metadata_prefix + "complete.json",
        Body=(json.dumps(marker, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )
    return {
        "uri": f"s3://{bucket}/{repository_prefix}",
        "completionMarker": f"s3://{bucket}/{metadata_prefix}complete.json",
        **marker,
    }


def collect(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    _, region, _, _, s3, _, _ = session_clients(args.region)
    prefix = f"{plan.get('artifactPrefix', 'deeplearning4j/releases')}/{args.run_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    objects = [item for page in paginator.paginate(Bucket=args.bucket, Prefix=prefix) for item in page.get("Contents", [])]
    if not objects:
        raise SystemExit(f"No release outputs found at s3://{args.bucket}/{prefix}")
    executions = execution_shards(plan)
    selected = set(args.shard or [])
    expected_shards = {
        item["id"] for item in executions
        if not selected or item["id"] in selected or item.get("parentShard") in selected
    }
    with tempfile.TemporaryDirectory(prefix="dl4j-release-collect-") as temporary:
        directory = Path(temporary)
        seen_status: set[str] = set()
        seen_manifests: set[str] = set()
        seen_outputs: dict[str, set[str]] = {shard: set() for shard in expected_shards}
        planned_shards = {item["id"]: item for item in executions if item["id"] in expected_shards}
        planned_workloads = {shard_id: set(item["workloads"]) for shard_id, item in planned_shards.items()}
        assets: list[dict[str, Any]] = []
        maven_archives: list[Path] = []
        for item in objects:
            relative = item["Key"][len(prefix):]
            parts = relative.split("/", 1)
            if len(parts) != 2 or parts[0] not in expected_shards:
                continue
            shard, name = parts
            output_name = f"{name.removesuffix('.tar.gz')}-{shard}.tar.gz" if name.endswith(".tar.gz") else f"{shard}-{name}"
            output = directory / output_name
            s3.download_file(args.bucket, item["Key"], str(output))
            if name == "status.json":
                status_data = json.loads(output.read_text(encoding="utf-8"))
                if status_data.get("exitCode") != 0 or status_data.get("shard") != shard:
                    raise RuntimeError(f"shard {shard} failed or returned an invalid status: {status_data}")
                seen_status.add(shard)
            if name == "shard-manifest.json":
                shard_manifest = json.loads(output.read_text(encoding="utf-8"))
                expected_identity = (
                    args.run_id, shard, args.commit, args.version, planned_workloads[shard],
                    planned_shards[shard]["os"], planned_shards[shard]["build"]["javacppPlatform"], planned_shards[shard]["build"]["backend"],
                )
                actual_identity = (
                    shard_manifest.get("runId"),
                    shard_manifest.get("shard"),
                    shard_manifest.get("commit"),
                    shard_manifest.get("releaseVersion"),
                    set(shard_manifest.get("workloads", [])),
                    shard_manifest.get("os"),
                    shard_manifest.get("platform"),
                    shard_manifest.get("backend"),
                )
                if actual_identity != expected_identity:
                    raise RuntimeError(f"shard manifest identity mismatch for {shard}: {actual_identity}")
                seen_manifests.add(shard)
            if name in {"maven-repository.tar.gz", "sdk-assets.tar.gz"}:
                seen_outputs[shard].add("maven" if name.startswith("maven-") else "sdk")
            if name == "maven-repository.tar.gz":
                maven_archives.append(output)
            if name in {"maven-repository.tar.gz", "sdk-assets.tar.gz", "shard-manifest.json", "build.log"}:
                assets.append({"fileName": output_name, "sha256": file_digest(output), "size": output.stat().st_size, "shard": shard, "sourceKey": item["Key"]})
        missing = sorted(expected_shards - seen_status)
        if missing:
            raise RuntimeError(f"release is incomplete; missing successful status for shards: {missing}")
        missing_manifests = sorted(expected_shards - seen_manifests)
        if missing_manifests:
            raise RuntimeError(f"release is incomplete; missing shard manifests: {missing_manifests}")
        missing_outputs = {shard: sorted(planned_workloads[shard] - seen_outputs[shard]) for shard in expected_shards if planned_workloads[shard] - seen_outputs[shard]}
        if missing_outputs:
            raise RuntimeError(f"release is incomplete; missing workload archives: {missing_outputs}")
        test_repository = directory / "s3-maven-repository"
        test_repository_manifest = directory / "s3-maven-repository-manifest.json"
        central_tool = Path(__file__).resolve().parents[1] / "central/repository.py"
        materialize_command = [
            sys.executable, str(central_tool), "materialize-test-repository",
            "--output", str(test_repository), "--manifest", str(test_repository_manifest),
            "--release-version", args.version, "--commit", args.commit,
        ]
        for archive in sorted(maven_archives):
            materialize_command.extend(("--input", str(archive)))
        subprocess.run(materialize_command, check=True)
        test_repository_info = publish_test_repository(
            s3, args.bucket, prefix, test_repository, test_repository_manifest,
            args.run_id, args.version, args.commit, not selected,
        )
        manifest = {
            "schemaVersion": 1,
            "runId": args.run_id,
            "releaseTag": args.release_tag,
            "releaseVersion": args.version,
            "commit": args.commit,
            "region": region,
            "bucket": args.bucket,
            "workloads": ["maven", "sdk"],
            "shards": sorted(expected_shards),
            "assets": sorted(assets, key=lambda value: value["fileName"]),
            "testMavenRepository": test_repository_info,
        }
        manifest_path = directory / "release-build-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksum_path = directory / "release-build-manifest.json.sha256"
        checksum_path.write_text(f"{file_digest(manifest_path)}  {manifest_path.name}\n", encoding="ascii")
        repository = args.github_repository
        view = subprocess.run(["gh", "release", "view", args.release_tag, "--repo", repository], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if view.returncode != 0:
            subprocess.run(["gh", "release", "create", args.release_tag, "--repo", repository, "--target", args.commit, "--draft", "--title", f"DL4J {args.version} external build"], check=True)
        upload_files = [str(directory / item["fileName"]) for item in assets]
        upload_files.extend([str(manifest_path), str(checksum_path)])
        subprocess.run(["gh", "release", "upload", args.release_tag, "--repo", repository, "--clobber", *upload_files], check=True)
        print(json.dumps(manifest, indent=2))


def stop_everything(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan)
    session, region, ec2, ssm, s3, sts, _ = session_clients(args.region)
    set_kill_switch(ssm, plan, True)
    victims = managed_instances(ec2)
    hosts = managed_hosts(ec2)
    ids = [item["InstanceId"] for item in victims]
    host_ids = [item["HostId"] for item in hosts if item.get("State") != "released"]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    spot = ec2.describe_spot_instance_requests(Filters=[{"Name": f"tag:{MANAGED_TAG}", "Values": ["true"]}, {"Name": "state", "Values": ["open", "active"]}])["SpotInstanceRequests"]
    spot_ids = [item["SpotInstanceRequestId"] for item in spot]
    if spot_ids:
        ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=spot_ids)
    if args.wait and ids:
        ec2.get_waiter("instance_terminated").wait(InstanceIds=ids)
    released_hosts: list[str] = []
    pending_hosts: dict[str, str] = {}
    for host_id in host_ids:
        try:
            ec2.release_hosts(HostIds=[host_id])
            released_hosts.append(host_id)
        except Exception as exc:
            pending_hosts[host_id] = str(exc)
    if args.purge_storage:
        bucket = ensure_bucket(s3, sts, region, args.bucket)
        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket):
            objects = [{"Key": item["Key"], "VersionId": item["VersionId"]} for key in ("Versions", "DeleteMarkers") for item in page.get(key, [])]
            if objects:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
    deleted_logs = []
    if getattr(args, "purge_logs", False):
        deleted_logs = delete_log_streams(session.client("logs"), log_group_name(plan))
    print(json.dumps({"region": region, "killSwitch": True, "terminatedInstances": ids, "cancelledSpotRequests": spot_ids, "releasedDedicatedHosts": released_hosts, "pendingDedicatedHosts": pending_hosts, "storagePurged": args.purge_storage, "deletedLogStreams": deleted_logs}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=root / "release/aws/release-plan.json")
    result.add_argument("--region")
    sub = result.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--shard", action="append")
    check.add_argument("--instance-type", help="validated CPU instance override for capacity-limited smoke tests")
    check.add_argument("--build-threads", type=int, help="build thread override to match a smaller smoke host")
    check.add_argument("--max-cores", type=int, help="greedily select the largest compatible size per lane within this EC2 vCPU limit")
    check.set_defaults(func=preflight)
    launch = sub.add_parser("start")
    launch.add_argument("--version", required=True)
    launch.add_argument("--snapshot-version", default="1.0.0-SNAPSHOT")
    source = launch.add_mutually_exclusive_group(required=True)
    source.add_argument("--commit")
    source.add_argument("--branch")
    launch.add_argument("--repository", default="https://github.com/deeplearning4j/deeplearning4j.git")
    launch.add_argument("--run-id")
    launch.add_argument("--shard", action="append")
    launch.add_argument("--instance-type", help="validated CPU instance override for capacity-limited smoke tests")
    launch.add_argument("--build-threads", type=int, help="build thread override to match a smaller smoke host")
    launch.add_argument("--max-cores", type=int, help="greedily select the largest compatible size per lane within this EC2 vCPU limit")
    launch.add_argument("--bucket")
    launch.add_argument("--subnet-id")
    launch.add_argument("--security-group-id")
    launch.add_argument("--reset-kill-switch", action="store_true")
    launch.set_defaults(func=start)
    show = sub.add_parser("status")
    show.add_argument("--run-id")
    show.set_defaults(func=status)
    live_logs = sub.add_parser("logs")
    live_logs.add_argument("--run-id", required=True)
    live_logs.add_argument("--shard", action="append")
    live_logs.add_argument("--follow", action="store_true")
    live_logs.add_argument("--since-minutes", type=float)
    live_logs.set_defaults(func=show_logs)
    remove_logs = sub.add_parser("delete-logs")
    target = remove_logs.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-id")
    target.add_argument("--all-runs", action="store_true")
    remove_logs.add_argument("--shard", action="append")
    remove_logs.add_argument("--yes", action="store_true")
    remove_logs.set_defaults(func=delete_logs)
    gather = sub.add_parser("collect")
    gather.add_argument("--run-id", required=True)
    gather.add_argument("--bucket", required=True)
    gather.add_argument("--release-tag", required=True)
    gather.add_argument("--version", required=True)
    gather.add_argument("--commit", required=True)
    gather.add_argument("--github-repository", default="deeplearning4j/deeplearning4j")
    gather.add_argument("--shard", action="append")
    gather.set_defaults(func=collect)
    stop = sub.add_parser("stop-everything")
    stop.add_argument("--bucket")
    stop.add_argument("--wait", action="store_true")
    stop.add_argument("--purge-storage", action="store_true")
    stop.add_argument("--purge-logs", action="store_true")
    stop.set_defaults(func=stop_everything)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
