#!/usr/bin/env python3
"""Unit tests for the release provisioner's fail-closed AWS validation."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("dl4j_aws_release", Path(__file__).with_name("release.py"))
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)
FORWARDER_SPEC = importlib.util.spec_from_file_location(
    "dl4j_aws_log_forwarder", Path(__file__).with_name("log-forwarder.py")
)
log_forwarder = importlib.util.module_from_spec(FORWARDER_SPEC)
assert FORWARDER_SPEC.loader is not None
FORWARDER_SPEC.loader.exec_module(log_forwarder)
BUILD_PLATFORM_SPEC = importlib.util.spec_from_file_location(
    "dl4j_aws_build_platform", Path(__file__).with_name("build-platform.py")
)
build_platform = importlib.util.module_from_spec(BUILD_PLATFORM_SPEC)
assert BUILD_PLATFORM_SPEC.loader is not None
BUILD_PLATFORM_SPEC.loader.exec_module(build_platform)


class FakeSsm:
    def get_parameter(self, **_kwargs):
        raise AssertionError("query-backed AMIs must not use an unverified SSM parameter")


class FakeEc2:
    def __init__(self, image=None, architectures=None, offered=True):
        self.image = image or {
            "ImageId": "ami-verified", "Name": "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-20260701",
            "OwnerId": "099720109477", "Architecture": "x86_64", "State": "available",
            "RootDeviceType": "ebs", "VirtualizationType": "hvm", "CreationDate": "2026-07-01T00:00:00Z",
        }
        self.architectures = architectures or ["x86_64"]
        self.offered = offered

    def describe_images(self, **_kwargs):
        return {"Images": [self.image]}

    def describe_instance_types(self, InstanceTypes):
        return {"InstanceTypes": [{
            "InstanceType": item, "ProcessorInfo": {"SupportedArchitectures": self.architectures},
            "VCpuInfo": {"DefaultVCpus": 96},
        } for item in InstanceTypes]}

    def describe_instance_type_offerings(self, **kwargs):
        values = kwargs["Filters"][0]["Values"] if self.offered else []
        zone = kwargs["Filters"][1]["Values"][0]
        return {"InstanceTypeOfferings": [{"InstanceType": item, "Location": zone} for item in values]}


class SizingEc2:
    def describe_instance_types(self, **_kwargs):
        return {"InstanceTypes": [
            {"InstanceType": "c7i.large", "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
             "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 4096}},
            {"InstanceType": "c7i.xlarge", "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
             "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 8192}},
            {"InstanceType": "c7i.2xlarge", "ProcessorInfo": {"SupportedArchitectures": ["x86_64"]},
             "VCpuInfo": {"DefaultVCpus": 8}, "MemoryInfo": {"SizeInMiB": 16384}},
        ]}

    def describe_instance_type_offerings(self, **kwargs):
        values = kwargs["Filters"][0]["Values"]
        return {"InstanceTypeOfferings": [{"InstanceType": item, "Location": "us-east-1"} for item in values]}


class FakeLogs:
    def __init__(self):
        self.requests = []
        self.published = []

    def put_log_events(self, **kwargs):
        self.published.append(kwargs)
        return {}

    def get_log_events(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) == 1:
            return {
                "events": [{"timestamp": 1, "message": "bootstrap ready"}],
                "nextForwardToken": "cursor-1",
            }
        return {
            "events": [{"timestamp": 2, "message": "building base"}],
            "nextForwardToken": "cursor-2",
        }


class FakeS3Paginator:
    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def paginate(self, **kwargs):
        self.requests.append(kwargs)
        return self.pages


class FakeS3:
    def __init__(self, pages):
        self.paginator = FakeS3Paginator(pages)
        self.deletions = []

    def get_paginator(self, name):
        if name != "list_object_versions":
            raise AssertionError(name)
        return self.paginator

    def delete_objects(self, **kwargs):
        self.deletions.append(kwargs)
        return {}


class FakeConsoleEc2:
    def __init__(self):
        self.output = "cloud-init starting\n"

    def get_console_output(self, **_kwargs):
        return {"Output": self.output}


class FakeHealthEc2:
    def describe_instance_status(self, **kwargs):
        self.request = kwargs
        return {"InstanceStatuses": [{
            "InstanceStatus": {"Status": "ok"},
            "SystemStatus": {"Status": "initializing"},
        }]}


class EventuallyConsistentEc2:
    def __init__(self):
        self.calls = 0

    def describe_instances(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            error = RuntimeError("not propagated")
            error.response = {"Error": {"Code": "InvalidInstanceID.NotFound"}}
            raise error
        return {"Reservations": [{"Instances": [{"InstanceId": "i-eventual", "State": {"Name": "pending"}}]}]}


class ReleaseValidationTest(unittest.TestCase):
    def shard(self):
        return {
            "id": "linux-test", "instanceType": "c7i.24xlarge",
            "amiQuery": {
                "owners": ["099720109477"], "ownerIds": ["099720109477"],
                "name": "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*",
                "architecture": "x86_64",
            },
        }

    def test_resolve_ami_verifies_owner_architecture_and_storage(self):
        self.assertEqual("ami-verified", release.resolve_ami(FakeEc2(), FakeSsm(), self.shard()))

    def test_resolve_ami_rejects_wrong_owner(self):
        image = dict(FakeEc2().image, OwnerId="111111111111")
        with self.assertRaisesRegex(RuntimeError, "owned by"):
            release.resolve_ami(FakeEc2(image=image), FakeSsm(), self.shard())

    def test_matrix_rejects_instance_architecture_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            release.validate_launch_matrix(FakeEc2(architectures=["arm64"]), FakeSsm(), [self.shard()], "us-east-1")

    def test_matrix_rejects_unavailable_instance_type_in_zone(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            release.validate_launch_matrix(FakeEc2(offered=False), FakeSsm(), [self.shard()], "us-east-1", "us-east-1a")

    def test_just_launched_instance_is_retried_during_ec2_eventual_consistency(self):
        ec2 = EventuallyConsistentEc2()
        with patch.object(release.time, "sleep") as sleep:
            instance = release.describe_instance_eventually(ec2, "i-eventual")
        self.assertEqual("i-eventual", instance["InstanceId"])
        self.assertEqual(2, ec2.calls)
        sleep.assert_called_once_with(2)

    def test_checked_in_plan_has_verification_for_every_shard(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        self.assertTrue(plan["shards"])
        for shard in plan["shards"]:
            self.assertIn("amiQuery", shard)
            self.assertNotIn("amiSsmParameter", shard)

    def test_smoke_overrides_instance_and_build_threads(self):
        shard = self.shard()
        shard["build"] = {"buildThreads": 48}
        release.apply_execution_overrides([shard], "c7i.xlarge", 4)
        self.assertEqual("c7i.xlarge", shard["instanceType"])
        self.assertEqual(4, shard["build"]["buildThreads"])

    def test_smoke_override_rejects_accelerator_instances(self):
        with self.assertRaisesRegex(SystemExit, "invalid CPU compile"):
            release.apply_execution_overrides([self.shard()], "g6.xlarge", 4)

    def test_matrix_uses_serial_reusable_lanes(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        lanes = release.execution_shards(plan)
        self.assertEqual(len(plan["shards"]), len(lanes))
        linux = next(item for item in lanes if item["id"] == "linux-x86_64-cpu")
        self.assertEqual(7, len(linux["build"]["variants"]))
        self.assertNotIn("--base", linux["id"])

    def test_s3_compiler_cache_uses_a_stable_cross_run_namespace(self):
        plan = {"artifactPrefix": "deeplearning4j/releases"}
        cache = release.compiler_cache_config(plan, "release-bucket", "us-east-2")
        self.assertEqual({
            "backend": "s3",
            "bucket": "release-bucket",
            "region": "us-east-2",
            "keyPrefix": "deeplearning4j/releases/compiler-cache/v1",
        }, cache)
        self.assertNotIn("runId", cache["keyPrefix"])

    def test_instance_role_can_read_and_publish_sccache_objects(self):
        class ClientError(Exception):
            pass

        class Iam:
            def __init__(self):
                self.policy = None

            def get_role(self, **_kwargs):
                return {"Role": {}}

            def put_role_policy(self, **kwargs):
                self.policy = json.loads(kwargs["PolicyDocument"])

            def get_instance_profile(self, **_kwargs):
                return {"InstanceProfile": {"Roles": [{"RoleName": "DL4JReleaseBuilderRole"}]}}

        iam = Iam()
        with patch.object(release, "_boto3", return_value=(object(), ClientError)):
            release.ensure_instance_profile(iam, "release-bucket", "kill", "/logs")
        s3_actions = set(iam.policy["Statement"][0]["Action"])
        self.assertTrue({"s3:GetObject", "s3:PutObject", "s3:ListBucket"}.issubset(s3_actions))

    def test_shared_driver_configures_two_level_remote_sccache_backends(self):
        settings = {
            "s3": {
                "backend": "s3", "bucket": "s3-bucket", "region": "us-east-2",
                "keyPrefix": "cache/v1",
            },
            "gcs": {
                "backend": "gcs", "bucket": "gcs-bucket", "keyPrefix": "cache/v1",
            },
            "azure": {
                "backend": "azure", "container": "releases", "keyPrefix": "cache/v1",
                "connectionString": "BlobEndpoint=https://account/;SharedAccessSignature=token",
            },
        }
        for backend, remote in settings.items():
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temp:
                source = Path(temp) / "source"
                source.mkdir()
                environment = {"PATH": "/existing/bin"}
                with patch.object(
                    build_platform, "ensure_sccache", return_value="/tools/sccache"
                ), patch.object(build_platform, "run") as execute:
                    executable, started = build_platform.configure_compiler_cache(
                        {"compilerCache": remote}, source, environment
                    )
                self.assertEqual("/tools/sccache", executable)
                self.assertTrue(started)
                self.assertEqual(f"disk,{backend}", environment["SCCACHE_MULTILEVEL_CHAIN"])
                self.assertEqual("all", environment["SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY"])
                self.assertEqual(str(source.resolve()), environment["SCCACHE_BASEDIRS"])
                self.assertEqual("1", environment["SD_USE_SCCACHE"])
                self.assertEqual(
                    ["/tools", "/existing/bin"], environment["PATH"].split(os.pathsep)
                )
                for launcher in (
                    "CMAKE_C_COMPILER_LAUNCHER",
                    "CMAKE_CXX_COMPILER_LAUNCHER",
                    "CMAKE_CUDA_COMPILER_LAUNCHER",
                ):
                    self.assertEqual("/tools/sccache", environment[launcher])
                execute.assert_called_once_with(
                    ["/tools/sccache", "--start-server"], source, environment
                )
        self.assertEqual("s3-bucket", settings["s3"]["bucket"])

    def test_shared_driver_activates_local_sccache_but_not_ccache(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()

            sccache_env = {"PATH": "/existing/bin"}
            with patch.object(
                build_platform.shutil,
                "which",
                side_effect=lambda name: "/tools/sccache" if name == "sccache" else None,
            ), patch.object(build_platform, "run") as execute:
                executable, started = build_platform.configure_compiler_cache(
                    {}, source, sccache_env
                )
            self.assertEqual("/tools/sccache", executable)
            self.assertTrue(started)
            self.assertEqual("1", sccache_env["SD_USE_SCCACHE"])
            self.assertEqual(
                ["/tools", "/existing/bin"], sccache_env["PATH"].split(os.pathsep)
            )
            execute.assert_called_once_with(
                ["/tools/sccache", "--start-server"], source, sccache_env
            )

            ccache_env = {"PATH": "/existing/bin"}
            with patch.object(
                build_platform.shutil,
                "which",
                side_effect=lambda name: "/tools/ccache" if name == "ccache" else None,
            ), patch.object(build_platform, "run") as execute:
                executable, started = build_platform.configure_compiler_cache(
                    {}, source, ccache_env
                )
            self.assertEqual("/tools/ccache", executable)
            self.assertFalse(started)
            self.assertNotIn("SD_USE_SCCACHE", ccache_env)
            self.assertEqual("/existing/bin", ccache_env["PATH"])
            execute.assert_called_once_with(
                ["/tools/ccache", "--zero-stats"], source, ccache_env
            )

    def test_shared_driver_rejects_incomplete_remote_cache_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "compilerCache.region"):
                with patch.object(build_platform, "ensure_sccache", return_value="sccache"):
                    build_platform.configure_compiler_cache(
                        {"compilerCache": {
                            "backend": "s3", "bucket": "bucket", "keyPrefix": "cache/v1",
                        }},
                        source,
                        {},
                    )

    def test_expanded_smoke_selector_keeps_single_variant(self):
        plan = release.load_plan(Path(__file__).with_name("release-plan.json"))
        selected = release.selected_executions(plan, ["linux-x86_64-cpu--base"])
        self.assertEqual(1, len(selected))
        self.assertEqual("linux-x86_64-cpu--base", selected[0]["id"])
        self.assertEqual(["base"], [item["name"] for item in selected[0]["build"]["variants"]])

    def test_core_constraint_greedily_selects_largest_feasible_size(self):
        lane = self.shard()
        lane["build"] = {"buildThreads": 48, "mavenHeapGiB": 32}
        schedule = release.apply_core_constraint(SizingEc2(), [lane], 5)
        self.assertEqual("c7i.xlarge", lane["instanceType"])
        self.assertEqual(4, lane["build"]["buildThreads"])
        self.assertEqual(4, lane["build"]["mavenHeapGiB"])
        self.assertEqual(4, schedule[0]["selectedVcpus"])

    def test_core_constraint_fails_before_launch_when_no_size_fits(self):
        lane = self.shard()
        lane["build"] = {"buildThreads": 48, "mavenHeapGiB": 32}
        with self.assertRaisesRegex(SystemExit, "Core constraint is infeasible"):
            release.apply_core_constraint(SizingEc2(), [lane], 1)

    def test_s3_log_purge_deletes_only_selected_build_log_versions(self):
        s3 = FakeS3([{
            "Versions": [
                {"Key": "deeplearning4j/releases/run-a/linux-cpu--base/build.log", "VersionId": "v1"},
                {"Key": "deeplearning4j/releases/run-a/linux-cpu/status.json", "VersionId": "v2"},
                {"Key": "deeplearning4j/releases/run-a/windows-cpu/build.log", "VersionId": "v3"},
            ],
            "DeleteMarkers": [
                {"Key": "deeplearning4j/releases/run-a/linux-cpu--base/build.log", "VersionId": "d1"},
            ],
        }])
        deleted = release.delete_s3_log_objects(
            s3, "release-bucket", "deeplearning4j/releases/run-a/", {"linux-cpu"},
        )
        self.assertEqual(["v1", "d1"], [item["VersionId"] for item in deleted])
        self.assertEqual("deeplearning4j/releases/run-a/", s3.paginator.requests[0]["Prefix"])
        request_objects = s3.deletions[0]["Delete"]["Objects"]
        self.assertEqual(deleted, request_objects)
        self.assertFalse(any(item["Key"].endswith("status.json") for item in request_objects))
        self.assertTrue(release.shard_selected("linux-cpu--avx2", {"linux-cpu"}))
        self.assertFalse(release.shard_selected("windows-cpu", {"linux-cpu"}))

    def test_s3_log_purge_batches_at_aws_limit(self):
        versions = [
            {"Key": f"deeplearning4j/releases/run-{index}/linux-cpu/build.log", "VersionId": str(index)}
            for index in range(1001)
        ]
        s3 = FakeS3([{"Versions": versions}])
        deleted = release.delete_s3_log_objects(s3, "release-bucket", "deeplearning4j/releases/")
        self.assertEqual(1001, len(deleted))
        self.assertEqual([1000, 1], [len(call["Delete"]["Objects"]) for call in s3.deletions])

    def test_controller_event_is_published_without_worker_bootstrap(self):
        logs = FakeLogs()
        self.assertTrue(release.emit_cloudwatch_event(logs, "/releases", "run/lane", "phase=launched"))
        self.assertEqual("/releases", logs.published[0]["logGroupName"])
        self.assertEqual("run/lane", logs.published[0]["logStreamName"])
        self.assertIn("[dl4j-controller] phase=launched", logs.published[0]["logEvents"][0]["message"])

    def test_stream_lane_logs_prints_events_and_reuses_cursor(self):
        logs = FakeLogs()
        output = StringIO()
        with redirect_stdout(output):
            token, count = release.stream_lane_logs(logs, "/releases", "run/lane")
            token, second_count = release.stream_lane_logs(logs, "/releases", "run/lane", token)
        self.assertEqual("cursor-2", token)
        self.assertEqual((1, 1), (count, second_count))
        self.assertEqual("cursor-1", logs.requests[1]["nextToken"])
        self.assertIn("[run/lane] bootstrap ready", output.getvalue())
        self.assertIn("[run/lane] building base", output.getvalue())

    def test_stream_console_output_only_prints_new_bytes(self):
        ec2 = FakeConsoleEc2()
        output = StringIO()
        with redirect_stdout(output):
            offset = release.stream_console_output(ec2, "i-test", 0)
            ec2.output += "worker downloaded\n"
            offset = release.stream_console_output(ec2, "i-test", offset)
        self.assertEqual(len(ec2.output), offset)
        self.assertEqual(1, output.getvalue().count("cloud-init starting"))
        self.assertEqual(1, output.getvalue().count("worker downloaded"))

    def test_instance_health_includes_aws_system_and_instance_checks(self):
        ec2 = FakeHealthEc2()
        self.assertEqual(("ok", "initializing"), release.instance_health(ec2, "i-test"))
        self.assertTrue(ec2.request["IncludeAllInstances"])

    def test_log_forwarder_supports_ubuntu_aws_cli_v1(self):
        with patch.object(log_forwarder.subprocess, "run") as run:
            log_forwarder.aws("us-east-1", "logs", "create-log-stream")
        command = run.call_args.args[0]
        self.assertNotIn("--no-cli-pager", command)
        self.assertEqual("", run.call_args.kwargs["env"]["AWS_PAGER"])

    def test_failed_lane_log_is_retrieved_from_s3(self):
        class LogS3:
            request = None

            def get_object(self, **kwargs):
                self.request = kwargs
                return {"Body": BytesIO(b"phase=rust-toolchain status=failed\nreal bootstrap error\n")}

        s3 = LogS3()
        output = StringIO()
        with redirect_stdout(output):
            retrieved = release.print_s3_build_log(
                s3, "release-bucket", "runs/run-1/lane/status.json", "lane"
            )
        self.assertTrue(retrieved)
        self.assertEqual("runs/run-1/lane/build.log", s3.request["Key"])
        self.assertEqual("bytes=-262144", s3.request["Range"])
        self.assertIn("real bootstrap error", output.getvalue())

    def test_native_lane_invokes_shared_github_script_before_cross_platform_script(self):
        build = {
            "javacppPlatform": "linux-x86_64", "backend": "cpu", "profiles": ["cpu", "sdx"],
            "modules": [":libnd4j"], "variants": [{"name": "base"}], "buildCrossPlatform": True,
        }
        shard = {"id": "linux-x86_64-cpu", "os": "linux", "build": build}
        events = []
        with patch.object(build_platform, "prepare_openblas"), \
                patch.object(build_platform, "run", side_effect=lambda command, *_args: events.append(Path(command[1]).name)), \
                patch.object(build_platform, "build_cross_platform", side_effect=lambda *_args: events.append("cross-platform.sh")):
            build_platform.build_native_platform(Path("/source"), shard, Path("/m2"), {}, None)
        self.assertEqual(["linux-x86_64.sh", "cross-platform.sh"], events)

    def test_every_covered_workflow_uses_a_shared_release_executor(self):
        repository_root = Path(__file__).resolve().parents[2]
        plan = json.loads((Path(__file__).with_name("release-plan.json")).read_text(encoding="utf-8"))
        shared = (
            "build-scripts/release/linux-x86_64.sh",
            "build-scripts/release/cross-platform.sh",
            "build-scripts/release/native-platform.sh",
        )
        missing = []
        for relative_path in plan["coveredWorkflows"]:
            workflow_path = Path(relative_path)
            if workflow_path.parent == Path("."):
                workflow_path = Path(".github/workflows") / workflow_path
            workflow = (repository_root / workflow_path).read_text(encoding="utf-8")
            if not any(script in workflow for script in shared):
                missing.append(relative_path)
        self.assertEqual([], missing)

    def test_every_accelerator_backend_is_profile_gated_like_cuda(self):
        repository_root = Path(__file__).resolve().parents[2]
        pom = ET.parse(repository_root / "nd4j/nd4j-backends/nd4j-backend-impls/pom.xml").getroot()
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        direct_modules = {node.text for node in pom.findall("m:modules/m:module", namespace)}
        expected_profiles = {
            "cuda": {"nd4j-cuda", "nd4j-cuda-preset", "nd4j-cuda-platform"},
            "metal": {"nd4j-metal", "nd4j-metal-preset"},
            "tpu": {"nd4j-tpu", "nd4j-tpu-preset"},
            "hexagon": {"nd4j-hexagon", "nd4j-hexagon-preset"},
            "vulkan": {"nd4j-vulkan", "nd4j-vulkan-preset", "nd4j-vulkan-platform"},
            "zluda": {"nd4j-zluda"},
            "zluda-amd": {"nd4j-zluda"},
            "zluda-intel": {"nd4j-zluda"},
        }
        profiles = {}
        for profile in pom.findall("m:profiles/m:profile", namespace):
            profile_id = profile.findtext("m:id", namespaces=namespace)
            modules = {node.text for node in profile.findall("m:modules/m:module", namespace)}
            active_by_default = profile.findtext("m:activation/m:activeByDefault", namespaces=namespace)
            profiles[profile_id] = (modules, active_by_default)

        accelerator_modules = set().union(*expected_profiles.values())
        self.assertTrue(direct_modules.isdisjoint(accelerator_modules))
        for profile_id, expected_modules in expected_profiles.items():
            self.assertIn(profile_id, profiles)
            self.assertEqual(expected_modules, profiles[profile_id][0])
            self.assertEqual("false", profiles[profile_id][1])

    def test_no_pom_has_duplicate_top_level_profiles_sections(self):
        repository_root = Path(__file__).resolve().parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        tracked_files = subprocess.run(
            ["git", "ls-files"], cwd=repository_root, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        duplicates = []
        for relative_path in tracked_files:
            if Path(relative_path).name != "pom.xml":
                continue
            pom = ET.parse(repository_root / relative_path).getroot()
            count = len(pom.findall("m:profiles", namespace))
            if count > 1:
                duplicates.append(f"{relative_path} ({count})")
        self.assertEqual([], duplicates)

    def test_shared_variant_names_preserve_workflow_matrix_semantics(self):
        self.assertEqual("mps-compile", build_platform.shared_variant_helper({"name": "mps-compile", "helper": "mps", "mlir": True}))
        self.assertEqual("compile-nnapi", build_platform.shared_variant_helper({"name": "compile-nnapi", "mlir": True}))
        self.assertEqual("compile", build_platform.shared_variant_helper({"name": "compile", "mlir": True}))
        self.assertEqual("compile", build_platform.shared_variant_helper({"name": "compile", "triton": True}))

    def test_zluda_release_contract_is_explicit_for_every_cloud_plan(self):
        root = Path(__file__).parents[2]
        expected = {
            "linux-x86_64-zluda": ("linux", "linux-x86_64"),
            "windows-x86_64-zluda": ("windows", "windows-x86_64"),
        }
        for provider in ("aws", "gcp", "azure"):
            plan = json.loads((root / f"release/{provider}/release-plan.json").read_text(encoding="utf-8"))
            for shard_id, (operating_system, platform) in expected.items():
                with self.subTest(provider=provider, shard=shard_id):
                    shard = next(item for item in plan["shards"] if item["id"] == shard_id)
                    self.assertEqual(operating_system, shard["os"])
                    build = shard["build"]
                    rules = shard["artifactRules"]
                    self.assertEqual(platform, build["javacppPlatform"])
                    self.assertEqual("12.9", build["cudaVersion"])
                    self.assertEqual("v6", build["zludaVersion"])
                    self.assertEqual(["cuda", "sdx", "zluda"], build["profiles"])
                    self.assertEqual(
                        {":nd4j-cuda-12.9", ":nd4j-cuda-12.9-preset", ":nd4j-zluda", ":libnd4j"},
                        set(build["modules"]),
                    )
                    self.assertEqual([{
                        "name": "zluda",
                        "classifierSuffix": "-cuda-12.9-zluda",
                        "platformExtension": "-zluda",
                    }], build["variants"])
                    self.assertIn("-Dlibnd4j.zluda=AMD", build["mavenArgs"])
                    self.assertNotIn("-Dlibnd4j.zluda=rocm6", build["mavenArgs"])
                    self.assertEqual(
                        {"nd4j-cuda-12.9", "nd4j-cuda-12.9-preset", "nd4j-zluda"},
                        set(rules["artifactIds"]),
                    )
                    self.assertEqual(
                        [f"{platform}-cuda-12.9-zluda"],
                        shard["artifactRules"]["classifierTokens"],
                    )
                    expected_unclassified = ["nd4j-zluda"] if operating_system == "linux" else []
                    self.assertEqual(expected_unclassified, rules.get("unclassifiedArtifactIds", []))

    def test_zluda_cmake_runtime_contract_is_registered(self):
        root = Path(__file__).parents[2]
        cmake_source = (root / "libnd4j/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("NAME zluda_windows_runtime_contract", cmake_source)
        self.assertIn("cmake/tests/ZludaWindowsRuntimeContractTest.cmake", cmake_source)

    def test_classifier_staging_keeps_only_explicit_unclassified_zluda_runtime(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root / "repository"
            output = root / "output"
            version = "1.0.0"
            artifacts = {
                "nd4j-zluda": [
                    f"nd4j-zluda-{version}.jar",
                    f"nd4j-zluda-{version}-sources.jar",
                    f"nd4j-zluda-{version}-wrong-classifier.jar",
                ],
                "nd4j-cuda-12.9": [
                    f"nd4j-cuda-12.9-{version}.jar",
                    f"nd4j-cuda-12.9-{version}-linux-x86_64-cuda-12.9-zluda.jar",
                ],
            }
            for artifact_id, names in artifacts.items():
                directory = repository / "org/nd4j" / artifact_id / version
                directory.mkdir(parents=True)
                for name in names:
                    (directory / name).write_bytes(b"jar")

            build_platform.stage_repository(repository, output, {
                "mode": "classifier",
                "artifactIds": list(artifacts),
                "classifierTokens": ["linux-x86_64-cuda-12.9-zluda"],
                "unclassifiedArtifactIds": ["nd4j-zluda"],
                "includeMetadata": False,
            })

            staged = {path.name for path in output.rglob("*.jar")}
            self.assertIn(f"nd4j-zluda-{version}.jar", staged)
            self.assertIn(f"nd4j-cuda-12.9-{version}-linux-x86_64-cuda-12.9-zluda.jar", staged)
            self.assertNotIn(f"nd4j-cuda-12.9-{version}.jar", staged)
            self.assertNotIn(f"nd4j-zluda-{version}-sources.jar", staged)
            self.assertNotIn(f"nd4j-zluda-{version}-wrong-classifier.jar", staged)
            with self.assertRaisesRegex(ValueError, "must be a subset"):
                build_platform.stage_repository(repository, output, {
                    "mode": "classifier",
                    "artifactIds": ["nd4j-cuda-12.9"],
                    "unclassifiedArtifactIds": ["nd4j-zluda"],
                })

    def test_zluda_target_and_attestation_fail_closed(self):
        build = {
            "backend": "cuda",
            "cudaVersion": "12.9",
            "zludaVersion": "v6",
            "javacppPlatform": "linux-x86_64",
            "profiles": ["cuda", "sdx", "zluda"],
            "modules": [":nd4j-cuda-12.9", ":nd4j-zluda"],
            "mavenArgs": ["-Dlibnd4j.zluda=AMD"],
            "variants": [{
                "name": "zluda",
                "classifierSuffix": "-cuda-12.9-zluda",
                "platformExtension": "-zluda",
            }],
        }
        for platform, runtime_names in (
                ("linux-x86_64", ("libcuda.so",)),
                ("windows-x86_64", build_platform.ZLUDA_WINDOWS_REQUIRED_FILES)):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as zluda_path:
                for runtime_name in runtime_names:
                    Path(zluda_path, runtime_name).write_bytes(b"runtime")
                output = StringIO()
                platform_build = dict(build, javacppPlatform=platform)
                with redirect_stdout(output):
                    build_platform.attest_zluda_configuration(
                        platform_build, {"ZLUDA_PATH": zluda_path}
                    )
                self.assertIn("target=AMD", output.getvalue())
                self.assertIn(runtime_names[0], output.getvalue())

        for arguments in ([], ["-Dlibnd4j.zluda=rocm6"], [
                "-Dlibnd4j.zluda=AMD", "-Dlibnd4j.zluda=AMD"]):
            with self.subTest(arguments=arguments):
                invalid = dict(build, mavenArgs=arguments)
                with self.assertRaises(ValueError):
                    build_platform.zluda_target(invalid)
        with self.assertRaisesRegex(RuntimeError, "prepared ZLUDA_PATH is missing"):
            build_platform.attest_zluda_configuration(build, {})
        with tempfile.TemporaryDirectory() as empty_zluda_path:
            with self.assertRaisesRegex(RuntimeError, "contains no linux runtime"):
                build_platform.attest_zluda_configuration(
                    build, {"ZLUDA_PATH": empty_zluda_path}
                )

    def test_zluda_download_retries_transient_open_failure(self):
        request = build_platform.urllib.request.Request("https://example.invalid/zluda")
        response = BytesIO(b"payload")
        with patch.object(
                build_platform.urllib.request,
                "urlopen",
                side_effect=[urllib.error.URLError("temporary"), response],
        ) as urlopen, patch.object(build_platform.time, "sleep") as sleep:
            with build_platform.urlopen_with_retry(request, "ZLUDA test asset") as opened:
                self.assertEqual(b"payload", opened.read())
        self.assertEqual(2, urlopen.call_count)
        sleep.assert_called_once_with(1.0)

    def test_prepare_zluda_selects_and_validates_windows_asset(self):
        release_metadata = BytesIO(json.dumps({
            "assets": [
                {"name": "zluda-linux.tar.gz", "browser_download_url": "https://example/linux"},
                {"name": "zluda-windows.zip", "browser_download_url": "https://example/windows"},
            ]
        }).encode("utf-8"))

        def write_windows_archive(url, destination, description):
            self.assertEqual("https://example/windows", url)
            self.assertIn("windows", description)
            with build_platform.zipfile.ZipFile(destination, "w") as bundle:
                for name in build_platform.ZLUDA_WINDOWS_REQUIRED_FILES:
                    bundle.writestr(f"zluda/{name}", b"runtime")
                bundle.writestr("zluda/trace/nvcuda.dll", b"trace")

        with tempfile.TemporaryDirectory() as temporary_directory, \
                patch.object(build_platform, "urlopen_with_retry", return_value=release_metadata), \
                patch.object(build_platform, "download_with_retry", side_effect=write_windows_archive):
            environment = {"PATH": "existing-search-path"}
            build_platform.prepare_zluda(
                Path(temporary_directory),
                {"zludaVersion": "v6", "javacppPlatform": "windows-x86_64"},
                environment,
            )
            runtime_directory = Path(temporary_directory, "zluda", "zluda")
            self.assertEqual(runtime_directory, Path(environment["ZLUDA_PATH"]))
            self.assertEqual(
                [str(runtime_directory), "existing-search-path"],
                environment["PATH"].split(os.pathsep),
            )
            child_environment = os.environ.copy()
            child_environment.update(environment)
            child = subprocess.run(
                [sys.executable, "-c", "import os; print(os.environ['PATH'])"],
                check=True,
                capture_output=True,
                text=True,
                env=child_environment,
            )
            self.assertEqual(str(runtime_directory), child.stdout.strip().split(os.pathsep)[0])

    def test_buildnativeoperations_rejects_unknown_zluda_target(self):
        root = Path(__file__).parents[2]
        result = subprocess.run(
            ["bash", str(root / "libnd4j/buildnativeoperations.sh"), "--zluda", "rocm6"],
            cwd=root / "libnd4j",
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("Unsupported --zluda target 'ROCM6'", result.stdout + result.stderr)

    def test_shared_native_script_emits_specialized_classifiers(self):
        root = Path(__file__).parents[2]
        script = root / "build-scripts/release/native-platform.sh"

        def command(**environment):
            process_environment = {"PATH": "/usr/bin:/bin", "DL4J_BUILD_THREADS": "16", "DL4J_MVN_FLAGS": ""}
            process_environment.update(environment)
            result = subprocess.run(["bash", str(script), "--print"], env=process_environment,
                                    check=True, capture_output=True, text=True)
            return shlex.split(result.stdout)

        cuda = command(DL4J_FAMILY="linux-cuda", DL4J_CUDA_VERSION="12.9", DL4J_HELPER="compile")
        self.assertIn("-Dlibnd4j.classifier=linux-x86_64-cuda-12.9-compile", cuda)
        self.assertIn("-Djavacpp.platform.extension=-compile", cuda)

        metal = command(DL4J_FAMILY="macos-arm64", DL4J_HELPER="mps")
        self.assertIn("-Pmetal", metal)

        for family in ("tpu", "hexagon", "vulkan"):
            accelerator = command(DL4J_FAMILY=family)
            self.assertIn(f"-P{family}", accelerator)

        vulkan = command(DL4J_FAMILY="vulkan-mlir")
        self.assertIn("-Pvulkan", vulkan)
        self.assertIn("-Dplatform.classifier=linux-x86_64-compile", vulkan)

        zluda = command(DL4J_FAMILY="zluda")
        self.assertIn("-Dlibnd4j.zluda=AMD", zluda)
        self.assertIn("-Dlibnd4j.classifier=linux-x86_64-cuda-12.9-zluda", zluda)
        self.assertIn("-Djavacpp.platform.extension=-zluda", zluda)
        self.assertIn("-Pzluda", zluda)
        self.assertEqual(":nd4j-cuda-12.9,:nd4j-cuda-12.9-preset,:nd4j-zluda,:libnd4j", zluda[zluda.index("-pl") + 1])

        windows_zluda = command(DL4J_FAMILY="windows-zluda", DL4J_ZLUDA_TARGET="AMD")
        self.assertIn("-Dlibnd4j.classifier=windows-x86_64-cuda-12.9-zluda", windows_zluda)
        self.assertIn("-Djavacpp.platform=windows-x86_64", windows_zluda)
        self.assertIn("-Dlibnd4j.platform=windows-x86_64", windows_zluda)
        self.assertIn("-Dlibnd4j.oom.killer=OFF", windows_zluda)

    def test_zluda_native_family_tracks_worker_os(self):
        build = {"zludaVersion": "v6"}
        variant = {"name": "zluda"}
        self.assertEqual(
            "zluda",
            build_platform.shared_native_family({"os": "linux", "build": build}, variant),
        )
        self.assertEqual(
            "windows-zluda",
            build_platform.shared_native_family({"os": "windows", "build": build}, variant),
        )

    def test_aws_cuda_compile_variant_uses_workflow_compile_classifier_path(self):
        shard = {
            "id": "linux-x86_64-cuda-12-9", "os": "linux", "workloads": ["maven"],
            "build": {"backend": "cuda", "cudaVersion": "12.9", "javacppPlatform": "linux-x86_64",
                      "modules": [], "variants": [{"name": "compile", "triton": True}]},
        }
        calls = []
        with patch.object(build_platform, "prepare_openblas"), \
                patch.object(build_platform, "run", side_effect=lambda command, _cwd, env: calls.append((command, env))):
            build_platform.build_native_platform(Path("/source"), shard, Path("/m2"), {}, None)
        self.assertEqual("compile", calls[0][1]["DL4J_HELPER"])
        self.assertEqual("12.9", calls[0][1]["DL4J_CUDA_VERSION"])

    def test_aws_cross_platform_invokes_the_shared_workflow_script(self):
        calls = []
        build = {"javacppPlatform": "linux-x86_64"}
        with patch.object(build_platform, "run", side_effect=lambda command, _cwd, env: calls.append((command, env))):
            build_platform.build_cross_platform(Path("/source"), build, Path("/m2"), {})
        self.assertEqual(["--run-tokenizers", "--run-java"], [call[0][-1] for call in calls])
        self.assertTrue(all(Path(call[0][1]).name == "cross-platform.sh" for call in calls))
        self.assertTrue(all(call[1]["DL4J_MAVEN_GOAL"] == "install" for call in calls))

    def test_linux_cross_platform_java_command_does_not_infer_accelerator_profiles(self):
        root = Path(__file__).parents[2]
        environment = {
            "DL4J_PLATFORM": "linux-x86_64",
            "DL4J_OS": "linux",
            "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            ["bash", str(root / "build-scripts/release/cross-platform.sh"), "--print-java"],
            env=environment, check=True, capture_output=True, text=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual([], [argument for argument in command if argument.startswith("-P")])

    def test_update_versions_propagates_maven_failure(self):
        root = Path(__file__).parents[2]
        script = root / "update-versions.sh"
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            pom = project / "pom.xml"
            pom.write_text("<project><version>1.0.0-SNAPSHOT</version></project>\n", encoding="utf-8")
            bin_directory = project / "bin"
            bin_directory.mkdir()
            fake_maven = bin_directory / "mvn"
            fake_maven.write_text("#!/usr/bin/env bash\nexit 23\n", encoding="utf-8")
            fake_maven.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{bin_directory}:{environment['PATH']}"
            result = subprocess.run(
                ["bash", str(script), "1.0.0-SNAPSHOT", "1.0.0"],
                cwd=project, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(23, result.returncode)
            self.assertIn("<version>1.0.0-SNAPSHOT</version>", pom.read_text(encoding="utf-8"))

    def test_shared_linux_script_emits_the_workflow_command(self):
        root = Path(__file__).parents[2]
        environment = {
            "DL4J_HELPER": "", "DL4J_EXTENSION": "", "DL4J_LIBND4J_FILE_DOWNLOAD": "",
            "DL4J_BUILD_THREADS": "16", "DL4J_MATRIX_MVN_EXT": "", "PATH": "/usr/bin:/bin",
        }
        result = subprocess.run(
            ["bash", str(root / "build-scripts/release/linux-x86_64.sh"), "--print"],
            env=environment, check=True, capture_output=True, text=True,
        )
        command = shlex.split(result.stdout)
        self.assertEqual("mvn", command[0])
        self.assertIn("-X", command)
        self.assertEqual(":nd4j-native,:nd4j-native-preset,:libnd4j", command[command.index("-pl") + 1])
        self.assertEqual("deploy", command[-2])
        self.assertEqual("-DskipTests", command[-1])

    def test_github_and_aws_reference_the_same_release_scripts(self):
        root = Path(__file__).parents[2]
        linux_workflow = (root / ".github/workflows/build-deploy-linux-x86_64.yml").read_text(encoding="utf-8")
        cross_workflow = (root / ".github/workflows/build-deploy-cross-platform.yml").read_text(encoding="utf-8")
        driver = (root / "release/aws/build-platform.py").read_text(encoding="utf-8")
        self.assertIn("build-scripts/release/linux-x86_64.sh --print", linux_workflow)
        self.assertIn("build-scripts/release/cross-platform.sh --print-tokenizers", cross_workflow)
        self.assertIn("build-scripts/release/cross-platform.sh --print-java", cross_workflow)
        self.assertIn('script_name = "linux-x86_64.sh"', driver)
        self.assertIn('source / "build-scripts/release/cross-platform.sh"', driver)
        self.assertIn('else "native-platform.sh"', driver)

    def test_omnihub_pipeline_dependencies_are_in_the_shared_java_reactor(self):
        root = Path(__file__).parents[2]
        namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
        nd4j_pom = ET.parse(root / "nd4j/pom.xml").getroot()
        reactor_modules = {
            module.text for module in nd4j_pom.findall("./m:modules/m:module", namespace)
        }
        omnihub_pom = ET.parse(root / "omnihub/pom.xml").getroot()
        pipeline_dependencies = {
            dependency.find("m:artifactId", namespace).text
            for dependency in omnihub_pom.findall("./m:dependencies/m:dependency", namespace)
            if dependency.find("m:artifactId", namespace) is not None
            and dependency.find("m:artifactId", namespace).text.startswith("samediff-pipeline-")
        }

        self.assertIn("samediff-pipeline-ggml", pipeline_dependencies)
        self.assertTrue(
            pipeline_dependencies.issubset(reactor_modules),
            f"OmniHub pipeline dependencies missing from nd4j reactor: {pipeline_dependencies - reactor_modules}",
        )
        for module in pipeline_dependencies:
            self.assertTrue((root / "nd4j" / module / "pom.xml").is_file(), module)

        service = root / "nd4j/samediff-pipeline-ggml/src/main/resources/META-INF/services" \
            / "org.eclipse.deeplearning4j.pipeline.PipelineLoader"
        self.assertEqual(
            "org.eclipse.deeplearning4j.ggml.GGMLPipelineLoader",
            service.read_text(encoding="utf-8").strip(),
        )

    def test_bootstrap_and_workers_emit_durable_lifecycle_phases(self):
        bootstrap = release.bootstrap_user_data("linux", "https://example.invalid/worker")
        self.assertIn("phase=cloud-init status=started", bootstrap)
        self.assertIn("phase=worker-download status=complete", bootstrap)
        self.assertIn("export HOME=${HOME:-/root}", bootstrap)
        self.assertLess(bootstrap.index("export HOME="), bootstrap.index("curl --fail"))
        root = Path(__file__).parent
        linux = (root / "worker.sh").read_text(encoding="utf-8")
        windows = (root / "worker.ps1").read_text(encoding="utf-8")
        for phase in ("logging-prerequisites", "toolchain-packages", "source-checkout", "matrix-build", "artifact-packaging", "finalize"):
            self.assertIn(phase, linux)
            self.assertIn(phase, windows)
        self.assertLess(linux.index("start_log_forwarder\n"), linux.index("phase toolchain-packages started"))
        self.assertLess(windows.index("Start-LogForwarder\n"), windows.index("Write-Phase 'toolchain-packages' 'started'"))
        self.assertLess(linux.index("export HOME="), linux.index("CONFIG_B64="))
        self.assertLess(linux.index("export CARGO_HOME="), linux.index("rust-toolchain started"))
        self.assertIn('SCCACHE_ROOT=${WORK_ROOT}/sccache', linux)
        self.assertIn('${SCCACHE_ROOT}:/sccache', linux)
        self.assertIn('${SCCACHE_ROOT}:/github/sccache', linux)
        self.assertNotIn('${HOME}/.cargo', linux)
        self.assertLess(windows.index("$env:CARGO_HOME ="), windows.index("rustup toolchain install"))
        self.assertNotIn("$env:USERPROFILE", windows)

    def test_unix_worker_initializes_cloud_init_environment_under_strict_mode(self):
        worker = (Path(__file__).parent / "worker.sh").read_text(encoding="utf-8")
        environment_prefix = worker.split("CONFIG_B64=", 1)[0]
        probe = environment_prefix + "printf '%s\\n' \"$HOME|$USER|$LOGNAME|$PATH\""
        result = subprocess.run(
            ["bash", "-u", "-c", probe], env={}, check=True, capture_output=True, text=True
        )
        home, user, logname, path = result.stdout.strip().split("|", 3)
        self.assertEqual("/root", home)
        self.assertTrue(user)
        self.assertEqual(user, logname)
        self.assertIn("/usr/bin", path)


class EnvironmentWizardTest(unittest.TestCase):
    class Credentials:
        def __init__(self, method):
            self.method = method

    class Sts:
        def __init__(self, usable):
            self.usable = usable

        def get_caller_identity(self):
            if not self.usable:
                raise AssertionError("STS must not run without resolved credentials")
            return {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"}

    class Session:
        available_profiles = ["release"]

        def __init__(self, **kwargs):
            self.region_name = kwargs.get("region_name") or os.environ.get("AWS_REGION")
            access_key = kwargs.get("aws_access_key_id") or os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = kwargs.get("aws_secret_access_key") or os.environ.get("AWS_SECRET_ACCESS_KEY")
            profile = kwargs.get("profile_name") or os.environ.get("AWS_PROFILE")
            self.method = "env" if access_key and secret_key else ("shared-credentials-file" if profile == "release" else None)

        def get_credentials(self):
            return EnvironmentWizardTest.Credentials(self.method) if self.method else None

        def client(self, name):
            if name == "sts":
                return EnvironmentWizardTest.Sts(bool(self.method))
            return object()

    class Boto3:
        @staticmethod
        def Session(**kwargs):
            return EnvironmentWizardTest.Session(**kwargs)

    def test_ci_never_enables_wizard_even_with_a_tty(self):
        tty = type("Tty", (), {"isatty": lambda self: True})()
        with patch.dict(os.environ, {"CI": "true"}, clear=True), patch.object(release.sys, "stdin", tty):
            self.assertFalse(release.interactive_wizard_enabled(True))

    def test_region_validation_is_partition_agnostic(self):
        self.assertTrue(release.valid_aws_region("us-east-1"))
        self.assertTrue(release.valid_aws_region("eusc-de-east-1"))
        self.assertFalse(release.valid_aws_region("global"))

    def test_noninteractive_missing_configuration_fails_with_recovery_command(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            self.assertRaisesRegex(SystemExit, "release/aws/release.py configure"),
        ):
            release.session_clients(allow_wizard=False)

    def test_existing_standard_environment_does_not_prompt(self):
        environment = {
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "AKIATEST",
            "AWS_SECRET_ACCESS_KEY": "already-configured",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch("builtins.input", side_effect=AssertionError("wizard unexpectedly prompted")),
        ):
            session, region, *_ = release.session_clients(allow_wizard=True)
        self.assertEqual("us-east-1", region)
        self.assertEqual("env", session.get_credentials().method)

    def test_existing_standard_profile_does_not_prompt(self):
        environment = {"AWS_REGION": "us-east-1", "AWS_PROFILE": "release"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch("builtins.input", side_effect=AssertionError("wizard unexpectedly prompted")),
        ):
            session, _, *_ = release.session_clients(allow_wizard=True)
        self.assertEqual("shared-credentials-file", session.get_credentials().method)

    def test_wizard_sets_region_and_hidden_access_credentials_for_current_command(self):
        stderr = StringIO()
        stdout = StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch.object(release, "interactive_wizard_enabled", return_value=True),
            patch("builtins.input", side_effect=["us-east-1", "AKIAWIZARD"]),
            patch.object(release.getpass, "getpass", side_effect=["never-print-this", ""]) as secret_input,
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            session, region, *_ = release.session_clients()
            self.assertNotIn("AWS_ACCESS_KEY_ID", os.environ)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", os.environ)
            self.assertNotIn("AWS_PROFILE", os.environ)
            self.assertEqual("us-east-1", os.environ["AWS_REGION"])
        self.assertEqual("env", session.get_credentials().method)
        self.assertEqual("us-east-1", region)
        self.assertNotIn("never-print-this", stderr.getvalue())
        self.assertIn("AWS_REGION", stderr.getvalue())
        self.assertIn("AWS_ACCESS_KEY_ID", stderr.getvalue())
        self.assertIn("AWS_SECRET_ACCESS_KEY", secret_input.call_args_list[0].args[0])
        self.assertIn("AWS_SESSION_TOKEN", secret_input.call_args_list[1].args[0])
        self.assertEqual("", stdout.getvalue())

    def test_rejected_profile_prompts_directly_for_access_key(self):
        with (
            patch.dict(os.environ, {"AWS_REGION": "us-east-1", "AWS_PROFILE": "broken"}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch.object(release, "interactive_wizard_enabled", return_value=True),
            patch("builtins.input", side_effect=["AKIADIRECT"]) as user_input,
            patch.object(release.getpass, "getpass", side_effect=["direct-secret", ""]),
            redirect_stderr(StringIO()) as stderr,
        ):
            session, _, *_ = release.session_clients()
            self.assertNotIn("AWS_PROFILE", os.environ)
        self.assertEqual("env", session.get_credentials().method)
        self.assertEqual(1, user_input.call_count)
        self.assertIn("AWS_PROFILE", stderr.getvalue())
        self.assertIn("AWS_ACCESS_KEY_ID", stderr.getvalue())

    def test_rejected_entered_credentials_exit_instead_of_looping(self):
        with (
            patch.dict(os.environ, {"AWS_REGION": "us-east-1", "AWS_PROFILE": "broken"}, clear=True),
            patch.object(release, "_boto3", return_value=(self.Boto3, Exception)),
            patch.object(release, "interactive_wizard_enabled", return_value=True),
            patch.object(
                release, "aws_credential_problem",
                return_value="credential validation returned InvalidClientTokenId",
            ),
            patch("builtins.input", return_value="AKIAREJECTED") as user_input,
            patch.object(release.getpass, "getpass", side_effect=["rejected-secret", ""]),
            redirect_stderr(StringIO()),
            self.assertRaisesRegex(SystemExit, "AWS rejected the values entered"),
        ):
            release.session_clients()
        self.assertEqual(1, user_input.call_count)

    def test_configure_and_no_wizard_options_parse(self):
        args = release.parser().parse_args(["--no-wizard", "configure"])
        self.assertEqual("configure", args.command)
        self.assertTrue(args.no_wizard)


if __name__ == "__main__":
    unittest.main()
