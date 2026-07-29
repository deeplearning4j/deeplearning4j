#!/usr/bin/env python3
"""Unit tests for the release provisioner's fail-closed AWS validation."""

import importlib.util
import json
import shlex
import subprocess
import unittest
from contextlib import redirect_stdout
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

    def test_shared_variant_names_preserve_workflow_matrix_semantics(self):
        self.assertEqual("mps-compile", build_platform.shared_variant_helper({"name": "mps-compile", "helper": "mps", "mlir": True}))
        self.assertEqual("compile-nnapi", build_platform.shared_variant_helper({"name": "compile-nnapi", "mlir": True}))
        self.assertEqual("compile", build_platform.shared_variant_helper({"name": "compile", "mlir": True}))

    def test_aws_cross_platform_invokes_the_shared_workflow_script(self):
        calls = []
        build = {"javacppPlatform": "linux-x86_64"}
        with patch.object(build_platform, "run", side_effect=lambda command, _cwd, env: calls.append((command, env))):
            build_platform.build_cross_platform(Path("/source"), build, Path("/m2"), {})
        self.assertEqual(["--run-tokenizers", "--run-java"], [call[0][-1] for call in calls])
        self.assertTrue(all(Path(call[0][1]).name == "cross-platform.sh" for call in calls))
        self.assertTrue(all(call[1]["DL4J_MAVEN_GOAL"] == "install" for call in calls))

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


if __name__ == "__main__":
    unittest.main()
