#!/usr/bin/env python3
"""Unit tests for the release provisioner's fail-closed AWS validation."""

import importlib.util
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("dl4j_aws_release", Path(__file__).with_name("release.py"))
release = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release)


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


class FakeConsoleEc2:
    def __init__(self):
        self.output = "cloud-init starting\n"

    def get_console_output(self, **_kwargs):
        return {"Output": self.output}


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


if __name__ == "__main__":
    unittest.main()
