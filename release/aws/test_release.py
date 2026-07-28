#!/usr/bin/env python3
"""Unit tests for the release provisioner's fail-closed AWS validation."""

import importlib.util
import unittest
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
            "ImageId": "ami-verified", "Name": "ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-amd64-server-20260701",
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


class ReleaseValidationTest(unittest.TestCase):
    def shard(self):
        return {
            "id": "linux-test", "instanceType": "c7i.24xlarge",
            "amiQuery": {
                "owners": ["099720109477"], "ownerIds": ["099720109477"],
                "name": "ubuntu/images/hvm-ssd-gp3/ubuntu-jammy-22.04-amd64-server-*",
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


if __name__ == "__main__":
    unittest.main()
