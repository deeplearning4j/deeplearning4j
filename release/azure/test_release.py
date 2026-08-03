#!/usr/bin/env python3
"""Unit and contract tests for the Azure external release backend."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release = load_module("dl4j_azure_release", HERE / "release.py")
cloud_io = load_module("dl4j_azure_cloud_io", HERE / "cloud-io.py")


def fake_sku(
    name: str,
    *,
    architecture: str = "x64",
    vcpus: int = 32,
    memory: float = 64.0,
    family: str = "standardFSv2Family",
    location: str = "eastus2",
    zones: tuple[str, ...] = ("1", "2", "3"),
    restrictions=None,
):
    return {
        "name": name,
        "resource_type": "virtualMachines",
        "locations": [location],
        "family": family,
        "capabilities": [
            {"name": "CpuArchitectureType", "value": architecture},
            {"name": "vCPUs", "value": str(vcpus)},
            {"name": "MemoryGB", "value": str(memory)},
        ],
        "location_info": [{"location": location, "zones": list(zones)}],
        "restrictions": list(restrictions or []),
    }


class ReleasePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.azure = release.load_plan(HERE / "release-plan.json")
        cls.aws = json.loads(
            (ROOT / "release/aws/release-plan.json").read_text(encoding="utf-8")
        )
        cls.gcp = json.loads(
            (ROOT / "release/gcp/release-plan.json").read_text(encoding="utf-8")
        )

    def test_resource_names_are_opaque_and_reserved_word_safe(self):
        run_id = "azure-windows-microsoft-login-office"
        identity_name = release.resource_name(
            "dl4j-release-identity", run_id, maximum=64
        )
        self.assertEqual(
            identity_name,
            release.resource_name("dl4j-release-identity", run_id, maximum=64),
        )
        self.assertRegex(
            identity_name, r"^dl4j-release-identity-[0-9a-f]{16}$"
        )
        self.assertNotIn("azure", identity_name)
        self.assertNotIn("windows", identity_name)
        self.assertNotIn("microsoft", identity_name)
        self.assertNotIn("login", identity_name)
        self.assertNotIn("office", identity_name)

        computer_name = release.resource_name(
            "dl4j", run_id, "windows-x86-64-2022", maximum=15
        )
        self.assertRegex(computer_name, r"^dl4j-[0-9a-f]{10}$")
        self.assertLessEqual(len(computer_name), 15)
        self.assertNotEqual(
            computer_name,
            release.resource_name("dl4j", run_id, "other-lane", maximum=15),
        )

    def test_identity_name_hashes_full_epoch_and_preserves_readable_tags(self):
        run_id = "azure-windows-full-20260803-105253"
        epochs = [
            "123456789abc-first-controller",
            "123456789abc-second-controller",
        ]
        created = []

        def create_identity(group, name, parameters):
            created.append((name, parameters))
            return SimpleNamespace(
                id="/identity",
                client_id="client",
                principal_id="principal",
            )

        context = {
            "subscription": "subscription",
            "identity": SimpleNamespace(
                user_assigned_identities=SimpleNamespace(
                    create_or_update=create_identity
                )
            ),
            "authorization": SimpleNamespace(
                role_assignments=SimpleNamespace(create=mock.Mock())
            ),
        }
        for epoch in epochs:
            release.ensure_identity(
                context,
                "group",
                "eastus2",
                run_id,
                "/storage/scope",
                controller_epoch=epoch,
            )

        self.assertNotEqual(created[0][0], created[1][0])
        for epoch, (name, parameters) in zip(epochs, created):
            self.assertRegex(name, r"^dl4j-release-identity-[0-9a-f]{16}$")
            self.assertEqual(run_id, parameters["tags"][release.RUN_TAG])
            self.assertEqual(
                epoch, parameters["tags"][release.CONTROLLER_EPOCH_TAG]
            )

    def test_azure_covers_every_aws_lane_except_unavailable_macos(self):
        expected = {
            item["id"] for item in self.aws["shards"]
            if item["id"] != "macos-14-arm64-cpu"
        }
        self.assertEqual(expected, {item["id"] for item in self.azure["shards"]})
        self.assertEqual(
            {"build-deploy-mac-arm64.yml"},
            set(self.azure["unsupportedWorkflows"]),
        )

    def test_azure_and_gcp_cover_the_same_portable_build_lanes(self):
        self.assertEqual(
            {item["id"] for item in self.gcp["shards"]},
            {item["id"] for item in self.azure["shards"]},
        )

    def test_full_plan_has_four_explicit_vm_compatibility_lanes(self):
        lanes = {item.get("lane") for item in self.azure["shards"]}
        self.assertNotIn(None, lanes)
        self.assertEqual(
            {
                "linux-x86-64-jammy",
                "linux-x86-64-noble",
                "linux-arm64-jammy",
                "windows-x86-64-2022",
            },
            lanes,
        )

    def test_shard_contract_digest_covers_artifact_producing_fields(self):
        shard = self.azure["shards"][0]
        original = release.shard_contract_digest(shard)
        changed = json.loads(json.dumps(shard))
        changed["workloads"] = [*changed["workloads"], "sdk"]
        self.assertNotEqual(original, release.shard_contract_digest(changed))
        stamped = release.with_shard_contract_digest(shard)
        self.assertEqual(original, stamped["contractDigest"])
        self.assertNotIn("contractDigest", shard)

    def test_shared_matrix_contract_is_byte_for_byte_equivalent_to_aws(self):
        aws = {item["id"]: item for item in self.aws["shards"]}
        for shard in self.azure["shards"]:
            canonical = aws[shard["id"]]
            self.assertEqual(canonical["build"], shard["build"], shard["id"])
            self.assertEqual(canonical["workloads"], shard["workloads"], shard["id"])
            self.assertEqual(
                canonical["artifactRules"], shard["artifactRules"], shard["id"]
            )

    def test_azure_has_no_provider_specific_tpu_smoke(self):
        self.assertNotIn("tpuSmoke", self.azure)
        commands = {
            action.dest
            for action in release.parser()._actions
            if action.dest == "command"
        }
        self.assertEqual({"command"}, commands)
        with self.assertRaises(SystemExit):
            release.parser().parse_args(["tpu-smoke"])

    def test_images_are_explicit_and_architecture_specific(self):
        for shard in self.azure["shards"]:
            image = shard["image"]
            expected = "Arm64" if shard["architecture"] == "arm64" else "x64"
            self.assertEqual(expected, image["architecture"], shard["id"])
            self.assertTrue(image["publisher"])
            self.assertTrue(image["offer"])
            self.assertTrue(image["sku"])
            self.assertEqual("latest", image["version"])
        arm = next(
            item for item in self.azure["shards"]
            if item["id"] == "linux-arm64-cpu"
        )
        self.assertIn("arm64", arm["image"]["sku"])

    def test_plan_rejects_image_architecture_mismatches(self):
        for shard_architecture, image_architecture in (
            ("arm64", "x64"),
            ("x86_64", "Arm64"),
        ):
            with self.subTest(
                shard_architecture=shard_architecture,
                image_architecture=image_architecture,
            ), tempfile.TemporaryDirectory() as temp:
                plan = json.loads(json.dumps(self.azure))
                shard = plan["shards"][0]
                shard["architecture"] = shard_architecture
                shard["machineClass"] = "arm" if shard_architecture == "arm64" else "x86"
                shard["image"]["architecture"] = image_architecture
                path = Path(temp) / "plan.json"
                path.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "image architecture"):
                    release.load_plan(path)

    def test_plan_rejects_malformed_shard_contracts_cleanly(self):
        mutations = [
            lambda plan: plan.update({"shards": ["not-an-object"]}),
            lambda plan: plan["shards"][0].update({"worker": "worker.ps1"}),
            lambda plan: plan["shards"][0].update({"machineCandidates": "size"}),
            lambda plan: plan["shards"][0].update({"image": {"publisher": "p"}}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temp:
                plan = json.loads(json.dumps(self.azure))
                mutate(plan)
                path = Path(temp) / "plan.json"
                path.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaises(ValueError):
                    release.load_plan(path)

    def test_machine_candidates_are_cpu_only_and_have_arm64_capacity(self):
        x86 = self.azure["defaults"]["x86MachineCandidates"]
        arm = self.azure["defaults"]["armMachineCandidates"]
        self.assertTrue(all(name.startswith("Standard_F") for name in x86))
        self.assertIn("Standard_F64as_v7", x86)
        self.assertLess(x86.index("Standard_F64s_v2"), x86.index("Standard_F64as_v7"))
        self.assertLess(x86.index("Standard_F64as_v7"), x86.index("Standard_F48s_v2"))
        self.assertTrue(all("ps_v" in name for name in arm))
        self.assertIn("Standard_D96ps_v6", arm)
        self.assertFalse(any(token in name for name in x86 + arm for token in ("NC", "ND", "NV")))

    def test_compile_only_accelerators_do_not_request_accelerator_hardware(self):
        ids = {
            "linux-x86_64-cuda-12-6",
            "linux-x86_64-cuda-12-9",
            "linux-x86_64-vulkan",
            "linux-x86_64-vulkan-mlir",
            "linux-x86_64-hexagon",
            "linux-x86_64-tpu",
            "linux-x86_64-zluda",
            "windows-x86_64-zluda",
        }
        for shard in self.azure["shards"]:
            if shard["id"] in ids:
                self.assertEqual("x86", shard["machineClass"])
                self.assertNotIn("accelerator", shard)
                self.assertEqual("x86_64", shard["architecture"])

    def test_variants_have_unique_names(self):
        for shard in self.azure["shards"]:
            names = [item["name"] for item in shard["build"]["variants"]]
            self.assertEqual(len(names), len(set(names)), shard["id"])

    def test_azure_only_collection_is_incomplete_until_aws_macos_is_merged(self):
        expected = release.matrix_coverage(
            self.aws, [item["id"] for item in self.aws["shards"]]
        )
        azure_only = release.matrix_coverage(
            self.aws, [item["id"] for item in self.azure["shards"]]
        )
        mac = release.matrix_coverage(self.aws, ["macos-14-arm64-cpu"])
        self.assertEqual(mac, expected - azure_only)
        self.assertEqual(expected, azure_only | mac)
        self.assertEqual(4, len(mac))

    def test_provider_merge_is_azure_only_or_hybrid(self):
        self.assertEqual("azure", release.merged_release_provider(None))
        self.assertEqual(
            "azure", release.merged_release_provider({"provider": "azure"})
        )
        self.assertEqual(
            "hybrid", release.merged_release_provider({"provider": "gcp"})
        )
        self.assertEqual(
            "hybrid", release.merged_release_provider({"provider": "aws"})
        )


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.plan = release.load_plan(HERE / "release-plan.json")

    def test_full_matrix_groups_shards_into_persistent_vm_lanes(self):
        selected = release.selected_executions(self.plan)
        self.assertEqual(len(self.plan["shards"]), len(selected))
        cpu = next(item for item in selected if item["id"] == "linux-x86_64-cpu")
        self.assertEqual(7, len(cpu["build"]["variants"]))
        lanes = release.group_execution_lanes(self.plan, selected)
        self.assertEqual(4, len(lanes))
        jammy = next(item for item in lanes if item["id"] == "linux-x86-64-jammy")
        self.assertGreater(len(jammy["shards"]), 1)
        self.assertIn(
            "linux-x86_64-cuda-12-9",
            {item["id"] for item in jammy["shards"]},
        )

    def test_variant_selection_keeps_its_parent_compatibility_lane(self):
        selected = release.selected_executions(
            self.plan, ["linux-x86_64-cpu--base"]
        )
        lanes = release.group_execution_lanes(self.plan, selected)
        self.assertEqual(["linux-x86-64-jammy"], [item["id"] for item in lanes])

    def test_explicit_lane_rejects_incompatible_members(self):
        first = json.loads(json.dumps(self.plan["shards"][0]))
        second = json.loads(json.dumps(first))
        first["lane"] = second["lane"] = "shared"
        second["id"] = "second"
        second["os"] = "windows"
        second["worker"] = "worker.ps1"
        with self.assertRaisesRegex(ValueError, "mixes incompatible"):
            release.group_execution_lanes(self.plan, [first, second])

    def test_explicit_lane_requires_a_shared_machine_candidate(self):
        first = json.loads(json.dumps(self.plan["shards"][0]))
        second = json.loads(json.dumps(first))
        first["lane"] = second["lane"] = "shared"
        first["machineCandidates"] = ["one"]
        second["machineCandidates"] = ["two"]
        second["id"] = "second"
        with self.assertRaisesRegex(ValueError, "no VM candidate shared"):
            release.group_execution_lanes(self.plan, [first, second])

    def test_specific_variant_selects_only_that_variant(self):
        selected = release.selected_executions(
            self.plan, ["linux-x86_64-cpu--base"]
        )
        self.assertEqual(1, len(selected))
        self.assertEqual("linux-x86_64-cpu--base", selected[0]["id"])
        self.assertEqual(["base"], [
            item["name"] for item in selected[0]["build"]["variants"]
        ])

    def test_excluding_one_variant_keeps_the_other_classifiers(self):
        selected = release.selected_executions(
            self.plan, ["linux-x86_64-cpu"], ["linux-x86_64-cpu--base"]
        )
        names = [item["name"] for item in selected[0]["build"]["variants"]]
        self.assertNotIn("base", names)
        self.assertIn("avx2", names)
        self.assertIn("compile", names)
        coverage = release.execution_matrix_coverage([{"shard": selected[0]}])
        self.assertNotIn("linux-x86_64-cpu--base", coverage)
        self.assertIn("linux-x86_64-cpu--avx2", coverage)

    def test_unknown_selector_fails_before_cloud_calls(self):
        with self.assertRaisesRegex(ValueError, "unknown shard selector"):
            release.selected_executions(self.plan, ["not-a-lane"])
        with self.assertRaisesRegex(ValueError, "unknown variant selector"):
            release.selected_executions(
                self.plan, ["linux-x86_64-cpu--not-a-variant"]
            )


class SchedulingTests(unittest.TestCase):
    def test_size_selection_is_greedy_and_enforces_core_limit(self):
        skus = [
            fake_sku("Standard_F72s_v2", vcpus=72, memory=144),
            fake_sku("Standard_F32s_v2", vcpus=32, memory=64),
            fake_sku("Standard_F16s_v2", vcpus=16, memory=32),
        ]
        selected = release.choose_size_from_skus(
            skus,
            ["Standard_F72s_v2", "Standard_F32s_v2", "Standard_F16s_v2"],
            "x64",
            "eastus2",
            32,
            None,
        )
        self.assertEqual("Standard_F32s_v2", selected["name"])

    def test_size_selection_falls_back_to_a_quota_eligible_family(self):
        skus = [
            fake_sku("Standard_F72s_v2", vcpus=72, memory=144),
            fake_sku(
                "Standard_F64as_v7",
                vcpus=64,
                memory=256,
                family="StandardFasv7Family",
            ),
        ]
        selected = release.choose_size_from_skus(
            skus,
            ["Standard_F72s_v2", "Standard_F64as_v7"],
            "x64",
            "eastus2",
            72,
            None,
            {
                "cores": (0, 278),
                "standardfsv2family": (0, 10),
                "standardfasv7family": (0, 64),
            },
        )
        self.assertEqual("Standard_F64as_v7", selected["name"])

    def test_forced_size_still_fails_closed_when_family_quota_is_short(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "standardFSv2Family quota requires 72 vCPUs, only 10 remain",
        ):
            release.choose_size_from_skus(
                [fake_sku("Standard_F72s_v2", vcpus=72, memory=144)],
                ["Standard_F72s_v2"],
                "x64",
                "eastus2",
                72,
                None,
                {
                    "cores": (0, 278),
                    "standardfsv2family": (0, 10),
                },
            )

    def test_size_selection_can_fall_back_for_total_regional_quota(self):
        skus = [
            fake_sku(
                "Standard_F64as_v7",
                vcpus=64,
                memory=256,
                family="StandardFasv7Family",
            ),
            fake_sku(
                "Standard_F32as_v7",
                vcpus=32,
                memory=128,
                family="StandardFasv7Family",
            ),
        ]
        selected = release.choose_size_from_skus(
            skus,
            ["Standard_F64as_v7", "Standard_F32as_v7"],
            "x64",
            "eastus2",
            72,
            None,
            {
                "cores": (230, 278),
                "standardfasv7family": (0, 64),
            },
        )
        self.assertEqual("Standard_F32as_v7", selected["name"])

    def test_missing_quota_rows_fail_closed(self):
        sku = fake_sku("Standard_F32s_v2")
        cases = [
            ({"standardfsv2family": (0, 64)}, "total regional vCPU quota"),
            ({"cores": (0, 278)}, "standardFSv2Family quota"),
        ]
        for quota_limits, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    RuntimeError, f"Azure did not return {expected}"
                ):
                    release.choose_size_from_skus(
                        [sku],
                        ["Standard_F32s_v2"],
                        "x64",
                        "eastus2",
                        72,
                        None,
                        quota_limits,
                    )

    def test_size_selection_rejects_architecture_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "architecture"):
            release.choose_size_from_skus(
                [fake_sku("Standard_D32ps_v6", architecture="x64")],
                ["Standard_D32ps_v6"],
                "Arm64",
                "eastus2",
                None,
                None,
            )

    def test_forced_zone_must_be_offered(self):
        with self.assertRaisesRegex(RuntimeError, "availability zone 3"):
            release.choose_size_from_skus(
                [fake_sku("Standard_F32s_v2", zones=("1", "2"))],
                ["Standard_F32s_v2"],
                "x64",
                "eastus2",
                None,
                "3",
            )

    def test_forced_zone_rejects_skus_without_zonal_offerings(self):
        with self.assertRaisesRegex(RuntimeError, "availability zone 1"):
            release.choose_size_from_skus(
                [fake_sku("Standard_F32s_v2", zones=())],
                ["Standard_F32s_v2"],
                "x64",
                "eastus2",
                None,
                "1",
            )

    def test_location_restrictions_block_regional_and_zonal_deployments(self):
        restriction = {
            "type": "Location",
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["eastus2"], "zones": []},
        }
        for zone in (None, "1"):
            with self.subTest(zone=zone):
                with self.assertRaisesRegex(
                    RuntimeError, "subscription restriction"
                ):
                    release.choose_size_from_skus(
                        [fake_sku("Standard_F32s_v2", restrictions=[restriction])],
                        ["Standard_F32s_v2"],
                        "x64",
                        "eastus2",
                        None,
                        zone,
                    )

    def test_location_restriction_for_another_region_is_ignored(self):
        restriction = {
            "type": "Location",
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["westus3"], "zones": []},
        }
        selected = release.choose_size_from_skus(
            [fake_sku("Standard_F32s_v2", restrictions=[restriction])],
            ["Standard_F32s_v2"],
            "x64",
            "eastus2",
            None,
            None,
        )
        self.assertEqual("Standard_F32s_v2", selected["name"])

    def test_unknown_subscription_restrictions_are_fail_closed(self):
        restriction = {
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["eastus2"], "zones": []},
        }
        with self.assertRaisesRegex(RuntimeError, "subscription restriction"):
            release.choose_size_from_skus(
                [fake_sku("Standard_F32s_v2", restrictions=[restriction])],
                ["Standard_F32s_v2"],
                "x64",
                "eastus2",
                None,
                None,
            )

    def test_zone_restriction_allows_unzoned_regional_deployment(self):
        restriction = {
            "type": "Zone",
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["eastus2"], "zones": ["3"]},
        }
        sku = fake_sku(
            "Standard_D8ps_v6",
            architecture="Arm64",
            restrictions=[restriction],
        )
        selected = release.choose_size_from_skus(
            [sku],
            ["Standard_D8ps_v6"],
            "Arm64",
            "eastus2",
            None,
            None,
        )
        self.assertEqual("Standard_D8ps_v6", selected["name"])

    def test_zone_restriction_rejects_only_the_restricted_forced_zone(self):
        restriction = {
            "type": "Zone",
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["eastus2"], "zones": ["3"]},
        }
        sku = fake_sku(
            "Standard_D8ps_v6",
            architecture="Arm64",
            restrictions=[restriction],
        )
        with self.assertRaisesRegex(RuntimeError, "subscription restriction"):
            release.choose_size_from_skus(
                [sku],
                ["Standard_D8ps_v6"],
                "Arm64",
                "eastus2",
                None,
                "3",
            )
        selected = release.choose_size_from_skus(
            [sku],
            ["Standard_D8ps_v6"],
            "Arm64",
            "eastus2",
            None,
            "1",
        )
        self.assertEqual("Standard_D8ps_v6", selected["name"])

    def test_zone_restriction_for_another_region_is_ignored(self):
        restriction = {
            "type": "Zone",
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["westus3"], "zones": ["3"]},
        }
        selected = release.choose_size_from_skus(
            [
                fake_sku(
                    "Standard_D8ps_v6",
                    architecture="Arm64",
                    restrictions=[restriction],
                )
            ],
            ["Standard_D8ps_v6"],
            "Arm64",
            "eastus2",
            None,
            "3",
        )
        self.assertEqual("Standard_D8ps_v6", selected["name"])

    def test_any_applicable_restriction_fails_closed(self):
        restrictions = [
            {
                "type": "Zone",
                "reason_code": "NotAvailableForSubscription",
                "restriction_info": {"locations": ["westus3"], "zones": ["3"]},
            },
            {
                "type": "Location",
                "reason_code": "NotAvailableForSubscription",
                "restriction_info": {"locations": ["eastus2"], "zones": []},
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "subscription restriction"):
            release.choose_size_from_skus(
                [fake_sku("Standard_F32s_v2", restrictions=restrictions)],
                ["Standard_F32s_v2"],
                "x64",
                "eastus2",
                None,
                None,
            )

    def test_location_restriction_match_is_case_insensitive(self):
        restriction = {
            "type": "lOcAtIoN",
            "reason_code": "NotAvailableForSubscription",
            "restriction_info": {"locations": ["EastUS2"], "zones": []},
        }
        with self.assertRaisesRegex(RuntimeError, "subscription restriction"):
            release.choose_size_from_skus(
                [fake_sku("Standard_F32s_v2", restrictions=[restriction])],
                ["Standard_F32s_v2"],
                "x64",
                "eastus2",
                None,
                None,
            )

    def test_concurrent_quota_sums_every_active_lane(self):
        lanes = [
            {"selectedMachine": {"family": "standardFSv2Family", "vcpus": 32}},
            {"selectedMachine": {"family": "standardFSv2Family", "vcpus": 16}},
        ]
        usage = [
            {
                "name": {"value": "cores"},
                "current_value": 16,
                "limit": 96,
            },
            {
                "name": {"value": "standardFSv2Family"},
                "current_value": 8,
                "limit": 64,
            },
        ]
        report = release.quota_report(usage, lanes)
        self.assertEqual("cores", report[0]["family"])
        self.assertEqual(48, report[0]["requiredConcurrentPeak"])
        self.assertEqual(48, report[1]["requiredConcurrentPeak"])
        self.assertEqual(80, report[0]["remaining"])
        self.assertEqual(56, report[1]["remaining"])

    def test_parallel_scheduler_packs_four_lanes_across_live_families(self):
        fas_sizes = [
            fake_sku(
                "Standard_F32as_v7",
                vcpus=32,
                memory=128,
                family="StandardFasv7Family",
            ),
            fake_sku(
                "Standard_F16as_v7",
                vcpus=16,
                memory=64,
                family="StandardFasv7Family",
            ),
            fake_sku(
                "Standard_F8s_v2",
                vcpus=8,
                memory=16,
                family="StandardFSv2Family",
            ),
        ]
        arm_sizes = [
            fake_sku(
                "Standard_D96ps_v6",
                architecture="Arm64",
                vcpus=96,
                memory=384,
                family="StandardDpsv6Family",
            )
        ]
        lanes = [
            {
                "id": lane_id,
                "architecture": "x86_64",
                "candidateNames": [
                    "Standard_F32as_v7",
                    "Standard_F16as_v7",
                    "Standard_F8s_v2",
                ],
            }
            for lane_id in ("linux-jammy", "linux-noble", "windows")
        ]
        lanes.append({
            "id": "linux-arm",
            "architecture": "arm64",
            "candidateNames": ["Standard_D96ps_v6"],
        })
        selected = release.choose_parallel_lane_machines(
            [*fas_sizes, *arm_sizes],
            lanes,
            "eastus2",
            None,
            None,
            None,
            {
                "cores": (0, 332),
                "standardfasv7family": (0, 64),
                "standardfsv2family": (0, 10),
                "standarddpsv6family": (0, 96),
            },
        )
        self.assertEqual([16, 16, 32, 96], sorted(
            int(item["vcpus"]) for item in selected
        ))
        self.assertEqual(
            64,
            sum(
                int(item["vcpus"])
                for item in selected
                if item["family"] == "StandardFasv7Family"
            ),
        )

    def test_parallel_scheduler_respects_aggregate_cost_cap(self):
        lanes = [
            {
                "id": lane_id,
                "architecture": "x86_64",
                "candidateNames": ["large", "small"],
            }
            for lane_id in ("linux", "windows", "android")
        ]
        sizes = [
            fake_sku("large", vcpus=32, family="family"),
            fake_sku("small", vcpus=8, memory=16, family="family"),
        ]
        selected = release.choose_parallel_lane_machines(
            sizes,
            lanes,
            "eastus2",
            None,
            48,
            None,
            {"cores": (0, 128), "family": (0, 128)},
        )
        self.assertLessEqual(sum(item["vcpus"] for item in selected), 48)
        self.assertEqual(3, len(selected))

    def test_parallel_scheduler_rejects_unknown_lane_override(self):
        with self.assertRaisesRegex(ValueError, "unselected Azure lane"):
            release.choose_parallel_lane_machines(
                [fake_sku("small", vcpus=8)],
                [{
                    "id": "linux",
                    "architecture": "x86_64",
                    "candidateNames": ["small"],
                }],
                "eastus2",
                None,
                None,
                None,
                {"cores": (0, 64), "standardfsv2family": (0, 64)},
                lane_machine_values=["windows=small"],
            )

    def test_parallel_scheduler_ties_follow_plan_candidate_order(self):
        lanes = [
            {"id": "one", "architecture": "x86_64", "candidateNames": ["b", "a"]},
            {"id": "two", "architecture": "x86_64", "candidateNames": ["a", "b"]},
        ]
        skus = [
            fake_sku("a", vcpus=8, memory=16, family="family"),
            fake_sku("b", vcpus=8, memory=16, family="family"),
        ]
        expected = ["b", "a"]
        quota = {"cores": (0, 32), "family": (0, 32)}
        for values in (skus, list(reversed(skus))):
            selected = release.choose_parallel_lane_machines(
                values, lanes, "eastus2", None, None, None, quota
            )
            self.assertEqual(expected, [item["name"] for item in selected])

    def test_quota_failure_reports_required_and_remaining(self):
        executions = [{
            "selectedMachine": {"family": "standardFSv2Family", "vcpus": 32}
        }]
        usage = [
            {"name": {"value": "cores"}, "current_value": 0, "limit": 128},
            {
                "name": {"value": "standardFSv2Family"},
                "current_value": 24,
                "limit": 32,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "requires 32, only 8"):
            release.quota_report(usage, executions)

    def test_total_regional_quota_is_checked_independently_of_family_quota(self):
        executions = [{
            "selectedMachine": {"family": "standardFSv2Family", "vcpus": 32}
        }]
        usage = [
            {"name": {"value": "cores"}, "current_value": 60, "limit": 64},
            {
                "name": {"value": "standardFSv2Family"},
                "current_value": 0,
                "limit": 64,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "only 4 total regional vCPUs"):
            release.quota_report(usage, executions)

    def test_build_threads_and_heap_scale_to_machine(self):
        shard = {"build": {"buildThreads": 48, "mavenHeapGiB": 32}}
        release.adapt_build_resources(shard, 16, 32, None)
        self.assertEqual(8, shard["build"]["buildThreads"])
        self.assertEqual(28, shard["build"]["mavenHeapGiB"])

    def test_nonpositive_build_thread_override_fails_before_cloud_calls(self):
        for value in (0, -1):
            args = SimpleNamespace(
                plan=HERE / "release-plan.json",
                shard=None,
                exclude_shard=None,
                build_threads=value,
            )
            with self.subTest(value=value), mock.patch.object(
                release, "cloud_context"
            ) as cloud:
                with self.assertRaisesRegex(ValueError, "build-threads must be positive"):
                    release.preflight_data(args)
                cloud.assert_not_called()

    def test_preflight_output_has_a_versioned_parallel_contract(self):
        plan = release.load_plan(HERE / "release-plan.json")
        shard = next(item for item in plan["shards"] if item["id"] == "linux-arm64-cpu")
        machine = release.candidate_names(plan, shard)[0]
        family = "standardDpsv6Family"
        images = mock.Mock()
        images.list.return_value = [SimpleNamespace(name="22.04.202507010")]
        images.get.return_value = object()
        context = {
            "subscription": "subscription",
            "subscriptions": SimpleNamespace(
                subscriptions=SimpleNamespace(
                    list_locations=lambda subscription: [SimpleNamespace(name="eastus2")]
                )
            ),
            "compute": SimpleNamespace(
                resource_skus=SimpleNamespace(
                    list=lambda: [fake_sku(
                        machine,
                        architecture="Arm64",
                        vcpus=8,
                        memory=32,
                        family=family,
                    )]
                ),
                usage=SimpleNamespace(list=lambda location: [
                    SimpleNamespace(name=SimpleNamespace(value="cores"), current_value=0, limit=32),
                    SimpleNamespace(name=SimpleNamespace(value=family), current_value=0, limit=32),
                ]),
                virtual_machine_images=images,
            ),
            "storage": SimpleNamespace(
                storage_accounts=SimpleNamespace(
                    get_properties=lambda group, account: object()
                )
            ),
        }
        args = SimpleNamespace(
            plan=HERE / "release-plan.json",
            shard=["linux-arm64-cpu--base"],
            exclude_shard=None,
            build_threads=None,
            subscription="subscription",
            no_wizard=True,
            location="eastus2",
            zone=None,
            root_volume_gib=None,
            max_cores=8,
            max_total_cores=8,
            machine_type=None,
            lane_machine=None,
            resource_group=None,
            storage_account="dl4jreleaseaccount",
        )
        with mock.patch.object(
            release, "cloud_context", return_value=context
        ), mock.patch.object(release, "resolve_location", return_value="eastus2"):
            result = release.preflight_data(args, include_context=True)
        public = release.printable_preflight(result)
        self.assertEqual(1, public["schemaVersion"])
        self.assertTrue(public["parallel"])
        self.assertFalse(public["serial"])
        self.assertEqual(1, public["laneCount"])
        self.assertEqual(1, public["executionCount"])
        self.assertEqual("linux-arm64-jammy", public["lanes"][0]["id"])
        self.assertEqual("linux-arm64-jammy", public["executions"][0]["laneId"])
        self.assertNotIn("context", public)
        self.assertNotIn("plan", public)
        json.dumps(public)


class MarketplaceImageTests(unittest.TestCase):
    def image(self, version="latest"):
        return {
            "publisher": "Canonical",
            "offer": "0001-com-ubuntu-server-jammy",
            "sku": "22_04-lts-arm64",
            "version": version,
            "architecture": "Arm64",
        }

    def test_latest_resolves_to_a_concrete_verified_version(self):
        images = mock.Mock()
        images.list.return_value = [SimpleNamespace(name="22.04.202507010")]
        image = self.image()

        version = release.resolve_marketplace_image_version(
            images, "eastus2", image
        )

        self.assertEqual("22.04.202507010", version)
        images.list.assert_called_once_with(
            "eastus2",
            image["publisher"],
            image["offer"],
            image["sku"],
            top=1,
            orderby="name desc",
        )
        images.get.assert_called_once_with(
            "eastus2",
            image["publisher"],
            image["offer"],
            image["sku"],
            version,
        )

    def test_explicit_version_is_verified_without_listing(self):
        images = mock.Mock()
        image = self.image("22.04.202507010")

        version = release.resolve_marketplace_image_version(
            images, "eastus2", image
        )

        self.assertEqual(image["version"], version)
        images.list.assert_not_called()
        images.get.assert_called_once_with(
            "eastus2",
            image["publisher"],
            image["offer"],
            image["sku"],
            image["version"],
        )

    def test_latest_requires_at_least_one_concrete_version(self):
        images = mock.Mock()
        images.list.return_value = []

        with self.assertRaisesRegex(
            RuntimeError, "no Azure Marketplace image versions"
        ):
            release.resolve_marketplace_image_version(
                images, "eastus2", self.image()
            )
        images.get.assert_not_called()

    def test_latest_rejects_an_invalid_version_response(self):
        for invalid in ("", "latest"):
            with self.subTest(version=invalid):
                images = mock.Mock()
                images.list.return_value = [SimpleNamespace(name=invalid)]
                with self.assertRaisesRegex(
                    RuntimeError, "invalid Marketplace image version"
                ):
                    release.resolve_marketplace_image_version(
                        images, "eastus2", self.image()
                    )
                images.get.assert_not_called()


class AzureSafetyTests(unittest.TestCase):
    def test_latest_run_uses_created_at_not_lexicographic_run_id(self):
        plan = {"artifactPrefix": "deeplearning4j/releases"}
        prefix = "deeplearning4j/releases/"
        manifests = {
            f"{prefix}z-old/run.json": {"createdAt": "2026-01-01T00:00:00Z"},
            f"{prefix}a-new/run.json": {"createdAt": "2026-08-02T00:00:00Z"},
        }
        container = mock.Mock()
        container.list_blobs.return_value = [
            SimpleNamespace(name=name) for name in manifests
        ]

        def download(name):
            return SimpleNamespace(
                readall=lambda: json.dumps(manifests[name]).encode("utf-8")
            )

        container.download_blob.side_effect = download
        self.assertEqual("a-new", release.latest_run_id(container, plan))

    def test_hybrid_asset_merge_rejects_conflicting_payloads(self):
        existing = [{
            "fileName": "asset.tar.gz",
            "sha256": "a" * 64,
            "size": 10,
            "shard": "lane",
            "sourceObject": "aws/object",
        }]
        identical = [{
            "fileName": "asset.tar.gz",
            "sha256": "a" * 64,
            "size": 10,
            "shard": "lane",
            "sourceObject": "azure/object",
        }]
        merged = release.merge_release_assets(existing, identical)
        self.assertEqual(
            [{"sourceObject": "aws/object"}, {"sourceObject": "azure/object"}],
            merged["asset.tar.gz"]["sources"],
        )
        self.assertNotIn("sourceObject", merged["asset.tar.gz"])
        conflicting = [dict(identical[0], sha256="b" * 64)]
        with self.assertRaisesRegex(RuntimeError, "conflicts across providers"):
            release.merge_release_assets(existing, conflicting)
        extra_metadata = [dict(identical[0], contentType="application/gzip")]
        with self.assertRaisesRegex(RuntimeError, "contentType"):
            release.merge_release_assets(existing, extra_metadata)

    def test_existing_release_manifest_download_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            release.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing a downloadable"):
                release.download_github_release_manifest(
                    "release-tag",
                    "owner/repository",
                    Path(temp),
                    required=True,
                )

    def test_concurrent_github_manifest_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            release, "github_release_exists", return_value=True
        ), mock.patch.object(
            release,
            "download_github_release_manifest",
            return_value=({"schemaVersion": 1}, "new-digest"),
        ):
            with self.assertRaisesRegex(RuntimeError, "changed concurrently"):
                release.assert_github_manifest_unchanged(
                    "release-tag",
                    "owner/repository",
                    True,
                    "old-digest",
                    Path(temp),
                )

    def test_existing_hybrid_maven_archives_are_downloaded_and_verified(self):
        payload = b"verified maven archive"
        asset = {
            "fileName": "maven-repository-macos-14-arm64-cpu.tar.gz",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "shard": "macos-14-arm64-cpu",
        }
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def download(command, check):
                name = command[command.index("--pattern") + 1]
                output_dir = Path(command[command.index("--dir") + 1])
                (output_dir / name).write_bytes(payload)
                return SimpleNamespace(returncode=0)

            with mock.patch.object(release.subprocess, "run", side_effect=download):
                archives = release.download_existing_maven_archives(
                    {"assets": [asset]},
                    directory,
                    "release-tag",
                    "owner/repository",
                )
            self.assertEqual(
                payload,
                archives[asset["fileName"]].read_bytes(),
            )

            bad = dict(asset, sha256="0" * 64)
            with mock.patch.object(release.subprocess, "run", side_effect=download):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    release.download_existing_maven_archives(
                        {"assets": [bad]},
                        directory,
                        "release-tag",
                        "owner/repository",
                    )

    def test_every_retained_github_asset_is_downloaded_and_verified(self):
        payloads = {
            "maven-repository-lane.tar.gz": b"maven",
            "sdk-assets-lane.tar.gz": b"sdk",
            "lane-shard-manifest.json": b"{}",
        }
        manifest = {"assets": [
            {
                "fileName": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "shard": "lane",
            }
            for name, payload in payloads.items()
        ]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)

            def download(command, check):
                name = command[command.index("--pattern") + 1]
                output_dir = Path(command[command.index("--dir") + 1])
                (output_dir / name).write_bytes(payloads[name])
                return SimpleNamespace(returncode=0)

            with mock.patch.object(release.subprocess, "run", side_effect=download):
                outputs = release.download_existing_release_assets(
                    manifest,
                    directory,
                    "release-tag",
                    "owner/repository",
                )
            self.assertEqual(set(payloads), set(outputs))

    def test_existing_shards_are_derived_from_verified_release_assets(self):
        aws_plan = json.loads(
            (ROOT / "release/aws/release-plan.json").read_text(encoding="utf-8")
        )
        shard_id = "macos-14-arm64-cpu"
        planned = next(item for item in aws_plan["shards"] if item["id"] == shard_id)
        version = "1.0.0"
        commit = "a" * 40
        run_id = "aws-run"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            shard_name = f"{shard_id}-shard-manifest.json"
            maven_name = f"maven-repository-{shard_id}.tar.gz"
            sdk_name = f"sdk-assets-{shard_id}.tar.gz"
            shard_path = directory / shard_name
            platform = planned["build"]["javacppPlatform"]
            classifier_files = [
                {
                    "path": (
                        "maven-repository/org/nd4j/artifact/1.0/"
                        f"artifact-1.0-{platform}"
                        f"{variant.get('classifierSuffix', variant.get('suffix', ''))}.jar"
                    )
                }
                for variant in planned["build"]["variants"]
            ]
            shard_path.write_text(json.dumps({
                "runId": run_id,
                "shard": shard_id,
                "commit": commit,
                "releaseVersion": version,
                "workloads": planned["workloads"],
                "os": planned["os"],
                "platform": platform,
                "backend": planned["build"]["backend"],
                "files": classifier_files,
            }), encoding="utf-8")
            (directory / maven_name).write_bytes(b"maven")
            (directory / sdk_name).write_bytes(b"sdk")
            paths = {
                shard_name: shard_path,
                maven_name: directory / maven_name,
                sdk_name: directory / sdk_name,
            }
            assets = {
                name: {"fileName": name, "shard": shard_id}
                for name in paths
            }
            manifest = {
                "runId": run_id,
                "shards": [shard_id],
                "workloads": planned["workloads"],
            }
            self.assertEqual(
                {shard_id},
                release.validate_existing_release_shards(
                    manifest, assets, paths, aws_plan, version, commit
                ),
            )
            expected_coverage = release.matrix_coverage(aws_plan, [shard_id])
            self.assertEqual(
                expected_coverage,
                release.verified_release_matrix_coverage(
                    manifest, {shard_id}, paths, aws_plan
                ),
            )
            forged = dict(manifest, matrixEntries=[f"{shard_id}--base"])
            with self.assertRaisesRegex(RuntimeError, "matrixEntries"):
                release.verified_release_matrix_coverage(
                    forged, {shard_id}, paths, aws_plan
                )
            partial_shard_manifest = json.loads(
                shard_path.read_text(encoding="utf-8")
            )
            partial_shard_manifest["files"] = partial_shard_manifest["files"][:-1]
            shard_path.write_text(
                json.dumps(partial_shard_manifest), encoding="utf-8"
            )
            release.validate_existing_release_shards(
                manifest, assets, paths, aws_plan, version, commit
            )
            partial_coverage = release.verified_release_matrix_coverage(
                manifest, {shard_id}, paths, aws_plan
            )
            omitted_variant = planned["build"]["variants"][-1]["name"]
            self.assertNotIn(f"{shard_id}--{omitted_variant}", partial_coverage)
            self.assertEqual(len(expected_coverage) - 1, len(partial_coverage))
            with self.assertRaisesRegex(RuntimeError, "not all downloaded and verified"):
                release.validate_existing_release_shards(
                    manifest,
                    assets,
                    {name: path for name, path in paths.items() if name != sdk_name},
                    aws_plan,
                    version,
                    commit,
                )

    def test_worker_archive_must_match_its_shard_attestation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "maven-repository.tar.gz"
            path.write_bytes(b"archive")
            manifest = {"files": [{
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }]}
            release.verify_attested_file(manifest, path.name, path)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                release.verify_attested_file(manifest, path.name, path)

    def test_lane_cleanup_is_dependency_ordered(self):
        events = []

        def operation(label):
            return SimpleNamespace(
                result=lambda timeout=None: events.append(f"wait-{label}")
            )

        compute = SimpleNamespace(
            virtual_machines=SimpleNamespace(
                begin_delete=lambda group, name: (
                    events.append("begin-vm") or operation("vm")
                )
            )
        )
        network = SimpleNamespace(
            network_interfaces=SimpleNamespace(
                begin_delete=lambda group, name: (
                    events.append("begin-nic") or operation("nic")
                )
            ),
            public_ip_addresses=SimpleNamespace(
                begin_delete=lambda group, name: (
                    events.append("begin-pip") or operation("pip")
                )
            ),
        )
        errors = release.delete_lane_resources(
            {"compute": compute, "network": network},
            "group",
            {"vm": "vm", "nic": "nic", "publicIp": "pip"},
            fence_check=lambda: events.append("check"),
        )
        self.assertEqual([], errors)
        self.assertEqual(
            [
                "check", "begin-vm", "check", "wait-vm", "check",
                "check", "begin-nic", "check", "wait-nic", "check",
                "check", "begin-pip", "check", "wait-pip", "check",
            ],
            events,
        )

    def test_lane_cleanup_stops_before_next_delete_if_fence_is_lost(self):
        started = []
        operation = SimpleNamespace(result=lambda timeout=None: None)
        context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(
                    begin_delete=lambda group, name: (
                        started.append(name) or operation
                    )
                )
            ),
            "network": SimpleNamespace(
                network_interfaces=SimpleNamespace(
                    begin_delete=lambda group, name: (
                        started.append(name) or operation
                    )
                ),
                public_ip_addresses=SimpleNamespace(
                    begin_delete=lambda group, name: (
                        started.append(name) or operation
                    )
                ),
            ),
        }
        checks = {"count": 0}

        def check():
            checks["count"] += 1
            if checks["count"] >= 4:
                raise RuntimeError("lease renewal failed")

        with self.assertRaisesRegex(RuntimeError, "lease renewal failed"):
            release.delete_lane_resources(
                context,
                "group",
                {"vm": "vm", "nic": "nic", "publicIp": "pip"},
                fence_check=check,
            )
        self.assertEqual(["vm"], started)

    def test_fenced_azure_operation_checks_submission_and_completion(self):
        events = []
        operation = SimpleNamespace(
            result=lambda timeout=None: events.append("wait") or "created"
        )
        actual = release.fenced_azure_operation(
            lambda: events.append("begin") or operation,
            lambda: events.append("check"),
            timeout=30,
        )
        self.assertEqual("created", actual)
        self.assertEqual(["check", "begin", "check", "wait", "check"], events)

    def test_ensure_network_derives_resource_ids_when_sdk_results_omit_them(self):
        calls = {}

        def operation(value):
            return SimpleNamespace(result=lambda timeout=None: value)

        network = SimpleNamespace(
            network_security_groups=SimpleNamespace(
                begin_create_or_update=lambda *args: operation(
                    SimpleNamespace(id=None)
                )
            ),
            virtual_networks=SimpleNamespace(
                begin_create_or_update=lambda *args: operation(
                    SimpleNamespace(name=None)
                )
            ),
            subnets=SimpleNamespace(
                begin_create_or_update=lambda group, vnet, name, parameters: (
                    calls.update(
                        group=group,
                        vnet=vnet,
                        name=name,
                        parameters=parameters,
                    )
                    or operation(SimpleNamespace(id="/subnet"))
                )
            ),
        )

        subnet_id, nsg_id = release.ensure_network(
            {"network": network, "subscription": "subscription"},
            "group",
            "eastus2",
        )

        expected_nsg_id = (
            "/subscriptions/subscription/resourceGroups/group/providers/"
            "Microsoft.Network/networkSecurityGroups/dl4j-release-nsg"
        )
        self.assertEqual("/subnet", subnet_id)
        self.assertEqual(expected_nsg_id, nsg_id)
        self.assertEqual("dl4j-release-vnet", calls["vnet"])
        self.assertEqual(
            {"id": expected_nsg_id},
            calls["parameters"]["network_security_group"],
        )

    def test_ensure_network_preserves_sdk_nsg_id(self):
        calls = {}

        def operation(value):
            return SimpleNamespace(result=lambda timeout=None: value)

        network = SimpleNamespace(
            network_security_groups=SimpleNamespace(
                begin_create_or_update=lambda *args: operation(
                    SimpleNamespace(id="/sdk/nsg")
                )
            ),
            virtual_networks=SimpleNamespace(
                begin_create_or_update=lambda *args: operation(SimpleNamespace())
            ),
            subnets=SimpleNamespace(
                begin_create_or_update=lambda group, vnet, name, parameters: (
                    calls.update(parameters=parameters)
                    or operation(SimpleNamespace(id="/subnet"))
                )
            ),
        )

        subnet_id, nsg_id = release.ensure_network(
            {"network": network}, "group", "eastus2"
        )

        self.assertEqual("/subnet", subnet_id)
        self.assertEqual("/sdk/nsg", nsg_id)
        self.assertEqual(
            {"id": "/sdk/nsg"},
            calls["parameters"]["network_security_group"],
        )

    def test_ensure_network_requires_subscription_when_sdk_omits_nsg_id(self):
        operation = SimpleNamespace(
            result=lambda timeout=None: SimpleNamespace(id=None)
        )
        network = SimpleNamespace(
            network_security_groups=SimpleNamespace(
                begin_create_or_update=lambda *args: operation
            )
        )

        with self.assertRaisesRegex(RuntimeError, "subscription is unavailable"):
            release.ensure_network({"network": network}, "group", "eastus2")

    def test_stream_blob_log_reads_bounded_ranges_from_current_offset(self):
        container = mock.Mock()
        container.download_blob.side_effect = [
            SimpleNamespace(readall=lambda: b"a\nb\n"),
            SimpleNamespace(readall=lambda: b"c\n"),
        ]
        output = io.StringIO()

        with mock.patch.object(release, "LOG_STREAM_CHUNK_BYTES", 4), mock.patch.object(
            release.sys, "stdout", output
        ):
            offset = release.stream_blob_log(
                container, "live.log", 7, label="lane/test"
            )

        self.assertEqual(13, offset)
        self.assertEqual("[lane/test] a\n[lane/test] b\n[lane/test] c\n", output.getvalue())
        self.assertEqual(
            [
                mock.call(
                    "live.log", offset=7, length=4, max_concurrency=1
                ),
                mock.call(
                    "live.log", offset=11, length=4, max_concurrency=1
                ),
            ],
            container.download_blob.call_args_list,
        )

    def test_stream_blob_log_decodes_windows_utf16_output(self):
        payload = "\ufeffline one\r\nline two\r\n".encode("utf-16-le")
        self.assertEqual(
            "continued\r\n",
            release.decode_log_payload("continued\r\n".encode("utf-16-le")),
        )
        container = mock.Mock()
        container.download_blob.return_value.readall.return_value = payload
        output = io.StringIO()

        with mock.patch.object(release.sys, "stdout", output):
            offset = release.stream_blob_log(
                container, "live.log", 0, label="lane/windows"
            )

        self.assertEqual(len(payload), offset)
        self.assertEqual(
            "[lane/windows] line one\r\n[lane/windows] line two\r\n",
            output.getvalue(),
        )

    def test_stream_blob_log_retries_concurrent_append_conflict(self):
        class ResourceModifiedError(Exception):
            status_code = 412
            error_code = "ConditionNotMet"

        first = mock.Mock()
        first.readall.side_effect = ResourceModifiedError(
            "Blob changed during download"
        )
        container = mock.Mock()
        container.download_blob.side_effect = [
            first,
            SimpleNamespace(readall=lambda: b"ok\n"),
        ]

        with mock.patch.object(release.time, "sleep") as sleep:
            self.assertEqual(22, release.stream_blob_log(container, "live.log", 19))

        sleep.assert_called_once_with(0.2)
        self.assertEqual(2, container.download_blob.call_count)

    def test_stream_blob_log_persistent_conflict_preserves_offset(self):
        class ResourceModifiedError(Exception):
            status_code = 412
            error_code = "ConditionNotMet"

        container = mock.Mock()
        container.download_blob.return_value.readall.side_effect = (
            ResourceModifiedError("Blob keeps changing")
        )

        with mock.patch.object(release.time, "sleep"):
            self.assertEqual(19, release.stream_blob_log(container, "live.log", 19))

        self.assertEqual(release.LOG_STREAM_CONFLICT_RETRIES, container.download_blob.call_count)

    def test_stream_blob_log_treats_exhausted_range_as_no_new_bytes(self):
        error = RuntimeError("requested range is beyond the Blob")
        error.status_code = 416
        container = mock.Mock()
        container.download_blob.return_value.readall.side_effect = error

        self.assertEqual(23, release.stream_blob_log(container, "live.log", 23))

    def test_wait_for_lane_preserves_success_before_later_failure(self):
        identities = {
            shard_id: {
                "controllerEpoch": "epoch",
                "runId": "run",
                "shard": shard_id,
                "repository": "repository",
                "commit": "a" * 40,
                "releaseVersion": "1.0.0",
                "snapshotVersion": "1.0.0-SNAPSHOT",
                "contractDigest": f"digest-{shard_id}",
                "variants": ["cpu"],
            }
            for shard_id in ("one", "two")
        }
        statuses = {
            "prefix/one/status.json": {**identities["one"], "exitCode": 0},
            "prefix/two/status.json": {**identities["two"], "exitCode": 1},
        }
        completed = []
        with mock.patch.object(
            release, "stream_blob_log", return_value=0
        ), mock.patch.object(
            release, "print_retained_shard_log"
        ) as retained_log, mock.patch.object(
            release,
            "get_json",
            side_effect=lambda container, name: statuses.get(name),
        ):
            with self.assertRaises(release.LaneWaitError) as raised:
                release.wait_for_lane(
                    {"compute": mock.Mock()},
                    "group",
                    "vm",
                    mock.Mock(),
                    mock.Mock(),
                    {},
                    "prefix",
                    "lane",
                    ["one", "two"],
                    1,
                    identities,
                    result_callback=lambda shard_id, status: completed.append(
                        (shard_id, status)
                    ),
                )
        self.assertEqual([("one", statuses["prefix/one/status.json"])], completed)
        self.assertEqual(
            {"one": statuses["prefix/one/status.json"]},
            raised.exception.completed_results,
        )
        retained_log.assert_called_once_with(mock.ANY, "prefix", "two")

    def test_missing_vm_fails_without_waiting_for_lane_timeout(self):
        class MissingVm(Exception):
            status_code = 404

        context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(
                    instance_view=mock.Mock(side_effect=MissingVm("gone"))
                )
            )
        }
        with mock.patch.object(
            release, "stream_blob_log", return_value=0
        ), mock.patch.object(
            release, "get_json", return_value=None
        ), mock.patch.object(
            release, "kill_switch_enabled", return_value=False
        ), mock.patch.object(release.time, "sleep") as sleep:
            with self.assertRaisesRegex(release.LaneWaitError, "disappeared"):
                release.wait_for_lane(
                    context,
                    "group",
                    "vm",
                    mock.Mock(),
                    mock.Mock(),
                    {},
                    "prefix",
                    "lane",
                    ["one"],
                    12,
                    {"one": {}},
                )
        sleep.assert_not_called()

    def test_wait_for_lane_ignores_stale_checkpoint_identity(self):
        expected = {
            "controllerEpoch": "current",
            "runId": "run",
            "shard": "one",
            "repository": "repository",
            "commit": "a" * 40,
            "releaseVersion": "1.0.0",
            "snapshotVersion": "1.0.0-SNAPSHOT",
            "contractDigest": "digest",
            "variants": ["cpu"],
        }
        observed = iter([
            {**expected, "controllerEpoch": "stale", "exitCode": 0},
            {**expected, "exitCode": 0},
        ])
        view = SimpleNamespace(
            statuses=[SimpleNamespace(code="PowerState/running")]
        )
        context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(
                    instance_view=mock.Mock(return_value=view)
                )
            )
        }
        with mock.patch.object(
            release, "stream_blob_log", return_value=0
        ), mock.patch.object(
            release, "get_json", side_effect=lambda container, name: next(observed)
        ), mock.patch.object(
            release, "kill_switch_enabled", return_value=False
        ), mock.patch.object(release.time, "sleep"):
            actual = release.wait_for_lane(
                context,
                "group",
                "vm",
                mock.Mock(),
                mock.Mock(),
                {},
                "prefix",
                "lane",
                ["one"],
                1,
                {"one": expected},
            )
        self.assertEqual("current", actual["one"]["controllerEpoch"])

    def test_logs_default_to_lane_transcript_and_shard_filter_uses_final_log(self):
        run = {
            "status": "succeeded",
            "lanes": [{"id": "linux-x86-64-jammy"}],
            "executions": [{
                "id": "linux-x86_64-cpu",
                "status": "succeeded",
                "shard": {"id": "linux-x86_64-cpu"},
            }],
        }
        container = mock.Mock()
        service = mock.Mock()
        service.get_container_client.return_value = container
        base = {
            "plan": HERE / "release-plan.json",
            "run_id": "run-id",
            "follow": False,
        }
        with mock.patch.object(
            release, "existing_storage", return_value=({}, "eastus2", "group", mock.Mock(), service)
        ), mock.patch.object(
            release, "load_run", return_value=run
        ), mock.patch.object(
            release, "stream_blob_log", return_value=10
        ) as stream:
            release.show_logs(SimpleNamespace(**base, shard=None))
            stream.assert_called_once_with(
                container,
                "deeplearning4j/releases/run-id/lanes/linux-x86-64-jammy/live.log",
                0,
                label="lane/linux-x86-64-jammy",
            )
            stream.reset_mock()
            release.show_logs(
                SimpleNamespace(**base, shard=["linux-x86_64-cpu"])
            )
            stream.assert_called_once_with(
                container,
                "deeplearning4j/releases/run-id/linux-x86_64-cpu/build.log",
                0,
                label="shard/linux-x86_64-cpu",
            )

    def test_status_includes_vm_instance_view_and_boot_console_tail(self):
        vm = SimpleNamespace(
            name="vm",
            location="eastus2",
            tags={release.MANAGED_TAG: "true", release.RUN_TAG: "run-id"},
            hardware_profile=SimpleNamespace(vm_size="Standard_F64as_v7"),
            provisioning_state="Succeeded",
        )
        virtual_machines = SimpleNamespace(
            list=lambda group: [vm],
            instance_view=lambda group, name: SimpleNamespace(statuses=[
                SimpleNamespace(
                    code="ProvisioningState/succeeded",
                    display_status="Provisioning succeeded",
                    level="Info",
                    message=None,
                    time=None,
                ),
                SimpleNamespace(
                    code="PowerState/running",
                    display_status="VM running",
                    level="Info",
                    message=None,
                    time=None,
                ),
            ]),
            retrieve_boot_diagnostics_data=lambda group, name: SimpleNamespace(
                serial_console_log_blob_uri="https://example.invalid/serial.log?sig=secret",
                console_screenshot_blob_uri="https://example.invalid/screenshot.bmp?sig=secret",
            ),
        )
        context = {
            "subscription": "subscription",
            "compute": SimpleNamespace(virtual_machines=virtual_machines),
        }
        service = mock.Mock()
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"boot line\nworker line\n"
        output = io.StringIO()
        args = SimpleNamespace(plan=HERE / "release-plan.json", run_id="run-id")
        with mock.patch.object(
            release,
            "existing_storage",
            return_value=(context, "eastus2", "group", SimpleNamespace(name="account"), service),
        ), mock.patch.object(
            release, "load_run", return_value={"runId": "run-id", "status": "running"}
        ), mock.patch.object(
            release.urllib.request, "urlopen", return_value=response
        ), mock.patch.object(release.sys, "stdout", output):
            release.status(args)
        actual = json.loads(output.getvalue())
        machine = actual["activeMachines"][0]
        self.assertEqual("Standard_F64as_v7", machine["size"])
        self.assertEqual("Succeeded", machine["provisioningState"])
        self.assertEqual("running", machine["powerState"])
        self.assertEqual(["boot line", "worker line"], machine["consoleOutputTail"])
        self.assertNotIn("sig=secret", json.dumps(machine))

    def test_azure_sccache_receives_only_a_short_lived_container_sas(self):
        class Permissions:
            def __init__(self, **values):
                self.values = values

        generate = mock.Mock(return_value="?sv=1&sig=short-lived")
        context = {
            "modules": {
                "ContainerSasPermissions": Permissions,
                "generate_container_sas": generate,
            }
        }
        plan = {"artifactPrefix": "deeplearning4j/releases", "artifactContainer": "releases"}
        cache = release.compiler_cache_config(
            context, "storageaccount", "account-key", plan, timeout_hours=12
        )
        self.assertEqual("azure", cache["backend"])
        self.assertEqual("releases", cache["container"])
        self.assertEqual(
            "deeplearning4j/releases/compiler-cache/v1", cache["keyPrefix"]
        )
        self.assertIn("SharedAccessSignature=sv=1&sig=short-lived", cache["connectionString"])
        self.assertNotIn("account-key", cache["connectionString"])
        permissions = generate.call_args.kwargs["permission"].values
        self.assertEqual({"read": True, "write": True, "create": True}, permissions)
        self.assertNotIn(
            "connectionString", release.compiler_cache_metadata(plan, "storageaccount")
        )

    def test_failed_extension_provisioning_state_is_fatal(self):
        succeeded = SimpleNamespace(provisioning_state="Succeeded")
        self.assertIs(
            succeeded,
            release.require_succeeded_provisioning_state(
                succeeded, "Windows worker extension"
            ),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Windows worker extension provisioning failed: Failed",
        ):
            release.require_succeeded_provisioning_state(
                SimpleNamespace(provisioning_state="Failed"),
                "Windows worker extension",
            )

    def test_partial_provisioning_invokes_transactional_cleanup(self):
        item = {"shard": {"id": "lane"}}
        fence_check = mock.Mock()
        with mock.patch.object(
            release, "_create_lane_vm_resources", side_effect=RuntimeError("boom")
        ), mock.patch.object(
            release, "delete_lane_resources", return_value=[]
        ) as cleanup:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                release.create_lane_vm(
                    {}, "group", "eastus2", "run", item, object(),
                    "subnet", "worker-url", "ssh-key",
                    fence_check=fence_check,
                )
        resources = cleanup.call_args.args[2]
        self.assertEqual({"vm", "nic", "publicIp"}, set(resources))
        self.assertIs(fence_check, cleanup.call_args.kwargs["fence_check"])

    def test_persistent_lane_receives_multiple_shards_and_one_vm_lifecycle(self):
        plan = release.load_plan(HERE / "release-plan.json")
        context = {
            "subscription": "subscription",
            "modules": {"ContentSettings": lambda **values: values},
        }
        data = {
            "context": context,
            "plan": plan,
            "location": "eastus2",
            "resourceGroup": "group",
            "storageAccount": "account",
        }
        lane = {
            "id": "linux-x86-64-jammy",
            "worker": "worker.sh",
            "executionIds": ["one", "two"],
        }
        executions = {
            "one": {
                "shard": {
                    "id": "one",
                    "build": {"variants": [{"name": "cpu"}]},
                }
            },
            "two": {
                "shard": {
                    "id": "two",
                    "build": {"variants": [{"name": "cpu"}]},
                }
            },
        }
        args = SimpleNamespace(
            run_id="run",
            version="1.0.0",
            snapshot_version="1.0.0-SNAPSHOT",
            commit="a" * 40,
            repository="repository",
            timeout_hours=12,
        )
        artifact = mock.Mock()
        events = release.queue.Queue()
        results = {
            "one": {"shard": "one", "exitCode": 0},
            "two": {"shard": "two", "exitCode": 0},
        }
        with mock.patch.object(
            release, "render_worker", return_value=b"worker"
        ) as render, mock.patch.object(
            release, "worker_sas_url", return_value="worker-url"
        ), mock.patch.object(
            release,
            "compiler_cache_config",
            return_value={
                "backend": "azure",
                "container": "releases",
                "keyPrefix": "deeplearning4j/releases/compiler-cache/v1",
                "connectionString": "sas",
            },
        ), mock.patch.object(
            release,
            "create_lane_vm",
            return_value={"vm": "vm", "nic": "nic", "publicIp": "pip"},
        ) as create, mock.patch.object(
            release, "wait_for_lane", return_value=results
        ) as wait, mock.patch.object(
            release, "delete_lane_resources", return_value=[]
        ) as cleanup:
            actual = release._run_parallel_lane(
                args,
                mock.Mock(),
                data,
                mock.Mock(),
                "account-key",
                artifact,
                mock.Mock(),
                SimpleNamespace(client_id="identity"),
                "subnet",
                "ssh-key",
                "prefix",
                lane,
                executions,
                events,
            )
        self.assertEqual(results, actual)
        config = render.call_args.args[1]
        self.assertEqual("linux-x86-64-jammy", config["laneId"])
        self.assertEqual(["one", "two"], [
            shard["id"] for shard in config["shards"]
        ])
        self.assertTrue(config["controllerEpoch"])
        self.assertEqual("azure", config["compilerCache"]["backend"])
        self.assertTrue(all(
            shard["contractDigest"] for shard in config["shards"]
        ))
        self.assertNotIn("shard", config)
        create.assert_called_once()
        cleanup.assert_called_once()
        self.assertEqual(["one", "two"], wait.call_args.args[8])
        self.assertEqual(
            {"one", "two"},
            set(wait.call_args.args[10]),
        )
        self.assertEqual(
            ["provisioning", "running", "succeeded"],
            [events.get_nowait()["status"] for _ in range(3)],
        )

    def test_controller_starts_three_selected_lanes_concurrently(self):
        plan = release.load_plan(HERE / "release-plan.json")
        lane_ids = ["linux", "arm", "windows"]
        lanes = [{
            "id": lane_id,
            "os": "windows" if lane_id == "windows" else "linux",
            "architecture": "arm64" if lane_id == "arm" else "x86_64",
            "worker": "worker.ps1" if lane_id == "windows" else "worker.sh",
            "image": {"publisher": "p", "offer": "o", "sku": "s", "version": "v"},
            "executionIds": [lane_id],
            "selectedMachine": {"name": "size", "vcpus": 8},
            "rootVolumeGiB": 128,
            "zone": None,
        } for lane_id in lane_ids]
        executions = [{
            "id": lane_id,
            "laneId": lane_id,
            "shard": {"id": lane_id},
            "selectedMachine": {"name": "size", "vcpus": 8},
            "rootVolumeGiB": 128,
            "zone": None,
        } for lane_id in lane_ids]
        context = {
            "subscription": "subscription",
            "modules": {},
        }
        data = {
            "context": context,
            "plan": plan,
            "location": "eastus2",
            "resourceGroup": "group",
            "storageAccount": "account",
            "lanes": lanes,
            "executions": executions,
        }
        args = SimpleNamespace(
            commit="a" * 40,
            run_id="run",
            version="1.0.0",
            snapshot_version="1.0.0-SNAPSHOT",
            branch="main",
            repository="repository",
            reset_kill_switch=False,
            ssh_public_key=None,
        )
        barrier = release.threading.Barrier(3)
        state_lock = release.threading.Lock()
        state = {"active": 0, "maximum": 0}

        def run_lane(*call_args):
            lane = call_args[11]
            events = call_args[13]
            events.put({"laneId": lane["id"], "status": "provisioning"})
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            barrier.wait(timeout=2)
            events.put({
                "laneId": lane["id"],
                "status": "running",
                "resources": {"vm": lane["id"], "nic": "nic", "publicIp": "pip"},
            })
            release.time.sleep(0.02)
            result = {
                lane["id"]: {"shard": lane["id"], "exitCode": 0}
            }
            events.put({
                "laneId": lane["id"],
                "status": "succeeded",
                "results": result,
            })
            with state_lock:
                state["active"] -= 1
            return result

        artifact = mock.Mock()
        control = mock.Mock()
        service = mock.Mock()
        service.get_container_client.side_effect = [artifact, control]
        lease = mock.Mock()
        with mock.patch.object(
            release, "get_json", side_effect=[None, {"enabled": False}]
        ), mock.patch.object(
            release, "put_json"
        ) as put, mock.patch.object(
            release, "set_kill_switch"
        ), mock.patch.object(
            release,
            "ensure_identity",
            return_value=(SimpleNamespace(client_id="identity"), {"name": "identity"}),
        ), mock.patch.object(
            release, "ensure_network", return_value=("subnet", {})
        ), mock.patch.object(
            release, "resolve_ssh_public_key", return_value="ssh"
        ), mock.patch.object(
            release, "_run_parallel_lane", side_effect=run_lane
        ) as runner:
            release._start_under_controller_lease(
                args,
                lease,
                data,
                SimpleNamespace(id="/storage/account"),
                service,
                "key",
            )
        self.assertEqual(3, runner.call_count)
        self.assertEqual(3, state["maximum"])
        final_manifest = put.call_args.args[2]
        self.assertEqual("succeeded", final_manifest["status"])
        self.assertTrue(all(
            item["status"] == "succeeded"
            for item in final_manifest["lanes"]
        ))

    def test_lane_failure_keeps_completed_shard_success_in_manifest(self):
        plan = release.load_plan(HERE / "release-plan.json")
        lane = {
            "id": "linux",
            "os": "linux",
            "architecture": "x86_64",
            "worker": "worker.sh",
            "image": {"publisher": "p", "offer": "o", "sku": "s", "version": "v"},
            "executionIds": ["one", "two"],
            "selectedMachine": {"name": "size", "vcpus": 8},
            "rootVolumeGiB": 128,
            "zone": None,
        }
        executions = [{
            "id": shard_id,
            "laneId": "linux",
            "shard": {"id": shard_id},
            "selectedMachine": {"name": "size", "vcpus": 8},
            "rootVolumeGiB": 128,
            "zone": None,
        } for shard_id in ("one", "two")]
        data = {
            "context": {"subscription": "subscription", "modules": {}},
            "plan": plan,
            "location": "eastus2",
            "resourceGroup": "group",
            "storageAccount": "account",
            "lanes": [lane],
            "executions": executions,
        }
        args = SimpleNamespace(
            commit="a" * 40,
            run_id="run",
            version="1.0.0",
            snapshot_version="1.0.0-SNAPSHOT",
            branch="main",
            repository="repository",
            reset_kill_switch=False,
            ssh_public_key=None,
        )

        def run_lane(*call_args):
            events = call_args[13]
            events.put({"laneId": "linux", "status": "provisioning"})
            events.put({"laneId": "linux", "status": "running", "resources": {"vm": "vm"}})
            result = {"shard": "one", "exitCode": 0}
            events.put({
                "laneId": "linux",
                "status": "shard-succeeded",
                "executionId": "one",
                "result": result,
            })
            events.put({
                "laneId": "linux",
                "status": "failed",
                "failure": "two failed",
                "results": {"one": result},
            })
            raise RuntimeError("two failed")

        artifact = mock.Mock()
        control = mock.Mock()
        service = mock.Mock()
        service.get_container_client.side_effect = [artifact, control]
        with mock.patch.object(
            release, "get_json", side_effect=[None, {"enabled": False}]
        ), mock.patch.object(
            release, "put_json"
        ) as put, mock.patch.object(
            release,
            "ensure_identity",
            return_value=(SimpleNamespace(client_id="identity"), {"name": "identity"}),
        ), mock.patch.object(
            release, "ensure_network", return_value=("subnet", {})
        ), mock.patch.object(
            release, "resolve_ssh_public_key", return_value="ssh"
        ), mock.patch.object(
            release, "_run_parallel_lane", side_effect=run_lane
        ) as runner, mock.patch.object(release, "set_kill_switch"):
            with self.assertRaisesRegex(RuntimeError, "parallel Azure lane failure"):
                release._start_under_controller_lease(
                    args,
                    mock.Mock(),
                    data,
                    SimpleNamespace(id="/storage/account"),
                    service,
                    "key",
                )
        final_manifest = put.call_args.args[2]
        records = {item["id"]: item for item in final_manifest["executions"]}
        self.assertEqual("succeeded", records["one"]["status"])
        self.assertEqual("failed", records["two"]["status"])
        self.assertEqual("failed", final_manifest["status"])
        self.assertTrue(runner.call_args.args[14].is_set())

    def test_failed_identity_assignment_rollback_is_fenced(self):
        class FatalRoleError(Exception):
            status_code = 403

        events = []

        def create_identity(group, name, parameters):
            events.append("identity-create")
            return SimpleNamespace(principal_id="principal")

        def create_role(scope, assignment_id, parameters):
            events.append("role-create")
            raise FatalRoleError("denied")

        context = {
            "subscription": "subscription",
            "identity": SimpleNamespace(
                user_assigned_identities=SimpleNamespace(
                    create_or_update=create_identity,
                    delete=lambda group, name: (
                        events.append("identity-delete")
                        or SimpleNamespace(
                            result=lambda timeout=None: events.append("identity-wait")
                        )
                    ),
                )
            ),
            "authorization": SimpleNamespace(
                role_assignments=SimpleNamespace(create=create_role)
            ),
        }
        with self.assertRaises(FatalRoleError):
            release.ensure_identity(
                context,
                "group",
                "eastus2",
                "run",
                "/storage/scope",
                fence_check=lambda: events.append("check"),
            )
        delete_index = events.index("identity-delete")
        self.assertEqual("check", events[delete_index - 1])
        self.assertEqual(
            ["check", "identity-wait", "check"],
            events[delete_index + 1:delete_index + 4],
        )

    def test_delete_logs_acquires_and_releases_controller_lease(self):
        plan = release.load_plan(HERE / "release-plan.json")
        context = {"compute": mock.Mock()}
        service = mock.Mock()
        lease = mock.Mock()
        lease.release.return_value = []
        args = SimpleNamespace(
            yes=True,
            plan=HERE / "release-plan.json",
            run_id="run-id",
            all_runs=False,
        )
        with mock.patch.object(
            release,
            "existing_storage",
            return_value=(context, "eastus2", "group", object(), service),
        ), mock.patch.object(
            release.ControllerLease, "acquire", return_value=lease
        ) as acquire, mock.patch.object(
            release,
            "_delete_logs_under_controller_lease",
            return_value=["log"],
        ) as delete_under_lease, mock.patch("sys.stdout", new_callable=io.StringIO):
            release.delete_logs(args)
        acquire.assert_called_once_with()
        delete_under_lease.assert_called_once_with(
            args, plan, context, "group", service, lease
        )
        lease.release.assert_called_once_with()

    def test_log_blob_deletion_checks_fence_before_and_after_each_call(self):
        plan = release.load_plan(HERE / "release-plan.json")
        events = []
        artifact_container = mock.Mock()
        artifact_container.list_blobs.return_value = [
            SimpleNamespace(name="deeplearning4j/releases/run-id/lane/live.log"),
            SimpleNamespace(name="deeplearning4j/releases/run-id/lane/build.log"),
            SimpleNamespace(name="deeplearning4j/releases/run-id/run.json"),
        ]
        artifact_container.delete_blob.side_effect = (
            lambda name, delete_snapshots=None: events.append(f"delete:{name}")
        )
        service = mock.Mock()
        service.get_container_client.return_value = artifact_container
        context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(list=lambda group: [])
            )
        }
        lease = mock.Mock()
        lease.check.side_effect = lambda: events.append("check")
        args = SimpleNamespace(run_id="run-id", all_runs=False)
        with mock.patch.object(
            release, "load_run", return_value={"status": "succeeded"}
        ):
            removed = release._delete_logs_under_controller_lease(
                args, plan, context, "group", service, lease
            )
        self.assertEqual(
            [
                "deeplearning4j/releases/run-id/lane/live.log",
                "deeplearning4j/releases/run-id/lane/build.log",
            ],
            removed,
        )
        for index, event in enumerate(events):
            if event.startswith("delete:"):
                self.assertEqual("check", events[index - 1])
                self.assertEqual("check", events[index + 1])

    def test_log_deletion_stops_before_next_blob_if_fence_is_lost(self):
        plan = release.load_plan(HERE / "release-plan.json")
        names = [
            "deeplearning4j/releases/run-id/lane/live.log",
            "deeplearning4j/releases/run-id/lane/build.log",
        ]
        deleted = []
        artifact_container = mock.Mock()
        artifact_container.list_blobs.return_value = [
            SimpleNamespace(name=name) for name in names
        ]
        artifact_container.delete_blob.side_effect = (
            lambda name, delete_snapshots=None: deleted.append(name)
        )
        service = mock.Mock()
        service.get_container_client.return_value = artifact_container
        context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(list=lambda group: [])
            )
        }
        checks_after_first = {"count": 0}

        def check():
            if deleted:
                checks_after_first["count"] += 1
                if checks_after_first["count"] >= 2:
                    raise RuntimeError("lease renewal failed")

        lease = mock.Mock()
        lease.check.side_effect = check
        args = SimpleNamespace(run_id="run-id", all_runs=False)
        with mock.patch.object(
            release, "load_run", return_value={"status": "succeeded"}
        ), self.assertRaisesRegex(RuntimeError, "lease renewal failed"):
            release._delete_logs_under_controller_lease(
                args, plan, context, "group", service, lease
            )
        self.assertEqual([names[0]], deleted)

    def test_log_deletion_refuses_nonterminal_or_still_provisioned_runs(self):
        plan = release.load_plan(HERE / "release-plan.json")
        artifact_container = mock.Mock()
        service = mock.Mock()
        service.get_container_client.return_value = artifact_container
        args = SimpleNamespace(run_id="run-id", all_runs=False)

        running_context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(list=lambda group: [])
            )
        }
        with mock.patch.object(
            release, "load_run", return_value={"status": "running"}
        ), self.assertRaisesRegex(RuntimeError, "non-terminal"):
            release._delete_logs_under_controller_lease(
                args, plan, running_context, "group", service, mock.Mock()
            )
        artifact_container.delete_blob.assert_not_called()

        active_context = {
            "compute": SimpleNamespace(
                virtual_machines=SimpleNamespace(
                    list=lambda group: [
                        SimpleNamespace(
                            name="vm",
                            tags={
                                release.MANAGED_TAG: "true",
                                release.RUN_TAG: "run-id",
                            },
                        )
                    ]
                )
            )
        }
        with mock.patch.object(
            release, "load_run", return_value={"status": "failed"}
        ), self.assertRaisesRegex(RuntimeError, "still has an Azure VM"):
            release._delete_logs_under_controller_lease(
                args, plan, active_context, "group", service, mock.Mock()
            )
        artifact_container.delete_blob.assert_not_called()

    def test_controller_blob_lease_is_acquired_and_released(self):
        lease_client = mock.Mock()
        blob = mock.Mock()
        blob.acquire_lease.return_value = lease_client
        container = mock.Mock()
        container.get_blob_client.return_value = blob
        controller = release.ControllerLease(
            container, "control/controller.lock", duration=60
        ).acquire()
        self.assertEqual([], controller.release())
        blob.acquire_lease.assert_called_once_with(lease_duration=60)
        lease_client.release.assert_called_once_with()

    def test_controller_lease_protects_the_kill_switch_blob_itself(self):
        plan = release.load_plan(HERE / "release-plan.json")
        self.assertEqual(
            release.kill_switch_blob(plan),
            release.controller_lock_blob(plan),
        )

    def test_controller_json_update_uses_epoch_and_etag_compare_and_swap(self):
        downloader = SimpleNamespace(
            properties={"etag": '"etag-1"'},
            readall=lambda: b'{"controllerEpoch":"epoch","status":"old"}',
        )
        container = mock.Mock()
        container.download_blob.return_value = downloader
        lease = SimpleNamespace(
            epoch="epoch",
            name="control/kill-switch.json",
            lease="lease-token",
            check=mock.Mock(),
        )
        modules = {
            "ContentSettings": lambda **values: values,
            "MatchConditions": SimpleNamespace(IfNotModified="if-not-modified"),
        }
        release.put_json(
            container,
            "release/run.json",
            {"controllerEpoch": "epoch", "status": "running"},
            modules,
            controller_lease=lease,
        )
        options = container.upload_blob.call_args.kwargs
        self.assertEqual('"etag-1"', options["etag"])
        self.assertEqual("if-not-modified", options["match_condition"])
        self.assertEqual(2, lease.check.call_count)
        payload = json.loads(container.upload_blob.call_args.args[1])
        self.assertEqual("epoch", payload["controllerEpoch"])

    def test_kill_switch_write_supplies_its_blob_lease(self):
        plan = release.load_plan(HERE / "release-plan.json")
        container = mock.Mock()
        lease = SimpleNamespace(
            epoch="epoch",
            name=release.kill_switch_blob(plan),
            lease="lease-token",
            check=mock.Mock(),
        )
        modules = {"ContentSettings": lambda **values: values}
        release.set_kill_switch(
            container,
            plan,
            True,
            modules,
            "test",
            controller_lease=lease,
        )
        self.assertEqual(
            "lease-token", container.upload_blob.call_args.kwargs["lease"]
        )
        payload = json.loads(container.upload_blob.call_args.args[1])
        self.assertEqual("epoch", payload["controllerEpoch"])
        self.assertFalse(payload["force"])

    def test_controller_check_actively_detects_a_broken_lease(self):
        lease_client = mock.Mock()
        lease_client.renew.side_effect = RuntimeError("lease broken")
        blob = mock.Mock()
        blob.acquire_lease.return_value = lease_client
        container = mock.Mock()
        container.get_blob_client.return_value = blob
        controller = release.ControllerLease(
            container, "control/controller.lock", duration=60
        ).acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "renewal failed"):
                controller.check()
        finally:
            controller.release()

    def test_controller_blob_lease_rejects_a_second_controller(self):
        class ConflictError(Exception):
            status_code = 409

        blob = mock.Mock()
        blob.acquire_lease.side_effect = ConflictError("leased")
        container = mock.Mock()
        container.get_blob_client.return_value = blob
        with self.assertRaisesRegex(RuntimeError, "another Azure release controller"):
            release.ControllerLease(container, "control/controller.lock").acquire()

    def test_start_failure_restores_kill_switch_and_audits_run(self):
        plan = release.load_plan(HERE / "release-plan.json")
        context = {"modules": {}, "subscription": "subscription"}
        data = {
            "context": context,
            "plan": plan,
            "resourceGroup": "group",
            "storageAccount": "account",
            "location": "eastus2",
        }
        artifact_container = mock.Mock()
        control_container = mock.Mock()
        service = mock.Mock()
        service.get_container_client.side_effect = [
            artifact_container,
            control_container,
        ]
        account = SimpleNamespace(id="/storage/account")
        lease = mock.Mock()
        lease.release.return_value = []
        run = {"status": "initializing"}
        args = SimpleNamespace(
            commit="a" * 40,
            repository="https://github.com/deeplearning4j/deeplearning4j.git",
            branch="main",
            version="1.0.0",
            run_id="run-id",
        )
        with mock.patch.object(
            release, "preflight_data", return_value=data
        ), mock.patch.object(
            release, "ensure_resource_group"
        ), mock.patch.object(
            release, "ensure_storage", return_value=(account, service, "key")
        ), mock.patch.object(
            release.ControllerLease, "acquire", return_value=lease
        ), mock.patch.object(
            release, "_start_under_controller_lease", side_effect=RuntimeError("setup failed")
        ), mock.patch.object(
            release, "managed_run_vm_names", return_value=[]
        ), mock.patch.object(
            release, "cleanup_managed_identities", return_value=([], [])
        ), mock.patch.object(
            release, "set_kill_switch"
        ) as kill, mock.patch.object(
            release, "load_run", return_value=run
        ), mock.patch.object(release, "put_json"):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                release.start(args)
        kill.assert_called_once_with(
            control_container,
            plan,
            True,
            context["modules"],
            "controller-failure",
            controller_lease=lease,
        )
        self.assertEqual("failed", run["status"])
        self.assertEqual("setup failed", run["failure"])
        lease.check.assert_called()
        lease.release.assert_called_once_with()

    def test_start_stops_all_shared_cleanup_mutations_after_lease_loss(self):
        plan = release.load_plan(HERE / "release-plan.json")
        context = {"modules": {}, "subscription": "subscription"}
        data = {
            "context": context,
            "plan": plan,
            "resourceGroup": "group",
            "storageAccount": "account",
            "location": "eastus2",
        }
        service = mock.Mock()
        service.get_container_client.side_effect = [mock.Mock(), mock.Mock()]
        lease = mock.Mock()
        lease.check.side_effect = RuntimeError("lease renewal failed")
        lease.release.return_value = []
        args = SimpleNamespace(
            commit="a" * 40,
            repository="repository",
            branch="main",
            version="1.0.0",
            run_id="run-id",
        )
        with mock.patch.object(
            release, "preflight_data", return_value=data
        ), mock.patch.object(release, "ensure_resource_group"), mock.patch.object(
            release,
            "ensure_storage",
            return_value=(SimpleNamespace(id="scope"), service, "key"),
        ), mock.patch.object(
            release.ControllerLease, "acquire", return_value=lease
        ), mock.patch.object(
            release,
            "_start_under_controller_lease",
            side_effect=RuntimeError("lease renewal failed"),
        ), mock.patch.object(release, "set_kill_switch") as kill, mock.patch.object(
            release, "managed_run_vm_names"
        ) as machines, mock.patch.object(
            release, "cleanup_managed_identities"
        ) as identities, mock.patch.object(
            release, "load_run"
        ) as load_run, mock.patch.object(release, "put_json") as put:
            with self.assertRaisesRegex(RuntimeError, "lease renewal failed"):
                release.start(args)
        kill.assert_not_called()
        machines.assert_not_called()
        identities.assert_not_called()
        load_run.assert_not_called()
        put.assert_not_called()
        lease.release.assert_called_once_with()

    def test_start_retains_identity_while_a_run_vm_survives_cleanup(self):
        plan = release.load_plan(HERE / "release-plan.json")
        context = {"modules": {}, "subscription": "subscription"}
        data = {
            "context": context,
            "plan": plan,
            "resourceGroup": "group",
            "storageAccount": "account",
            "location": "eastus2",
        }
        artifact = mock.Mock()
        control = mock.Mock()
        service = mock.Mock()
        service.get_container_client.side_effect = [artifact, control]
        lease = mock.Mock()
        lease.release.return_value = []
        run = {"status": "succeeded"}
        args = SimpleNamespace(
            commit="a" * 40,
            repository="repository",
            branch="main",
            version="1.0.0",
            run_id="run-id",
        )
        with mock.patch.object(
            release, "preflight_data", return_value=data
        ), mock.patch.object(release, "ensure_resource_group"), mock.patch.object(
            release,
            "ensure_storage",
            return_value=(SimpleNamespace(id="scope"), service, "key"),
        ), mock.patch.object(
            release.ControllerLease, "acquire", return_value=lease
        ), mock.patch.object(
            release, "_start_under_controller_lease"
        ), mock.patch.object(
            release, "managed_run_vm_names", return_value=["vm-one"]
        ), mock.patch.object(
            release, "cleanup_managed_identities"
        ) as identities, mock.patch.object(
            release, "load_run", return_value=run
        ), mock.patch.object(release, "put_json"):
            with self.assertRaisesRegex(RuntimeError, "identity retained"):
                release.start(args)
        identities.assert_not_called()
        self.assertEqual("failed", run["status"])
        self.assertIn("identity retained", run["failure"])

    def test_emergency_stop_does_not_create_storage(self):
        source = (HERE / "release.py").read_text(encoding="utf-8")
        body = source.split("def stop_everything", 1)[1].split(
            "def add_selection_options", 1
        )[0]
        fence = source.split("def fence_release_controller", 1)[1].split(
            "def stop_everything", 1
        )[0]
        self.assertNotIn("ensure_storage(", body)
        self.assertIn("fence_release_controller", body)
        self.assertIn("get_properties(group, account_name)", fence)
        self.assertIn("break_lease", fence)
        self.assertIn("fence_lease = ControllerLease", fence)
        self.assertIn('"stop-everything fenced"', fence)

    def test_emergency_stop_refuses_deletion_when_fencing_fails(self):
        plan = release.load_plan(HERE / "release-plan.json")
        compute = mock.Mock()
        context = {"subscription": "subscription", "compute": compute}
        args = SimpleNamespace(
            plan=HERE / "release-plan.json",
            subscription="subscription",
            location="eastus2",
            no_wizard=True,
            resource_group=None,
            storage_account="dl4jreleaseaccount",
        )
        with mock.patch.object(
            release, "cloud_context", return_value=context
        ), mock.patch.object(
            release, "resolve_location", return_value="eastus2"
        ), mock.patch.object(
            release,
            "fence_release_controller",
            side_effect=RuntimeError("storage unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be fenced"):
                release.stop_everything(args)
        compute.virtual_machines.list.assert_not_called()

    def test_emergency_cleanup_stops_before_next_delete_if_fence_is_lost(self):
        plan = release.load_plan(HERE / "release-plan.json")
        started = []
        virtual_machines = SimpleNamespace(
            list=lambda group: [
                SimpleNamespace(name="vm-one", tags={release.MANAGED_TAG: "true"}),
                SimpleNamespace(name="vm-two", tags={release.MANAGED_TAG: "true"}),
            ],
            begin_delete=lambda group, name: (
                started.append(name)
                or SimpleNamespace(result=lambda timeout=None: None)
            ),
        )
        context = {
            "subscription": "subscription",
            "compute": SimpleNamespace(virtual_machines=virtual_machines),
            "network": mock.Mock(),
        }
        checks = {"count": 0}

        def check():
            checks["count"] += 1
            if checks["count"] >= 5:
                raise RuntimeError("lease renewal failed")

        fence = mock.Mock()
        fence.check.side_effect = check
        args = SimpleNamespace(purge_logs=False, purge_storage=False)
        with self.assertRaisesRegex(RuntimeError, "lease renewal failed"):
            release._stop_fenced_resources(
                args,
                plan,
                context,
                "eastus2",
                "group",
                "scope",
                mock.Mock(),
                mock.Mock(),
                fence,
            )
        self.assertEqual(["vm-one"], started)

    def test_identity_cleanup_checks_fence_around_each_destructive_call(self):
        events = []
        identity = SimpleNamespace(
            name="identity",
            principal_id="principal",
            tags={release.MANAGED_TAG: "true"},
        )
        context = {
            "authorization": SimpleNamespace(
                role_assignments=SimpleNamespace(
                    delete=lambda scope, assignment: (
                        events.append("role-delete")
                        or SimpleNamespace(result=lambda timeout=None: None)
                    )
                )
            ),
            "identity": SimpleNamespace(
                user_assigned_identities=SimpleNamespace(
                    list_by_resource_group=lambda group: [identity],
                    delete=lambda group, name: (
                        events.append("identity-delete")
                        or SimpleNamespace(result=lambda timeout=None: None)
                    ),
                )
            ),
        }
        release.cleanup_managed_identities(
            context,
            "group",
            "scope",
            fence_check=lambda: events.append("check"),
        )
        self.assertGreaterEqual(events.count("check"), 6)
        self.assertLess(events.index("check"), events.index("role-delete"))
        self.assertLess(events.index("role-delete"), events.index("identity-delete"))

    def test_emergency_stop_holds_fence_until_cleanup_finishes(self):
        events = []
        context = {"subscription": "subscription"}
        lease = mock.Mock()
        lease.release.side_effect = lambda: events.append("release") or []
        args = SimpleNamespace(
            plan=HERE / "release-plan.json",
            subscription="subscription",
            location="eastus2",
            no_wizard=True,
            resource_group=None,
            storage_account="dl4jreleaseaccount",
        )
        with mock.patch.object(
            release, "cloud_context", return_value=context
        ), mock.patch.object(
            release, "resolve_location", return_value="eastus2"
        ), mock.patch.object(
            release,
            "fence_release_controller",
            return_value=("scope", mock.Mock(), mock.Mock(), lease),
        ), mock.patch.object(
            release,
            "_stop_fenced_resources",
            side_effect=lambda *unused: events.append("cleanup"),
        ):
            release.stop_everything(args)
        self.assertEqual(["cleanup", "release"], events)

    def test_windows_vm_contract_uses_durable_task_and_valid_computer_name(self):
        source = (HERE / "release.py").read_text(encoding="utf-8")
        self.assertIn(
            'resource_name("dl4j", resource_run_id, lane_id, 15)', source
        )
        self.assertIn('"type": "UserAssigned"', source)
        command = release.windows_worker_bootstrap_command()
        self.assertIn("Get-ChildItem -LiteralPath .", command)
        self.assertIn("-Recurse -File", command)
        self.assertIn("& $worker.FullName -Register", command)
        self.assertNotIn("-File worker.ps1", command)
        worker = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn("Register-ScheduledTask", worker)
        self.assertIn("Start-ScheduledTask", worker)
        self.assertIn(
            "$SccacheSha256 = "
            "'b0b257a164bf438b2dea134ca7ded41c100f59a64b3bf275a202f1e8102ab217'",
            worker,
        )
        self.assertIn(
            "Get-FileHash -LiteralPath $SccacheArchive -Algorithm SHA256",
            worker,
        )
        self.assertIn("sccache archive SHA-256 mismatch", worker)
        self.assertIn("$script:WindowsTarExe = Join-Path $env:SystemRoot 'System32\\tar.exe'", worker)
        self.assertIn("& $script:WindowsTarExe -xzf $SccacheArchive -C $env:TEMP", worker)
        self.assertIn("& $script:WindowsTarExe -C $MavenOutput -czf", worker)
        self.assertIn("& $script:WindowsTarExe -C $SdkOutput -czf", worker)
        self.assertNotRegex(worker, r"(?m)^\s+tar (?:-xzf|-C)")
        self.assertIn("$null = $Process.Handle", worker)
        self.assertIn("$Process.WaitForExit()\n  $BuildExitCode = $Process.ExitCode", worker)
        self.assertIn("Build process exited without an available exit code", worker)
        self.assertIn("$null = $Probe.Handle", worker)
        self.assertIn("$Probe.WaitForExit()\n      $State = $Probe.ExitCode", worker)
        self.assertIn("worker-started.txt", worker)
        self.assertIn("worker-attempt.txt", worker)
        self.assertIn("Remove-Item -LiteralPath $LogForwarderStop", worker)
        self.assertIn("$SourceRoot = Join-Path $WorkRoot 'sources'", worker)
        self.assertIn("git -c core.autocrlf=false clone --filter=blob:none", worker)
        self.assertIn("git -C $SourceDir config core.autocrlf false", worker)
        self.assertNotIn("\n    git clone --filter=blob:none", worker)
        self.assertIn("$MavenRepoRoot = Join-Path $WorkRoot 'm2'", worker)
        self.assertIn("$script:ShardMavenRepo = Join-Path $MavenRepoRoot $SafeId", worker)
        self.assertIn("Test-RemoteShardSuccess", worker)
        self.assertNotIn(
            "Remove-Item -LiteralPath $MavenRepo -Recurse", worker
        )
        self.assertIn("Unregister-ScheduledTask", worker)

    def test_mingw_gcc_16_does_not_dllexport_thread_local_variables(self):
        expected = (
            "#if __GNUC__ >= 16\n"
            "#define SD_TLS_EXPORT\n"
            "#else\n"
            "#define SD_TLS_EXPORT __attribute__((dllexport))\n"
            "#endif"
        )
        for relative_path in (
            "libnd4j/include/system/common.h",
            "libnd4j/include/system/sd_export.h",
        ):
            with self.subTest(path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(expected, source)

    def test_data_buffer_tls_uses_exported_accessor(self):
        header = (ROOT / "libnd4j/include/array/DataBuffer.h").read_text(encoding="utf-8")
        self.assertIn(
            "SD_LIB_EXPORT DataBufferThreadState& dataBufferThreadState();",
            header,
        )
        self.assertIn(
            "#define tl_graphExecutionActive   dataBufferThreadState().graphExecutionActive",
            header,
        )
        self.assertNotIn(
            "extern SD_TLS_EXPORT thread_local DataBufferThreadState",
            header,
        )
        for relative_path in (
            "libnd4j/include/array/cpu/DataBuffer.cpp",
            "libnd4j/include/array/cuda/DataBuffer.cu",
            "libnd4j/include/array/vulkan/DataBuffer.cpp",
        ):
            with self.subTest(path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(
                    "SD_LIB_EXPORT DataBufferThreadState& dataBufferThreadState()",
                    source,
                )
                self.assertIn("static thread_local DataBufferThreadState state;", source)
                self.assertNotIn(
                    "SD_TLS_EXPORT thread_local DataBufferThreadState",
                    source,
                )
        native_ops = (ROOT / "libnd4j/include/legacy/cuda/NativeOps.cu").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("sd::tl_dataBufferState", native_ops)
        self.assertIn("sd::dataBufferThreadState()", native_ops)


class WorkerTransportTests(unittest.TestCase):
    def test_bucket_parser_and_blob_url_are_azure_native(self):
        self.assertEqual(
            ("dl4jaccount", "releases"),
            cloud_io.parse_bucket("dl4jaccount/releases"),
        )
        url = cloud_io.blob_url(
            "dl4jaccount/releases", "a path/shard-manifest.json"
        )
        self.assertEqual(
            "https://dl4jaccount.blob.core.windows.net/releases/"
            "a%20path/shard-manifest.json",
            url,
        )

    def test_missing_checkpoint_download_is_a_quiet_cache_miss(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "status.json"
            args = SimpleNamespace(
                bucket="dl4jaccount/releases",
                object="run/shard/status.json",
                file=str(output),
                client_id="client",
            )
            with mock.patch.object(
                cloud_io, "download_bytes", side_effect=FileNotFoundError
            ):
                self.assertEqual(1, cloud_io.command_download(args))
            self.assertFalse(output.exists())
            with mock.patch.object(
                cloud_io, "download_bytes", return_value=b'{"exitCode":0}'
            ):
                self.assertEqual(0, cloud_io.command_download(args))
            self.assertEqual(b'{"exitCode":0}', output.read_bytes())

    def test_kill_switch_is_fail_closed(self):
        args = SimpleNamespace(
            bucket="dl4jaccount/control",
            object="control/kill-switch.json",
            client_id=None,
            controller_epoch=None,
        )
        with mock.patch.object(cloud_io, "download_bytes", side_effect=FileNotFoundError):
            self.assertEqual(2, cloud_io.command_kill_enabled(args))
        with mock.patch.object(
            cloud_io, "download_bytes", return_value=b'{"enabled":false}'
        ):
            self.assertEqual(1, cloud_io.command_kill_enabled(args))
        with mock.patch.object(
            cloud_io, "download_bytes", return_value=b'{"enabled":true}'
        ):
            self.assertEqual(0, cloud_io.command_kill_enabled(args))

    def test_kill_switch_rejects_stale_epochs_but_honors_emergency_force(self):
        args = SimpleNamespace(
            bucket="dl4jaccount/control",
            object="control/kill-switch.json",
            client_id=None,
            controller_epoch="current",
        )
        with mock.patch.object(
            cloud_io,
            "download_bytes",
            return_value=b'{"enabled":false,"controllerEpoch":"stale"}',
        ):
            self.assertEqual(2, cloud_io.command_kill_enabled(args))
        with mock.patch.object(
            cloud_io,
            "download_bytes",
            return_value=b'{"enabled":true,"controllerEpoch":"stale"}',
        ):
            self.assertEqual(1, cloud_io.command_kill_enabled(args))
        with mock.patch.object(
            cloud_io,
            "download_bytes",
            return_value=(
                b'{"enabled":true,"controllerEpoch":"emergency","force":true}'
            ),
        ):
            self.assertEqual(0, cloud_io.command_kill_enabled(args))

    def test_workers_render_all_payloads_and_use_managed_identity(self):
        config = {
            "provider": "azure",
            "managedIdentityClientId": "client-id",
            "laneId": "lane",
            "shards": [{"id": "shard"}],
        }
        for name in ("worker.sh", "worker.ps1"):
            rendered = release.render_worker(HERE / name, config).decode("utf-8")
            self.assertNotRegex(rendered, r"__DL4J_[A-Z0-9_]+__")
            self.assertIn("client-id", rendered)
            self.assertIn("--client-id", rendered)
            self.assertIn("--controller-epoch", rendered)
            self.assertIn("contractDigest", rendered)
            self.assertIn("controllerEpoch", rendered)
            self.assertIn("repository", rendered)
            self.assertIn("provider", rendered)
            self.assertNotIn("provider':'gcp", rendered)
            self.assertNotIn('"provider":"gcp"', rendered)

    def test_worker_transport_uses_imds_not_storage_keys(self):
        source = (HERE / "cloud-io.py").read_text(encoding="utf-8")
        self.assertIn("169.254.169.254", source)
        self.assertIn("managed identity", source.lower())
        self.assertNotIn("account_key", source)
        self.assertNotIn("AZURE_STORAGE_KEY", source)


    def test_authenticated_requests_retry_rbac_403_and_refresh_401(self):
        for status in (401, 403):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://example.invalid/blob",
                    status,
                    "pending",
                    {},
                    io.BytesIO(b"authorization pending"),
                )
                response = mock.MagicMock()
                response.__enter__.return_value.read.return_value = b"ok"
                cloud_io._TOKEN["expires"] = 9999999999
                with mock.patch.object(
                    cloud_io, "access_token", return_value="token"
                ), mock.patch.object(
                    cloud_io.urllib.request,
                    "urlopen",
                    side_effect=[error, response],
                ) as urlopen, mock.patch.object(cloud_io.time, "sleep"):
                    self.assertEqual(
                        b"ok",
                        cloud_io.request(
                            "https://example.invalid/blob",
                            authenticated=True,
                            client_id="client",
                        ),
                    )
                self.assertEqual(2, urlopen.call_count)
                if status == 401:
                    self.assertEqual(0, cloud_io._TOKEN["expires"])

    def test_imds_retries_identity_propagation_400_and_404(self):
        errors = [
            urllib.error.HTTPError(
                cloud_io.METADATA_TOKEN_URL,
                status,
                "identity pending",
                {},
                io.BytesIO(b"identity pending"),
            )
            for status in (400, 404)
        ]
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"access_token":"token","expires_on":"9999999999"}'
        )
        cloud_io._TOKEN.update({"value": None, "expires": 0.0, "client_id": None})
        with mock.patch.object(
            cloud_io.urllib.request,
            "urlopen",
            side_effect=[*errors, response],
        ) as urlopen, mock.patch.object(cloud_io.time, "sleep"):
            self.assertEqual("token", cloud_io.access_token("identity-client"))
        self.assertEqual(3, urlopen.call_count)

    def test_append_blob_creation_accepts_existing_blob(self):
        error = urllib.error.HTTPError(
            "https://example.invalid/live.log",
            409,
            "exists",
            {},
            io.BytesIO(b"already exists"),
        )
        with mock.patch.object(
            cloud_io, "access_token", return_value="token"
        ), mock.patch.object(
            cloud_io.urllib.request, "urlopen", side_effect=error
        ):
            self.assertEqual(
                b"already exists",
                cloud_io.request(
                    "https://example.invalid/live.log",
                    method="PUT",
                    authenticated=True,
                    accepted_statuses=(409,),
                ),
            )

    def test_live_forwarder_appends_only_new_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "build.log"
            stop = Path(temp) / "stop"
            path.write_bytes(b"hello")
            stop.touch()
            args = SimpleNamespace(
                file=str(path),
                stop_file=str(stop),
                bucket="dl4jaccount/releases",
                object="prefix/live.log",
                client_id="client",
                interval=0,
            )
            with mock.patch.object(
                cloud_io, "create_append_blob"
            ) as create, mock.patch.object(
                cloud_io, "download_bytes", return_value=b""
            ), mock.patch.object(
                cloud_io, "append_bytes"
            ) as append, mock.patch.object(cloud_io.time, "sleep"):
                self.assertEqual(0, cloud_io.command_forward(args))
            create.assert_called_once_with(
                "dl4jaccount/releases", "prefix/live.log", "client"
            )
            append.assert_called_once_with(
                "dl4jaccount/releases", "prefix/live.log", b"hello", 0, "client"
            )

    def test_live_forwarder_resumes_without_duplicate_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "build.log"
            stop = Path(temp) / "stop"
            path.write_bytes(b"hello world")
            stop.touch()
            args = SimpleNamespace(
                file=str(path),
                stop_file=str(stop),
                bucket="dl4jaccount/releases",
                object="prefix/live.log",
                client_id="client",
                interval=0,
            )
            with mock.patch.object(
                cloud_io, "create_append_blob"
            ), mock.patch.object(
                cloud_io, "download_bytes", return_value=b"hello"
            ), mock.patch.object(
                cloud_io, "append_bytes"
            ) as append, mock.patch.object(cloud_io.time, "sleep"):
                self.assertEqual(0, cloud_io.command_forward(args))
            append.assert_called_once_with(
                "dl4jaccount/releases", "prefix/live.log", b" world", 5, "client"
            )

    def test_live_forwarder_reconciles_a_committed_append_with_lost_response(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "build.log"
            stop = Path(temp) / "stop"
            path.write_bytes(b"hello")
            stop.touch()
            args = SimpleNamespace(
                file=str(path),
                stop_file=str(stop),
                bucket="dl4jaccount/releases",
                object="prefix/live.log",
                client_id="client",
                interval=0,
            )
            with mock.patch.object(
                cloud_io, "create_append_blob"
            ), mock.patch.object(
                cloud_io, "download_bytes", side_effect=[b"", b"hello"]
            ) as download, mock.patch.object(
                cloud_io, "append_bytes", side_effect=RuntimeError("response lost")
            ) as append, mock.patch.object(cloud_io.time, "sleep"):
                self.assertEqual(0, cloud_io.command_forward(args))
            append.assert_called_once_with(
                "dl4jaccount/releases", "prefix/live.log", b"hello", 0, "client"
            )
            self.assertEqual(2, download.call_count)

    def test_append_uses_expected_position_and_disables_ambiguous_retries(self):
        with mock.patch.object(cloud_io, "request", return_value=b"") as request:
            cloud_io.append_bytes(
                "dl4jaccount/releases", "prefix/live.log", b"bytes", 17, "client"
            )
        self.assertEqual(
            "17",
            request.call_args.kwargs["headers"]["x-ms-blob-condition-appendpos"],
        )
        self.assertEqual(1, request.call_args.kwargs["retries"])

    def test_windows_watchdog_and_live_logs_keep_the_identity_client_id(self):
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn("param([switch]$Register)", source)
        self.assertIn(
            "$Config.controllerEpoch,$Config.managedIdentityClientId,"
            "$KillRequestedFile",
            source,
        )
        self.assertIn(
            "Set-Content -LiteralPath $KillRequestedFile -Value $Reason",
            source,
        )
        self.assertNotIn("$ActiveBuildLogFile", source)
        self.assertIn("'--client-id', $ClientId", source)
        self.assertIn("$ObjectPrefix/live.log", source)
        self.assertIn("*>> '$LaneLog'", source)
        self.assertIn("/lanes/$($Config.laneId)/live.log", source)
        self.assertIn("Start-LaneLogging", source)
        self.assertIn("Stop-LaneLogging", source)
        self.assertIn("Copy-NewLogContent $MatrixLog", source)
        self.assertIn("git -c core.autocrlf=false clone --filter=blob:none $Config.repository $SourceDir", source)
        self.assertNotIn("worktree add", source)
        controller = (HERE / "release.py").read_text(encoding="utf-8")
        self.assertIn("managed identity retained", controller)
        self.assertIn(
            '"diagnostics_profile": {"boot_diagnostics": {"enabled": True}}',
            controller,
        )
        linux = (HERE / "worker.sh").read_text(encoding="utf-8")
        self.assertIn('${CURRENT_OBJECT_PREFIX}/live.log', linux)
        self.assertIn("remote_shard_succeeded", linux)
        self.assertIn('SOURCE_ROOT=${WORK_ROOT}/sources', linux)
        self.assertIn('SCCACHE_ROOT=${SOURCE_ROOT}/sccache', linux)
        self.assertIn('${SCCACHE_ROOT}:/sccache', linux)
        self.assertIn('${SCCACHE_ROOT}:/github/sccache', linux)
        self.assertIn('MAVEN_REPO_ROOT=${WORK_ROOT}/m2', linux)
        self.assertIn('maven_repo="${MAVEN_REPO_ROOT}/${safe_id}"', linux)
        self.assertIn('git clone --filter=blob:none "${REPOSITORY}" "${source_dir}"', linux)
        self.assertNotIn('worktree add', linux)

    def test_windows_transcript_has_one_writer_for_the_build_log(self):
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn("function Write-BuildContent", source)
        self.assertIn(
            "$script:BuildLog -and -not $script:TranscriptStarted",
            source,
        )
        self.assertIn("if ($Text) { Write-BuildContent $Text }", source)
        self.assertIn("Write-BuildContent ($_ | Out-String)", source)
        self.assertNotIn("[IO.File]::AppendAllText($BuildLog", source)
        self.assertNotIn("$_ | Out-String | Add-Content $BuildLog", source)

    def test_windows_native_commands_ignore_stderr_but_check_exit_codes(self):
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn("function Invoke-NativeChecked", source)
        self.assertIn("$ErrorActionPreference = 'Continue'", source)
        self.assertIn("$ErrorActionPreference = $PreviousPreference", source)
        self.assertIn("$global:LASTEXITCODE = $null", source)
        self.assertIn("& $Command", source)
        self.assertIn("$Code = $global:LASTEXITCODE", source)
        self.assertNotIn("$LASTEXITCODE = $null\n    & $Command", source)
        self.assertIn("$SuccessCodes -notcontains [int]$Code", source)
        for description in (
            "Chocolatey toolchain installation",
            "MSYS2 toolchain installation",
            "Rust GNU toolchain installation",
            "Source clone",
            "Maven repository packaging",
            "Python 3.12 installation",
        ):
            self.assertIn(f"Invoke-NativeChecked -Description '{description}'", source)
        self.assertIn("$FallbackPython = Join-Path $env:SystemDrive 'Python312\\python.exe'", source)
        self.assertIn("Get-ChildItem 'C:\\ProgramData\\chocolatey\\lib\\maven'", source)
        self.assertIn("$env:MAVEN_HOME = $MavenHome.FullName", source)
        self.assertIn("$env:M2_HOME = $MavenHome.FullName", source)
        self.assertIn("$MavenExe = Join-Path $MavenHome.FullName 'bin\\mvn.cmd'", source)
        self.assertIn("& $MavenExe --version", source)
        self.assertIn("Invoke-NativeChecked -Description 'Maven toolchain validation'", source)
        self.assertIn("$RustBinCandidates = @((Join-Path $env:CARGO_HOME 'bin'))", source)
        self.assertIn("$RustupExe = Join-Path $RustBin 'rustup.exe'", source)
        self.assertIn("$CargoExe = Join-Path $RustBin 'cargo.exe'", source)
        self.assertIn("& $RustupExe toolchain install stable-x86_64-pc-windows-gnu", source)
        self.assertIn("& $RustupExe default stable-x86_64-pc-windows-gnu", source)
        self.assertIn("& $CargoExe install --locked cbindgen", source)
        self.assertNotRegex(source, r"(?m)^\s+rustup (?:toolchain|default)")
        self.assertIn("Write-Phase 'worker-bootstrap' 'failed'", source)
        self.assertIn("[void](Complete-Shard 1)", source)

    def test_windows_native_checked_executes_a_real_process(self):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is not available")
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        helper_start = source.index("function Invoke-NativeChecked")
        helper_end = source.index("\nfunction Invoke-KillSwitchProbe", helper_start)
        helper = source[helper_start:helper_end]
        probe = helper + r'''
$Python = $env:DL4J_TEST_NATIVE_EXE
$SuccessOutput = Invoke-NativeChecked -Description 'success probe' -Command {
  & $Python -c "import sys; print('native-warning', file=sys.stderr); print('native-ok')"
} | Out-String
if ($SuccessOutput -notmatch 'native-ok') { throw 'Native success output was lost' }
$CaughtExpectedFailure = $false
try {
  Invoke-NativeChecked -Description 'failure probe' -Command {
    & $Python -c "import sys; sys.exit(7)"
  }
}
catch {
  if ($_.Exception.Message -notmatch 'failure probe failed with exit code 7') { throw }
  $CaughtExpectedFailure = $true
}
if (-not $CaughtExpectedFailure) { throw 'Native exit code 7 was accepted' }
Write-Output 'native-helper-probe-ok'
'''
        with tempfile.TemporaryDirectory() as temp:
            probe_path = Path(temp) / "native-helper-probe.ps1"
            probe_path.write_text(probe, encoding="utf-8")
            env = os.environ.copy()
            env["DL4J_TEST_NATIVE_EXE"] = sys.executable
            completed = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-File",
                    str(probe_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        self.assertEqual(
            0,
            completed.returncode,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("native-helper-probe-ok", completed.stdout)

    def test_windows_bootstrap_failure_and_cleanup_are_fault_isolated(self):
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn("function Publish-BootstrapFailureWithoutPython", source)
        self.assertIn("function Upload-AzureBlobPowerShell", source)
        self.assertIn("'x-ms-blob-type'='BlockBlob'", source)
        self.assertIn("'x-ms-date'=[DateTime]::UtcNow.ToString('R'", source)
        self.assertIn("foreach ($Candidate in $Shards) {\n          try {", source)
        self.assertIn("function Invoke-CleanupStep", source)
        self.assertIn("Invoke-CleanupStep 'active shard finalization'", source)
        self.assertIn("Invoke-CleanupStep 'VM shutdown'", source)
        self.assertIn("Stop-Computer -Force", source)
        main_try = source.index("try {\n  if (Test-Path -LiteralPath $AttemptFile)")
        self.assertLess(main_try, source.index("[IO.File]::WriteAllBytes($ConfigFile"))
        config_parse = "$Config = Get-Content -Raw $ConfigFile | ConvertFrom-Json"
        self.assertLess(main_try, source.index(config_parse))

    def test_windows_direct_blob_fallback_builds_a_timestamped_put(self):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is not available")
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        helper_start = source.index("function Upload-AzureBlobPowerShell")
        helper_end = source.index(
            "\nfunction Publish-BootstrapFailureWithoutPython", helper_start
        )
        helper = source[helper_start:helper_end]
        probe = helper + r'''
$Config = [pscustomobject]@{bucket='account/container'}
function Invoke-WebRequest {
  param(
    [string]$Method,
    [string]$Uri,
    [hashtable]$Headers,
    [string]$InFile,
    [string]$ContentType,
    [switch]$UseBasicParsing
  )
  if ($Method -ne 'Put') { throw "Unexpected method: $Method" }
  if ($Uri -ne 'https://account.blob.core.windows.net/container/prefix/status.json') {
    throw "Unexpected URI: $Uri"
  }
  if ($Headers.Authorization -ne 'Bearer access-token') { throw 'Bearer token missing' }
  if ($Headers['x-ms-blob-type'] -ne 'BlockBlob') { throw 'Blob type missing' }
  if (-not $Headers['x-ms-date']) { throw 'Storage timestamp missing' }
  $ParsedDate = [DateTimeOffset]::MinValue
  if (-not [DateTimeOffset]::TryParseExact(
      [string]$Headers['x-ms-date'], 'R',
      [Globalization.CultureInfo]::InvariantCulture,
      [Globalization.DateTimeStyles]::AssumeUniversal,
      [ref]$ParsedDate)) { throw "Invalid storage timestamp: $($Headers['x-ms-date'])" }
  if ($ContentType -ne 'application/json') { throw "Unexpected content type: $ContentType" }
  if (-not (Test-Path -LiteralPath $InFile)) { throw "Input file missing: $InFile" }
  return [pscustomobject]@{StatusCode=201}
}
Upload-AzureBlobPowerShell $env:DL4J_TEST_BLOB_FILE 'prefix/status.json' 'access-token' 'application/json'
Write-Output 'blob-put-probe-ok'
'''
        with tempfile.TemporaryDirectory() as temp:
            blob_path = Path(temp) / "status.json"
            blob_path.write_text("{}", encoding="utf-8")
            probe_path = Path(temp) / "blob-put-probe.ps1"
            probe_path.write_text(probe, encoding="utf-8")
            env = os.environ.copy()
            env["DL4J_TEST_BLOB_FILE"] = str(blob_path)
            completed = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-File",
                    str(probe_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
        self.assertEqual(
            0,
            completed.returncode,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("blob-put-probe-ok", completed.stdout)

    def test_windows_worker_uses_python_before_machine_path_refresh(self):
        source = (HERE / "worker.ps1").read_text(encoding="utf-8")
        self.assertIn(
            "$PythonInstall = Join-Path $env:SystemDrive 'Python312'", source
        )
        self.assertIn(
            "$script:PythonExe = Join-Path $PythonInstall 'python.exe'", source
        )
        self.assertIn("& $script:PythonExe $CloudIo kill-enabled", source)
        self.assertIn("Start-Process $script:PythonExe", source)
        self.assertIn("function Wait-ForCloudAccess", source)
        self.assertIn("[DateTime]::UtcNow.AddMinutes(15)", source)
        self.assertIn("phase=azure-blob-auth status=waiting", source)
        self.assertIn("phase=azure-blob-auth status=ready", source)
        self.assertNotRegex(
            source,
            r"(?m)^\s*(?:&\s+|Start-Process\s+)?python(?:\.exe)?\s",
        )


class CliTests(unittest.TestCase):
    def test_all_operational_commands_parse(self):
        subscription = "00000000-0000-0000-0000-000000000001"
        prefix = ["--subscription", subscription, "--location", "eastus2"]
        cases = [
            ["configure"],
            ["preflight", "--shard", "linux-arm64-cpu", "--max-cores", "32"],
            [
                "start", "--version", "1.0.0-SNAPSHOT",
                "--commit", "a" * 40, "--reset-kill-switch",
            ],
            ["status", "--run-id", "run"],
            ["logs", "--run-id", "run", "--follow"],
            ["delete-logs", "--run-id", "run", "--yes"],
            [
                "collect", "--run-id", "run", "--release-tag", "tag",
                "--version", "1.0.0", "--commit", "a" * 40,
            ],
            ["stop-everything", "--wait"],
        ]
        for command in cases:
            with self.subTest(command=command[0]):
                parsed = release.parser().parse_args(prefix + command)
                self.assertTrue(callable(parsed.func))

    def test_standard_environment_names_are_documented(self):
        source = (HERE / "release.py").read_text(encoding="utf-8")
        self.assertIn("AZURE_SUBSCRIPTION_ID", source)
        self.assertIn("AZURE_LOCATION", source)
        self.assertIn("DefaultAzureCredential", source)

    def test_ci_never_enables_the_wizard(self):
        with mock.patch.dict(os.environ, {"CI": "true"}, clear=False), mock.patch.object(
            sys.stdin, "isatty", return_value=True
        ):
            self.assertFalse(release.interactive_wizard_enabled(True))


if __name__ == "__main__":
    unittest.main()
