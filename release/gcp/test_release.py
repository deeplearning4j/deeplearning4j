#!/usr/bin/env python3
"""Offline tests for the Google Cloud release controller and matrix parity."""

import importlib.util
import inspect
import io
import json
from pathlib import Path
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


release = load_module("dl4j_gcp_release", ROOT / "release/gcp/release.py")
cloud_io = load_module("dl4j_gcp_cloud_io", ROOT / "release/gcp/cloud-io.py")

try:
    from google.cloud import compute_v1, tpu_v2
except ImportError:
    compute_v1 = None
    tpu_v2 = None


class NotFound(RuntimeError):
    pass


class FakeMachineClient:
    def __init__(self, values):
        self.values = values

    def get(self, *, project, zone, machine_type):
        key = (zone, machine_type)
        if key not in self.values:
            raise NotFound(f"missing machine type: {key}")
        cpus, memory_mb = self.values[key]
        return types.SimpleNamespace(guest_cpus=cpus, memory_mb=memory_mb)


class ReleasePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gcp = release.load_plan(ROOT / "release/gcp/release-plan.json")
        cls.aws = json.loads((ROOT / "release/aws/release-plan.json").read_text(encoding="utf-8"))

    def test_gcp_covers_every_aws_lane_except_unavailable_macos(self):
        gcp_ids = {item["id"] for item in self.gcp["shards"]}
        aws_ids = {item["id"] for item in self.aws["shards"]}
        self.assertEqual(aws_ids - {"macos-14-arm64-cpu"}, gcp_ids)
        self.assertIn("build-deploy-mac-arm64.yml", self.gcp["unsupportedWorkflows"])

    def test_build_matrix_is_byte_for_byte_equivalent_to_aws(self):
        aws = {item["id"]: item for item in self.aws["shards"]}
        for gcp_shard in self.gcp["shards"]:
            aws_shard = aws[gcp_shard["id"]]
            self.assertEqual(aws_shard["build"], gcp_shard["build"], gcp_shard["id"])
            self.assertEqual(aws_shard["workloads"], gcp_shard["workloads"], gcp_shard["id"])
            self.assertEqual(aws_shard["artifactRules"], gcp_shard["artifactRules"], gcp_shard["id"])

    def test_full_matrix_is_serial_lanes_not_one_vm_per_variant(self):
        executions = release.selected_executions(self.gcp)
        self.assertEqual(len(self.gcp["shards"]), len(executions))
        cpu = next(item for item in executions if item["id"] == "linux-x86_64-cpu")
        self.assertEqual(
            ["base", "avx2", "avx512", "onednn", "onednn-avx2", "onednn-avx512", "compile"],
            [item["name"] for item in cpu["build"]["variants"]],
        )

    def test_specific_variant_selects_only_that_variant(self):
        selected = release.selected_executions(self.gcp, ["linux-x86_64-cpu--avx2"])
        self.assertEqual(1, len(selected))
        self.assertEqual("linux-x86_64-cpu--avx2", selected[0]["id"])
        self.assertEqual(["avx2"], [item["name"] for item in selected[0]["build"]["variants"]])

    def test_excluding_cpu_base_retains_every_other_cpu_classifier(self):
        selected = release.selected_executions(self.gcp, exclusions=["linux-x86_64-cpu--base"])
        cpu = next(item for item in selected if item["id"] == "linux-x86_64-cpu")
        self.assertEqual(
            ["avx2", "avx512", "onednn", "onednn-avx2", "onednn-avx512", "compile"],
            [item["name"] for item in cpu["build"]["variants"]],
        )
        self.assertEqual(len(self.gcp["shards"]), len(selected))

    def test_unknown_selector_fails_before_cloud_calls(self):
        with self.assertRaisesRegex(ValueError, "unknown shard"):
            release.selected_executions(self.gcp, ["not-real"])
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            release.selected_executions(self.gcp, ["linux-x86_64-cpu--not-real"])

    def test_compile_only_accelerators_do_not_request_accelerator_hardware(self):
        alternative = {"cuda", "tpu", "vulkan", "hexagon"}
        for shard in self.gcp["shards"]:
            backend = shard["build"]["backend"]
            if backend in alternative or shard["id"].endswith("zluda"):
                self.assertNotIn("guestAccelerators", shard, shard["id"])
                self.assertIn(shard["machineClass"], {"x86", "arm"})
        tpu = next(item for item in self.gcp["shards"] if item["id"] == "linux-x86_64-tpu")
        self.assertEqual("x86", tpu["machineClass"])

    def test_official_image_families_are_architecture_specific(self):
        arm = next(item for item in self.gcp["shards"] if item["id"] == "linux-arm64-cpu")
        windows = next(item for item in self.gcp["shards"] if item["id"] == "windows-x86_64-cpu")
        self.assertEqual(("ubuntu-os-cloud", "ubuntu-2204-lts-arm64"), (arm["imageProject"], arm["imageFamily"]))
        self.assertEqual(("windows-cloud", "windows-2022"), (windows["imageProject"], windows["imageFamily"]))

    def test_tpu_defaults_are_a_live_validated_single_chip_v5e_configuration(self):
        settings = self.gcp["tpuSmoke"]
        self.assertEqual("v5litepod-1", settings["acceleratorType"])
        self.assertEqual("v2-alpha-tpuv5-lite", settings["runtimeVersion"])
        self.assertEqual("us-central1-a", settings["defaultZone"])
        self.assertEqual(
            {"us-central1-a", "us-south1-a", "us-west1-c", "us-west4-a", "europe-west4-b"},
            set(settings["zones"]),
        )

    def test_every_variant_has_a_unique_name(self):
        for shard in self.gcp["shards"]:
            names = [item["name"] for item in shard["build"]["variants"]]
            self.assertEqual(len(names), len(set(names)), shard["id"])

    def test_hybrid_coverage_requires_every_variant_and_macos(self):
        expected = release.matrix_coverage(self.aws, [item["id"] for item in self.aws["shards"]])
        gcp_only = release.matrix_coverage(self.aws, [item["id"] for item in self.gcp["shards"]])
        self.assertEqual(
            {
                "macos-14-arm64-cpu--base", "macos-14-arm64-cpu--compile",
                "macos-14-arm64-cpu--mps", "macos-14-arm64-cpu--mps-compile",
            },
            expected - gcp_only,
        )
        hybrid = gcp_only | release.matrix_coverage(self.aws, ["macos-14-arm64-cpu"])
        self.assertEqual(expected, hybrid)

    def test_one_selected_variant_does_not_claim_a_complete_lane(self):
        covered = release.matrix_coverage(self.aws, ["linux-x86_64-cpu--base"])
        self.assertIn("linux-x86_64-cpu--base", covered)
        self.assertNotIn("linux-x86_64-cpu--avx2", covered)

    def test_legacy_aws_manifest_is_identified_as_hybrid_when_merged(self):
        self.assertEqual("gcp", release.merged_release_provider(None))
        self.assertEqual("gcp", release.merged_release_provider({"provider": "gcp"}))
        self.assertEqual("hybrid", release.merged_release_provider({"shards": ["linux-x86_64-cpu"]}))
        self.assertEqual("hybrid", release.merged_release_provider({"provider": "hybrid"}))


class SchedulingTests(unittest.TestCase):
    def test_machine_selection_is_greedy_within_constraint(self):
        client = FakeMachineClient({
            ("us-central1-a", "c4-highcpu-96"): (96, 196608),
            ("us-central1-a", "c4-highcpu-48"): (48, 98304),
            ("us-central1-a", "c4-highcpu-32"): (32, 65536),
        })
        result = release.choose_machine_live(
            client, "project", ["us-central1-a"],
            ["c4-highcpu-96", "c4-highcpu-48", "c4-highcpu-32"], 48,
        )
        self.assertEqual("c4-highcpu-48", result["machineType"])
        self.assertEqual(48, result["vcpus"])
        self.assertEqual(
            ["c4-highcpu-48", "c4-highcpu-32"],
            [item["machineType"] for item in result["launchAlternatives"]],
        )

    def test_machine_selection_retains_every_verified_zone_for_capacity_fallback(self):
        client = FakeMachineClient({
            ("us-central1-a", "c4-highcpu-16"): (16, 32768),
            ("us-central1-b", "c4-highcpu-16"): (16, 32768),
            ("us-central1-b", "c4-highcpu-8"): (8, 16384),
        })
        result = release.choose_machine_live(
            client, "project", ["us-central1-a", "us-central1-b"],
            ["c4-highcpu-16", "c4-highcpu-8"], 16,
        )
        self.assertEqual(["us-central1-a", "us-central1-b"], result["launchAlternatives"][0]["zones"])
        self.assertEqual("c4-highcpu-8", result["launchAlternatives"][1]["machineType"])

    def test_impossible_core_constraint_fails_before_launch(self):
        client = FakeMachineClient({("us-central1-a", "c4-highcpu-8"): (8, 16384)})
        with self.assertRaisesRegex(RuntimeError, "max-cores=4"):
            release.choose_machine_live(client, "project", ["us-central1-a"], ["c4-highcpu-8"], 4)

    def test_build_threads_and_heap_scale_to_selected_machine(self):
        shard = {"build": {"buildThreads": 48, "mavenHeapGiB": 32}}
        release.adapt_build_resources(shard, 16, 32, None)
        self.assertEqual(8, shard["build"]["buildThreads"])
        self.assertEqual(8, shard["build"]["mavenHeapGiB"])

    def test_serial_quota_uses_largest_lane_not_sum(self):
        region = types.SimpleNamespace(quotas=[])
        report = release.quota_report(region, [
            {"machineType": "c4-highcpu-96", "vcpus": 96},
            {"machineType": "c4-highcpu-48", "vcpus": 48},
            {"machineType": "c4-highcpu-96", "vcpus": 96},
        ], {"C4": 96})
        self.assertEqual(96, report["checks"]["C4"]["required"])
        self.assertEqual([], report["failures"])

    def test_quota_failure_explains_required_and_remaining(self):
        region = types.SimpleNamespace(quotas=[])
        report = release.quota_report(
            region, [{"machineType": "c4-highcpu-48", "vcpus": 48}], {"C4": 16}
        )
        self.assertIn("needs 48 vCPUs", report["failures"][0])
        self.assertIn("limit is 16", report["failures"][0])

    def test_cloud_quota_dimensions_select_region_and_machine_family(self):
        infos = [types.SimpleNamespace(
            metric="compute.googleapis.com/cpus_per_vm_family",
            dimensions_infos=[
                types.SimpleNamespace(
                    dimensions={"region": "us-central1", "vm_family": "C4"},
                    applicable_locations=["us-central1"], details=types.SimpleNamespace(value=96),
                ),
                types.SimpleNamespace(
                    dimensions={"region": "europe-west4", "vm_family": "C4"},
                    applicable_locations=["europe-west4"], details=types.SimpleNamespace(value=48),
                ),
            ],
        )]
        self.assertEqual({"C4": 96.0}, release.cloud_family_limits(infos, "us-central1"))


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = release.load_plan(ROOT / "release/gcp/release-plan.json")

    def test_linux_worker_renders_all_payloads_and_uses_shared_driver(self):
        shard = next(item for item in self.plan["shards"] if item["id"] == "linux-x86_64-cpu")
        config = {
            "project": "p", "region": "us-central1", "bucket": "b", "artifactPrefix": "x",
            "runId": "r", "releaseVersion": "1", "snapshotVersion": "1-SNAPSHOT",
            "commit": "a" * 40, "repository": "https://example/repo.git",
            "killSwitchBucket": "control", "killSwitchObject": "kill",
            "logId": "log", "shard": shard,
        }
        rendered = release.render_worker(ROOT / "release/gcp/worker.sh", config)
        self.assertNotRegex(rendered, r"__DL4J_[A-Z0-9_]+__")
        self.assertIn("dl4j-build-platform.py", rendered)
        self.assertIn("kill-enabled", rendered)
        self.assertIn("KILL_SWITCH_BUCKET", rendered)
        self.assertIn("cloud-logging", rendered)

    def test_windows_worker_has_mid_build_kill_polling(self):
        text = (ROOT / "release/gcp/worker.ps1").read_text(encoding="utf-8")
        self.assertIn("while (-not $Process.HasExited)", text)
        self.assertIn("Test-KillSwitch", text)
        self.assertIn("Start-KillWatchdog", text)
        self.assertIn("Start-Job", text)
        self.assertIn("taskkill /PID", text)
        self.assertIn("-RedirectStandardOutput $MatrixLog", text)
        self.assertNotIn("-RedirectStandardOutput $BuildLog -RedirectStandardError", text)

    def test_tpu_worker_copies_github_action_smoke_arguments(self):
        text = (ROOT / "release/gcp/tpu-worker.sh").read_text(encoding="utf-8")
        for token in [
            "-Ptpu,test-tpu", "-Dbackend.artifactId=nd4j-native",
            "-Djavacpp.platform=linux-x86_64", "-Dtest=TpuBackendSmokeTest",
            "TPU_LIBRARY_PATH", "unset LD_PRELOAD",
        ]:
            self.assertIn(token, text)

    def test_cloud_storage_object_names_are_url_encoded(self):
        url = cloud_io.object_url("bucket", "a path/with+symbols", upload=False)
        self.assertIn("a%20path%2Fwith%2Bsymbols", url)

    def test_kill_switch_transport_is_fail_closed(self):
        args = types.SimpleNamespace(bucket="bucket", object="control/kill.json")
        with mock.patch.object(cloud_io.sys, "stderr", new=io.StringIO()):
            for payload, expected in [({"enabled": True}, 0), ({"enabled": False}, 1), ({}, 2)]:
                with self.subTest(payload=payload), mock.patch.object(
                    cloud_io, "download_bytes", return_value=json.dumps(payload).encode("utf-8")
                ):
                    self.assertEqual(expected, cloud_io.command_kill_enabled(args))
            with mock.patch.object(cloud_io, "download_bytes", side_effect=RuntimeError("network unavailable")):
                self.assertEqual(2, cloud_io.command_kill_enabled(args))
            with mock.patch.object(cloud_io, "download_bytes", side_effect=FileNotFoundError()):
                self.assertEqual(2, cloud_io.command_kill_enabled(args))

    def test_controller_explicitly_uses_c4_hyperdisk_and_gvnic(self):
        text = (ROOT / "release/gcp/release.py").read_text(encoding="utf-8")
        self.assertIn("diskTypes/hyperdisk-balanced", text)
        self.assertIn('nic_type="GVNIC"', text)

    def test_serial_console_uses_generated_request_cursor_shape(self):
        text = (ROOT / "release/gcp/release.py").read_text(encoding="utf-8")
        self.assertIn("get_serial_port_output(request={", text)
        self.assertIn('getattr(serial, "next_"', text)

    def test_tpu_delete_uses_supported_v2_signature(self):
        text = (ROOT / "release/gcp/release.py").read_text(encoding="utf-8")
        self.assertIn("delete_node(name=", text)
        self.assertNotIn("delete_node(name=node.name, force=", text)

    def test_emergency_tpu_discovery_consumes_every_location_page(self):
        class Client:
            def __init__(self):
                self.requests = []

            def list_locations(self, *, request):
                self.requests.append(dict(request))
                if "page_token" not in request:
                    return types.SimpleNamespace(
                        locations=[types.SimpleNamespace(location_id="us-central1-a", name="")],
                        next_page_token="next",
                    )
                return types.SimpleNamespace(
                    locations=[types.SimpleNamespace(location_id="", name="projects/p/locations/europe-west4-b")],
                    next_page_token="",
                )

        client = Client()
        self.assertEqual(["europe-west4-b", "us-central1-a"], release.discover_tpu_zones(client, "p"))
        self.assertEqual("next", client.requests[1]["page_token"])

    def test_bucket_setup_disables_versioning_and_soft_delete(self):
        text = (ROOT / "release/gcp/release.py").read_text(encoding="utf-8")
        self.assertIn("versioning_enabled = False", text)
        self.assertIn('"retentionDurationSeconds": "0"', text)
        self.assertIn("BUCKET_MANAGED_LABEL", text)

    def test_shutdown_bucket_discovery_requires_the_managed_label(self):
        self.assertTrue(release.is_managed_bucket(types.SimpleNamespace(labels={release.BUCKET_MANAGED_LABEL: "true"})))
        self.assertFalse(release.is_managed_bucket(types.SimpleNamespace(labels={})))
        self.assertFalse(release.is_managed_bucket(types.SimpleNamespace(labels=None)))

    def test_worker_bucket_access_is_scoped_and_idempotent(self):
        policy = types.SimpleNamespace(bindings=[])
        bucket = mock.Mock()
        bucket.get_iam_policy.return_value = policy
        release.ensure_worker_bucket_access(bucket, "worker@example.iam.gserviceaccount.com")
        release.ensure_worker_bucket_access(bucket, "worker@example.iam.gserviceaccount.com")
        self.assertEqual(
            [{
                "role": "roles/storage.objectAdmin",
                "members": ["serviceAccount:worker@example.iam.gserviceaccount.com"],
            }],
            policy.bindings,
        )
        bucket.set_iam_policy.assert_called_once_with(policy)
        viewer_policy = types.SimpleNamespace(bindings=[])
        viewer_bucket = mock.Mock()
        viewer_bucket.get_iam_policy.return_value = viewer_policy
        release.ensure_worker_bucket_access(
            viewer_bucket, "worker@example.iam.gserviceaccount.com", "roles/storage.objectViewer"
        )
        self.assertEqual("roles/storage.objectViewer", viewer_policy.bindings[0]["role"])
        self.assertEqual(
            "123-compute@developer.gserviceaccount.com",
            release.worker_service_account_email(None, "123"),
        )

    def test_control_bucket_name_is_project_wide_not_regional(self):
        self.assertEqual("dl4j-release-project-control", release.control_bucket_name("project"))
        self.assertNotEqual(
            release.control_bucket_name("project"), release.release_bucket_name("project", "us-central1")
        )

    def test_central_workflow_rejects_missing_or_non_boolean_completeness(self):
        text = (ROOT / ".github/workflows/publish-central-from-release.yml").read_text(encoding="utf-8")
        self.assertIn("if complete is not True or missing:", text)
        self.assertIn("manifest.get('testMavenRepository', {}).get('completeMatrix')", text)

    @unittest.skipUnless(compute_v1 is not None and tpu_v2 is not None, "Google Cloud SDKs not installed")
    def test_generated_google_sdk_resource_shapes(self):
        item = {
            "zone": "us-central1-a",
            "machineType": "c4-highcpu-16",
            "instanceName": "dl4j-test",
            "network": {"network": "global/networks/default", "subnetwork": None},
            "image": {"selfLink": "projects/ubuntu-os-cloud/global/images/test"},
            "planDefaults": {"rootVolumeGiB": 1000},
            "planProjectLabel": "deeplearning4j-release",
            "runId": "run",
            "shard": {"id": "linux-x86_64-cpu", "os": "linux"},
        }
        args = types.SimpleNamespace(root_volume_gib=None, service_account=None)
        instance = release.instance_resource(
            {"compute": compute_v1, "project": "project"}, args, item, "#!/bin/bash\ntrue\n"
        )
        self.assertEqual("GVNIC", instance.network_interfaces[0].nic_type)
        self.assertTrue(instance.disks[0].initialize_params.disk_type.endswith("/hyperdisk-balanced"))
        node = tpu_v2.Node(
            accelerator_type="v5litepod-1",
            runtime_version="v2-alpha-tpuv5-lite",
            metadata={"startup-script": "true"},
            network_config=tpu_v2.NetworkConfig(enable_external_ips=True),
            scheduling_config=tpu_v2.SchedulingConfig(preemptible=False),
        )
        self.assertEqual("v5litepod-1", node.accelerator_type)
        self.assertNotIn("force", inspect.signature(tpu_v2.TpuClient.delete_node).parameters)


class CliTests(unittest.TestCase):
    def test_all_operational_commands_parse(self):
        command_sets = [
            ["preflight", "--max-cores", "96"],
            ["start", "--branch", "main", "--version", "1.0.0", "--exclude-shard", "linux-x86_64-cpu--base"],
            ["status", "--run-id", "run"],
            ["logs", "--run-id", "run", "--follow"],
            ["delete-logs", "--run-id", "run", "--yes"],
            ["collect", "--run-id", "run", "--release-tag", "tag", "--version", "1.0.0", "--commit", "a" * 40],
            ["tpu-smoke", "--branch", "main"],
            ["stop-everything", "--wait", "--purge-logs"],
        ]
        parser = release.parser()
        for argv in command_sets:
            with self.subTest(argv=argv):
                self.assertTrue(parser.parse_args(argv).command)

    def test_default_region_environment_names_are_standard(self):
        text = (ROOT / "release/gcp/release.py").read_text(encoding="utf-8")
        self.assertIn("GOOGLE_CLOUD_REGION", text)
        self.assertIn("CLOUDSDK_COMPUTE_REGION", text)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS", (ROOT / "release/gcp/README.md").read_text(encoding="utf-8"))


class ShutdownTests(unittest.TestCase):
    def test_storage_failure_cannot_prevent_direct_compute_and_tpu_deletion(self):
        storage_client = mock.Mock()
        storage_client.lookup_bucket.side_effect = RuntimeError("storage unavailable")
        storage_client.list_buckets.side_effect = RuntimeError("storage unavailable")
        compute_client = mock.Mock()
        compute_client.delete.return_value = mock.Mock()
        tpu_client = mock.Mock()
        tpu_client.delete_node.return_value = mock.Mock()
        context = {
            "project": "project",
            "credentials": object(),
            "storage": types.SimpleNamespace(Client=mock.Mock(return_value=storage_client)),
            "compute": types.SimpleNamespace(InstancesClient=mock.Mock(return_value=compute_client)),
            "tpu": types.SimpleNamespace(TpuClient=mock.Mock(return_value=tpu_client)),
        }
        args = types.SimpleNamespace(
            plan=ROOT / "release/gcp/release-plan.json", project="project", region="us-central1",
            bucket=None, tpu_zone=None, wait=False, purge_logs=False, purge_storage=False,
        )
        instance = types.SimpleNamespace(name="vm", labels={})
        node = types.SimpleNamespace(name="projects/project/locations/us-central1-a/nodes/tpu")
        with (
            mock.patch.object(release, "cloud_context", return_value=context),
            mock.patch.object(release, "ensure_control_bucket", side_effect=RuntimeError("storage unavailable")),
            mock.patch.object(release, "managed_instances", return_value=[("us-central1-a", instance)]),
            mock.patch.object(release, "list_tpu_nodes", return_value=([node], [])),
            mock.patch.object(release.sys, "stdout", new=io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "completed best-effort"),
        ):
            release.stop_everything(args)
        compute_client.delete.assert_called_once_with(project="project", zone="us-central1-a", instance="vm")
        tpu_client.delete_node.assert_called_once_with(name=node.name)


if __name__ == "__main__":
    unittest.main()
