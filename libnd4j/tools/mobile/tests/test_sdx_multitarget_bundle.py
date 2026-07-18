#!/usr/bin/env python3
"""Tests for canonical SDX bundles with multiple mobile target artifacts."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


COMPILER = Path(__file__).resolve().parents[2] / "sdx-compile.sh"


class MultiTargetBundleTest(unittest.TestCase):

    def test_packages_canonical_graph_vulkan_and_tensor_derivative(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sdx-multitarget-") as temporary:
            root = Path(temporary)
            model = root / "model.sdz"
            model.write_bytes(b"canonical-samediff-model")

            spirv = root / "spirv"
            spirv.mkdir()
            spirv_payload = struct.pack("<5I", 0x07230203, 0x00010000, 0, 1, 0)
            (spirv / "spv_0123456789abcdef.spv").write_bytes(spirv_payload)
            (spirv / "spv_0123456789abcdef.meta").write_text(
                "cacheAbi=vulkan-spirv-disk-cache-v2\n"
                "descriptorBindings=0;1\n"
                f"spirvWords={len(spirv_payload) // 4}\n",
                encoding="utf-8",
            )

            tensor_model = root / "model.litertlm"
            tensor_payload = b"tensor-g5-derived-model"
            tensor_model.write_bytes(tensor_payload)
            bundle = root / "model.dspb-dir"
            packed = root / "model.dspb"

            subprocess.run(
                [
                    str(COMPILER),
                    "--input",
                    str(model),
                    "--output",
                    str(bundle),
                    "--packed-output",
                    str(packed),
                    "--targets",
                    "android-arm64",
                    "--backends",
                    "VULKAN,HEXAGON,TPU",
                    "--gpu-target",
                    "vulkan",
                    "--vulkan-spirv-dir",
                    str(spirv),
                    "--tensor-g5-model",
                    str(tensor_model),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("graph/model.sdz", manifest["modelPath"])
            self.assertEqual(
                "artifacts/vulkan/spirv",
                manifest["compiledArtifacts"]["vulkanSpirv"],
            )
            tensor = manifest["compiledArtifacts"]["tensorG5LiteRtLm"]
            self.assertEqual("artifacts/tensor-g5/model.litertlm", tensor["path"])
            self.assertEqual(hashlib.sha256(tensor_payload).hexdigest(), tensor["sha256"])
            self.assertEqual(
                tensor_payload,
                (bundle / tensor["path"]).read_bytes(),
            )
            with zipfile.ZipFile(packed) as archive:
                self.assertEqual(
                    tensor_payload,
                    archive.read("artifacts/tensor-g5/model.litertlm"),
                )
                self.assertEqual(
                    manifest,
                    json.loads(archive.read("manifest.json")),
                )
                self.assertTrue(all(
                    entry.date_time == (1980, 1, 1, 0, 0, 0)
                    for entry in archive.infolist()
                ))
            expected_sidecar = (
                f"{hashlib.sha256(packed.read_bytes()).hexdigest()}  {packed.name}\n"
            )
            self.assertEqual(
                expected_sidecar,
                packed.with_suffix(".dspb.sha256").read_text(encoding="utf-8"),
            )

            second_bundle = root / "second.dspb-dir"
            second_packed = root / "second.dspb"
            subprocess.run(
                [
                    str(COMPILER),
                    "--input",
                    str(model),
                    "--output",
                    str(second_bundle),
                    "--packed-output",
                    str(second_packed),
                    "--targets",
                    "android-arm64",
                    "--backends",
                    "VULKAN,HEXAGON,TPU",
                    "--gpu-target",
                    "vulkan",
                    "--vulkan-spirv-dir",
                    str(spirv),
                    "--tensor-g5-model",
                    str(tensor_model),
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(packed.read_bytes(), second_packed.read_bytes())

    def test_rejects_non_litertlm_tensor_derivative(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sdx-multitarget-invalid-") as temporary:
            root = Path(temporary)
            model = root / "model.sdz"
            model.write_bytes(b"canonical-samediff-model")
            invalid = root / "model.bin"
            invalid.write_bytes(b"not-litertlm")
            result = subprocess.run(
                [
                    str(COMPILER),
                    "--input",
                    str(model),
                    "--output",
                    str(root / "bundle"),
                    "--tensor-g5-model",
                    str(invalid),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn(".litertlm", result.stderr)


if __name__ == "__main__":
    unittest.main()
