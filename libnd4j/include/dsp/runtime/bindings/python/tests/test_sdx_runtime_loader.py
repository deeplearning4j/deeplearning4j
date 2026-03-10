# ******************************************************************************
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

import pathlib
import sys
import tempfile
import unittest

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from sdx_runtime import (
    _backend_priority,
    _build_library_search_plan,
    _runtime_library_filename,
    detect_host_platform_id,
)


class SdxRuntimeLoaderTests(unittest.TestCase):
    def test_detect_host_platform_id_normalization(self):
        self.assertEqual(
            detect_host_platform_id(sys_platform="linux", machine="x86_64", os_name="posix", environ={}),
            "linux-x86_64",
        )
        self.assertEqual(
            detect_host_platform_id(sys_platform="win32", machine="AMD64", os_name="nt", environ={}),
            "windows-x86_64",
        )
        self.assertEqual(
            detect_host_platform_id(sys_platform="darwin", machine="arm64", os_name="posix", environ={}),
            "macos-arm64",
        )
        self.assertEqual(
            detect_host_platform_id(
                sys_platform="linux",
                machine="aarch64",
                os_name="posix",
                environ={"ANDROID_ROOT": "/system", "ANDROID_DATA": "/data"},
            ),
            "android-arm64",
        )

    def test_runtime_library_filename_by_platform(self):
        self.assertEqual(_runtime_library_filename("linux-x86_64", "cpu"), "libnd4jcpu.so")
        self.assertEqual(_runtime_library_filename("macos-arm64", "cuda"), "libnd4jcuda.dylib")
        self.assertEqual(_runtime_library_filename("windows-x86_64", "amd"), "nd4jamd.dll")

    def test_backend_priority_validation(self):
        self.assertEqual(_backend_priority("cuda"), ["cuda", "cpu", "amd"])
        with self.assertRaisesRegex(ValueError, "Invalid backend preference"):
            _backend_priority("bogus")

    def test_build_plan_prefers_library_dir_and_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            lib_dir = tmp_path / "runtime-lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            (lib_dir / "libnd4jcuda.so").write_bytes(b"x")
            (lib_dir / "libnd4jcpu.so").write_bytes(b"x")

            plan = _build_library_search_plan(
                explicit_library=None,
                platform_id="linux-x86_64",
                preferred_backend="cuda",
                module_dir=tmp_path / "module",
                environ={"SDX_RUNTIME_LIBRARY_DIR": str(lib_dir)},
            )

            self.assertGreaterEqual(len(plan), 3)
            self.assertEqual(plan[0], str(lib_dir / "libnd4jcuda.so"))
            self.assertIn(str(lib_dir / "libnd4jcpu.so"), plan)
            self.assertIn("nd4jcuda", plan)

    def test_build_plan_uses_sdx_runtime_home_bindings_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            lib_path = tmp_path / "bindings" / "linux-x86_64" / "amd" / "lib"
            lib_path.mkdir(parents=True, exist_ok=True)
            expected = lib_path / "libnd4jamd.so"
            expected.write_bytes(b"x")

            plan = _build_library_search_plan(
                explicit_library=None,
                platform_id="linux-x86_64",
                preferred_backend="amd",
                module_dir=tmp_path / "module",
                environ={"SDX_RUNTIME_HOME": str(tmp_path)},
            )

            self.assertIn(str(expected), plan)
            self.assertLess(plan.index(str(expected)), plan.index("nd4jamd"))

    def test_explicit_library_is_first(self):
        plan = _build_library_search_plan(
            explicit_library="/opt/custom/libnd4jcpu.so",
            platform_id="linux-x86_64",
            preferred_backend=None,
            module_dir=pathlib.Path("/tmp/module"),
            environ={},
        )
        self.assertEqual(plan[0], "/opt/custom/libnd4jcpu.so")


if __name__ == "__main__":
    unittest.main()
