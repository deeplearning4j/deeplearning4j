"""Exact-byte source-repair contracts and opt-in real javac regression."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import xml.etree.ElementTree as ET

from release.github import source_recovery as repair

ROOT = Path(__file__).resolve().parents[2]


class SourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.fix = Path(self.temp.name) / "fix"
        self.output = Path(self.temp.name) / "output"
        self.output.mkdir()
        self.path = self.source / repair.PATH
        self.path.parent.mkdir(parents=True)
        self.original = b"// preserved byte\n" * 40 + repair.BEFORE + b"        super.afterRead(n);\n    }\n"
        self.fixed = repair.repaired(self.original)
        self.path.write_bytes(self.original)

    def invoke(self, **kwargs):
        params = dict(source=self.source, fix_source=self.fix, fix_commit=repair.FIX_COMMIT,
                      commit=repair.SOURCE_COMMIT, run_id=f"github-{repair.SOURCE_RUN}-1", output=self.output)
        params.update(kwargs)
        def git(command, cwd, **options):
            if command[1] == "rev-parse":
                return (repair.SOURCE_COMMIT if cwd == self.source else repair.FIX_COMMIT) + "\n"
            return self.original if cwd == self.source else self.fixed
        with patch.object(repair.subprocess, "check_output", side_effect=git):
            return repair.prepare(**params)

    def test_exact_bytes_and_distinct_provenance(self):
        provenance = self.invoke()
        self.assertEqual(self.fixed, self.path.read_bytes())
        self.assertTrue(provenance["compiledJavaChanged"])
        self.assertTrue(provenance["nativeWorkerReceiptsUnchanged"])
        self.assertEqual(repair.SOURCE_COMMIT, provenance["sourceCommit"])
        self.assertEqual(hashlib.sha256(self.fixed).hexdigest(), provenance["files"][0]["repairedSha256"])
        self.assertEqual(provenance, json.loads((self.output / "source-recovery-provenance.json").read_text()))
        self.assertFalse((self.output / "documentation-recovery-provenance.json").exists())

    def test_rejects_unaudited_pins_and_worker_run(self):
        for args in ({"commit": "a" * 40}, {"fix_commit": "b" * 40},
                     {"fix_commit": "main"}, {"run_id": "github-34357258064-1"},
                     {"run_id": f"github-{repair.SOURCE_RUN}-0"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.invoke(**args)
        self.assertEqual(self.original, self.path.read_bytes())

    def test_rejects_extra_compiled_changes(self):
        self.fixed += b"// unreviewed\n"
        with self.assertRaisesRegex(ValueError, "unaudited"):
            self.invoke()
        self.assertEqual(self.original, self.path.read_bytes())

    def test_rejects_nonpristine_source(self):
        self.path.write_bytes(self.fixed)
        with self.assertRaisesRegex(ValueError, "pristine"):
            self.invoke()

    def test_rejects_wrong_line_bytes(self):
        with self.assertRaises(ValueError):
            repair.repaired(self.original.replace(repair.BEFORE, repair.BEFORE.replace(b"n)", b"count)")))

    def test_rejects_checkout_mismatch(self):
        with patch.object(repair.subprocess, "check_output", return_value="0" * 40):
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                repair.prepare(self.source, self.fix, repair.FIX_COMMIT, repair.SOURCE_COMMIT,
                               f"github-{repair.SOURCE_RUN}-1", self.output)

    @unittest.skipUnless(os.environ.get("SOURCE_REPAIR_COMPILE_CONTRACT") == "1", "remote javac contract")
    def test_real_original_failure_and_repaired_compile(self):
        original_root = Path(os.environ["DOCUMENTATION_CONTRACT_SOURCE"])
        original = subprocess.check_output(["git", "show", f"{repair.SOURCE_COMMIT}:{repair.PATH}"], cwd=original_root)
        fixed = subprocess.check_output(["git", "show", f"{repair.FIX_COMMIT}:{repair.PATH}"], cwd=ROOT)
        self.assertEqual(repair.repaired(original), fixed)
        ns = {"m": "http://maven.apache.org/POM/4.0.0"}
        pom = ET.fromstring(subprocess.check_output(["git", "show", f"{repair.SOURCE_COMMIT}:pom.xml"], cwd=original_root))
        def prop(name):
            return pom.findtext(f"m:properties/m:{name}", namespaces=ns)
        jars = []
        for group, artifact, version in (("commons-io", "commons-io", prop("commonsio.version")),
                                         ("org.slf4j", "slf4j-api", prop("slf4j.version")),
                                         ("me.tongfei", "progressbar", "0.5.5")):
            self.assertTrue(version)
            jar = self.output / f"{artifact}-{version}.jar"
            url = f"https://repo.maven.apache.org/maven2/{group.replace('.', '/')}/{artifact}/{version}/{jar.name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                jar.write_bytes(response.read())
            jars.append(str(jar))
        java = self.output / "ProgressInputStream.java"
        command = ["javac", "-proc:none", "-classpath", os.pathsep.join(jars), "-d", str(self.output), str(java)]
        java.write_bytes(original)
        failed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(failed.stdout)
        self.assertNotEqual(0, failed.returncode, "original must reproduce checked IOException failure")
        self.assertIn("IOException", failed.stdout)
        java.write_bytes(fixed)
        passed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(passed.stdout)
        self.assertEqual(0, passed.returncode, passed.stdout)
        self.assertTrue((self.output / "org/eclipse/deeplearning4j/omnihub/ProgressInputStream.class").is_file())
        print(f"Verified actual original javac failure and repaired compilation: {repair.FIX_COMMIT}")
