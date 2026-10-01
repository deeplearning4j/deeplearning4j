"""Synthetic-only receipt, JAR and Sonatype metadata contracts; no Maven/network."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import warnings
import xml.etree.ElementTree as ET
import zipfile

SPEC = importlib.util.spec_from_file_location("tokenizer_publication", Path(__file__).with_name("tokenizer-publication.py"))
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)
SHA = "a" * 40
PROVENANCE = dict(commit=SHA, run_id="12345", run_attempt=2, version=pub.VERSION)
STAMP = "1.0.0-20260814.123456-1"


def pom(artifact):
    parent = "nd4j-tokenizers" if artifact in pub.ARTIFACTS else {
        "nd4j-tokenizers": "nd4j", "nd4j": "deeplearning4j", "deeplearning4j": None,
    }[artifact]
    ancestry = (f"<parent><groupId>{pub.GROUP}</groupId><artifactId>{parent}</artifactId>"
                f"<version>{pub.VERSION}</version></parent>") if parent else ""
    return (f'<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
            f"{ancestry}<groupId>{pub.GROUP}</groupId><artifactId>{artifact}</artifactId>"
            f"<version>{pub.VERSION}</version><packaging>{'jar' if artifact in pub.ARTIFACTS else 'pom'}</packaging>"
            "<description>same API</description></project>").encode()


def binary(host):
    if host.startswith("linux-"):
        result = bytearray(128)
        result[:6] = b"\x7fELF\x02\x01"
        result[18:20] = (62 if host == "linux-x86_64" else 183).to_bytes(2, "little")
        return bytes(result)
    if host.startswith("windows-"):
        result = bytearray(128)
        result[:2] = b"MZ"
        result[60:64] = (64).to_bytes(4, "little")
        result[64:70] = b"PE\0\0\x64\x86"
        return bytes(result)
    return b"\xcf\xfa\xed\xfe" + (0x0100000c).to_bytes(4, "little") + b"native bytes"


def natives(artifact, host, root=pub.RESOURCE_ROOTS[0]):
    prefix = root + ("" if artifact == "libtokenizers" else "bindings/") + host + "/"
    ext = ".so" if host.startswith("linux-") else (".dll" if host.startswith("windows-") else ".dylib")
    result = {prefix + "libtokenizers_wrapper" + ext: binary(host)}
    if artifact != "libtokenizers":
        result[prefix + ("" if host.startswith("windows-") else "lib") + "jnitokenizers" + ext] = binary(host)
    else:
        result[prefix + "manifest.properties"] = ("build.javacpp.platform=" + host).encode()
        result[root + "include/tokenizers_c.h"] = b"int tokenize(void);"
    return result


def jar(path, entries, host="linux-x86_64"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name, (2025, 1, 1 if host == pub.CANONICAL else 2, 0, 0, 0))
            archive.writestr(info, data)


def installed(root, host, docs=True):
    for artifact in (*pub.PARENTS, *pub.ARTIFACTS):
        path = root / pub.pom_path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pom(artifact))
        if artifact not in pub.ARTIFACTS:
            continue
        common = {"META-INF/MANIFEST.MF": ("Built-By: " + host).encode(),
                  f"META-INF/maven/{pub.GROUP}/{artifact}/pom.xml": pom(artifact)}
        main = dict(common)
        if artifact == "libtokenizers":
            main.update(natives(artifact, host))
        else:
            main[pub.RESOURCE_ROOTS[0] + artifact + ".class"] = b"shared JVM API"
        jar(root / pub.artifact_path(artifact), main, host)
        jar(root / pub.artifact_path(artifact, host), {**common, **natives(artifact, host)}, host)
        if docs:
            for classifier in ("sources", "javadoc"):
                jar(root / pub.artifact_path(artifact, classifier), {classifier + "/api.txt": b"same public API"}, host)
    # Deliberately unrelated installed dependencies/metadata must never be collected.
    (root / "unrelated.jar").write_bytes(b"unrelated")
    extra = root / pub.GROUP_PATH / "nd4j-api" / pub.VERSION / "nd4j-api-1.0.0-SNAPSHOT.jar"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unrelated dependency")


def metadata(artifact, identities=None, namespace=""):
    if identities is None:
        identities = [("", "pom")]
        if artifact in pub.ARTIFACTS:
            identities += [("", "jar"), *((h, "jar") for h in pub.HOSTS), ("sources", "jar"), ("javadoc", "jar")]
    root = ET.Element("metadata", {"xmlns": namespace} if namespace else {})
    for name, value in (("groupId", pub.GROUP), ("artifactId", artifact), ("version", pub.VERSION)):
        ET.SubElement(root, name).text = value
    versions = ET.SubElement(ET.SubElement(root, "versioning"), "snapshotVersions")
    for classifier, extension in identities:
        entry = ET.SubElement(versions, "snapshotVersion")
        if classifier:
            ET.SubElement(entry, "classifier").text = classifier
        for name, value in (("extension", extension), ("value", STAMP), ("updated", "20260814123456")):
            ET.SubElement(entry, name).text = value
    return ET.tostring(root)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.inputs = self.base / "inputs"
        self.output = self.base / "merged"
        for host in pub.HOSTS:
            source = self.base / "installed" / host
            installed(source, host)
            pub.collect(source, self.inputs / host, host, **PROVENANCE)

    def receipt(self, host=pub.CANONICAL):
        path = self.inputs / host / pub.RECEIPT
        return path, json.loads(path.read_text())

    def edit_receipt(self, change, host=pub.CANONICAL):
        path, data = self.receipt(host)
        change(data)
        path.write_text(json.dumps(data))

    def rewrite_jar(self, host, artifact, update, classifier=""):
        path = self.inputs / host / pub.artifact_path(artifact, classifier)
        with zipfile.ZipFile(path) as archive:
            entries = {i.filename: archive.read(i) for i in archive.infolist()}
        update(entries)
        jar(path, entries, host)
        self.refresh(host, path)

    def refresh(self, host, path):
        def change(receipt):
            for i, record in enumerate(receipt["files"]):
                if record["path"] == path.relative_to(self.inputs / host).as_posix():
                    receipt["files"][i] = pub.file_record(path, self.inputs / host)
                    return
        self.edit_receipt(change, host)

    def do_merge(self):
        return pub.merge(self.inputs, self.output, **PROVENANCE)

    def fetch_metadata(self, url):
        artifact = url.split("/")[-3]
        return metadata(artifact)

    def test_collect_scope_hashes_parent_closure_and_documentation(self):
        root = self.inputs / pub.CANONICAL
        receipt, files = pub.load_receipt(root)
        self.assertEqual(receipt["classifier"], pub.CANONICAL)
        self.assertEqual(len(files), 18)
        self.assertEqual(pub.closure(root, pub.VERSION), set((*pub.PARENTS, *pub.ARTIFACTS)))
        self.assertFalse(any("nd4j-api" in n or "unrelated" in n for n in files))
        for r in receipt["files"]:
            self.assertEqual(r, pub.file_record(root / r["path"], root))

    def test_merge_explicit_canonical_no_repack_and_verify(self):
        receipt = self.do_merge()
        self.assertEqual(receipt["canonicalMainOwner"], "linux-x86_64")
        self.assertEqual(receipt["classifiers"], list(pub.HOSTS))
        self.assertEqual(len(receipt["files"]), 27)
        for artifact in pub.ARTIFACTS:
            name = pub.artifact_path(artifact)
            self.assertEqual((self.output / name).read_bytes(), (self.inputs / pub.CANONICAL / name).read_bytes())
        self.assertEqual(pub.verify(self.output, **PROVENANCE), receipt)

    def test_merge_missing_host(self):
        shutil.rmtree(self.inputs / "macosx-arm64")
        with self.assertRaisesRegex(ValueError, "four host"):
            self.do_merge()
        self.assertFalse(self.output.exists())

    def test_duplicate_host_receipt(self):
        shutil.copytree(self.inputs / pub.CANONICAL, self.inputs / "duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate/unknown host"):
            self.do_merge()

    def test_provenance_mismatches(self):
        for key, value in (("commit", "b" * 40), ("runId", "987"), ("runAttempt", 3), ("version", "2-SNAPSHOT")):
            with self.subTest(key=key):
                path, original = self.receipt()
                changed = dict(original, **{key: value})
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.do_merge()
                path.write_text(json.dumps(original))

    def test_checksum_mismatch(self):
        (self.inputs / pub.CANONICAL / pub.artifact_path("tokenizers-native")).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum/size"):
            self.do_merge()

    def test_missing_classified_jar_even_when_receipt_is_consistent(self):
        host = "linux-arm64"
        name = pub.artifact_path("tokenizers-native", host).as_posix()
        (self.inputs / host / name).unlink()
        self.edit_receipt(lambda r: r.update(files=[f for f in r["files"] if f["path"] != name]), host)
        with self.assertRaisesRegex(ValueError, "missing scoped"):
            self.do_merge()

    def test_receipted_unrelated_coordinate_is_rejected(self):
        root = self.inputs / pub.CANONICAL
        extra = root / pub.GROUP_PATH / "unrelated" / pub.VERSION / f"unrelated-{pub.VERSION}.pom"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"unrelated")
        self.edit_receipt(lambda r: r["files"].append(pub.file_record(extra, root)))
        with self.assertRaisesRegex(ValueError, "unexpected/missing scoped"):
            self.do_merge()

    def test_malformed_receipt_field_types_fail_closed(self):
        path, original = self.receipt()
        for key, value in (("size", "10"), ("sha256", 42)):
            changed = json.loads(json.dumps(original))
            changed["files"][0][key] = value
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "invalid digest/size"):
                self.do_merge()
        path.write_text(json.dumps(original))

    def test_duplicate_receipt_file(self):
        self.edit_receipt(lambda r: r["files"].append(r["files"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate receipt path"):
            self.do_merge()

    def test_duplicate_json_keys(self):
        path, _ = self.receipt()
        path.write_text('{"schemaVersion":1,"schemaVersion":1}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            self.do_merge()

    def test_receipt_traversal_and_unreceipted(self):
        self.edit_receipt(lambda r: r["files"][0].update(path="../outside"))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.do_merge()

    def test_unreceipted_payload(self):
        (self.inputs / pub.CANONICAL / "unexpected.pom").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "unreceipted"):
            self.do_merge()

    def test_wrong_pom_coordinates_and_parent_chain(self):
        for old, new in ((pub.GROUP.encode(), b"wrong.group"), (b"nd4j-tokenizers", b"unrelated-parent")):
            with self.subTest(new=new):
                path = self.inputs / pub.CANONICAL / pub.pom_path("tokenizers-native")
                original = path.read_bytes()
                path.write_bytes(original.replace(old, new))
                self.refresh(pub.CANONICAL, path)
                with self.assertRaises(ValueError):
                    self.do_merge()
                path.write_bytes(original)
                self.refresh(pub.CANONICAL, path)

    def test_genuine_class_conflict(self):
        self.rewrite_jar("linux-arm64", "tokenizers-native", lambda e: e.update({pub.RESOURCE_ROOTS[0] + "tokenizers-native.class": b"conflicting API"}))
        with self.assertRaisesRegex(ValueError, "shared JAR content conflict"):
            self.do_merge()

    def test_libtokenizers_header_conflict(self):
        for classifier in ("", "linux-arm64"):
            self.rewrite_jar("linux-arm64", "libtokenizers", lambda e: e.update({pub.RESOURCE_ROOTS[0] + "include/tokenizers_c.h": b"changed header"}), classifier)
        with self.assertRaisesRegex(ValueError, "shared JAR content conflict"):
            self.do_merge()

    def test_classifier_header_must_match_own_main(self):
        self.rewrite_jar("linux-arm64", "libtokenizers", lambda e: e.update({pub.RESOURCE_ROOTS[0] + "include/tokenizers_c.h": b"changed header"}), "linux-arm64")
        with self.assertRaisesRegex(ValueError, "header conflict"):
            self.do_merge()

    def test_classifier_api_must_match_shared_main(self):
        self.rewrite_jar("linux-arm64", "tokenizers-native", lambda e: e.update({pub.RESOURCE_ROOTS[0] + "tokenizers-native.class": b"different class"}), "linux-arm64")
        with self.assertRaisesRegex(ValueError, "classifier JVM API"):
            self.do_merge()

    def test_classifier_native_must_match_own_libtokenizers_main(self):
        prefix = pub.RESOURCE_ROOTS[0] + "linux-arm64/"
        self.rewrite_jar("linux-arm64", "libtokenizers", lambda e: e.update({prefix + "libtokenizers_wrapper.so": binary("linux-arm64") + b"different native"}), "linux-arm64")
        with self.assertRaisesRegex(ValueError, "native/API content conflict"):
            self.do_merge()

    def test_unknown_libtokenizers_build_entry_not_exempt(self):
        self.rewrite_jar("linux-arm64", "libtokenizers", lambda e: e.update({"unknown.properties": b"different"}))
        with self.assertRaisesRegex(ValueError, "shared JAR content conflict"):
            self.do_merge()

    def test_genuine_pom_content_conflict(self):
        path = self.inputs / "linux-arm64" / pub.pom_path("nd4j")
        path.write_bytes(path.read_bytes().replace(b"same API", b"changed model"))
        self.refresh("linux-arm64", path)
        with self.assertRaisesRegex(ValueError, "POM content conflict"):
            self.do_merge()

    def test_embedded_external_pom_model_conflict(self):
        embedded = f"META-INF/maven/{pub.GROUP}/tokenizers-native/pom.xml"
        self.rewrite_jar("linux-arm64", "tokenizers-native", lambda e: e.update({embedded: pom("tokenizers-native").replace(b"same API", b"different model")}))
        with self.assertRaisesRegex(ValueError, "embedded/external POM content conflict"):
            self.do_merge()

    def test_wrong_embedded_coordinates(self):
        self.rewrite_jar("linux-arm64", "tokenizers-native", lambda e: e.update({f"META-INF/maven/{pub.GROUP}/tokenizers-native/pom.xml": pom("tokenizers-native-preset")}))
        with self.assertRaisesRegex(ValueError, "embedded POM"):
            self.do_merge()

    def test_unsafe_zip_members(self):
        for name in ("../escape", "/absolute", "C:/drive", "a\\b", "a/./b"):
            with self.subTest(name=name):
                path = self.base / "unsafe.jar"
                jar(path, {name: b"x"})
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    pub.jar_entries(path, "tokenizers-native")

    def test_duplicate_zip_and_symlink(self):
        path = self.base / "unsafe.jar"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("entry", b"1")
                archive.writestr("entry", b"2")
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            pub.jar_entries(path, "tokenizers-native")
        with zipfile.ZipFile(path, "w") as archive:
            item = zipfile.ZipInfo("symlink")
            item.create_system = 3
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, b"../escape")
        with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
            pub.jar_entries(path, "tokenizers-native")

    def test_native_wrong_arch_and_missing_jni(self):
        host = "linux-arm64"
        classifier = pub.artifact_path("tokenizers-native", host)
        path = self.inputs / host / classifier
        original = path.read_bytes()
        for mode in ("arch", "missing"):
            def change(entries):
                name = next(n for n in entries if "jnitokenizers.so" in n)
                if mode == "arch":
                    entries[name] = binary(pub.CANONICAL)
                else:
                    del entries[name]
            self.rewrite_jar(host, "tokenizers-native", change, host)
            with self.assertRaises(ValueError):
                self.do_merge()
            path.write_bytes(original)
            self.refresh(host, path)

    def test_foreign_platform_payload(self):
        self.rewrite_jar("linux-arm64", "libtokenizers", lambda e: e.update(natives("libtokenizers", pub.CANONICAL)), "linux-arm64")
        with self.assertRaisesRegex(ValueError, "foreign native"):
            self.do_merge()

    def test_legacy_resource_package_not_rewritten(self):
        path = self.base / "legacy.jar"
        jar(path, natives("tokenizers-native", pub.CANONICAL, pub.RESOURCE_ROOTS[1]))
        pub.jar_entries(path, "tokenizers-native", pub.CANONICAL)

    def test_docs_must_match_and_no_first_writer(self):
        self.rewrite_jar("linux-arm64", "tokenizers-native", lambda e: e.update({"sources/api.txt": b"different"}), "sources")
        with self.assertRaisesRegex(ValueError, "shared JAR content conflict"):
            self.do_merge()

    def test_verify_ownership_and_missing_host_attestation(self):
        self.do_merge()
        path = self.output / pub.RECEIPT
        original = json.loads(path.read_text())
        for key, value in (("canonicalMainOwner", "linux-arm64"), ("hostReceipts", {})):
            path.write_text(json.dumps(dict(original, **{key: value})))
            with self.assertRaises(ValueError):
                pub.verify(self.output, **PROVENANCE)

    def test_deploy_commands_all_hosts_docs_and_parent_order(self):
        self.do_merge()
        commands = pub.deploy_commands(self.output, pub.VERSION, "/absolute/mvn")
        self.assertEqual(len(commands), 6)
        for artifact, command in zip((*pub.PARENTS, *pub.ARTIFACTS), commands):
            self.assertTrue(any(arg.endswith(pub.pom_path(artifact).as_posix()) for arg in command))
            self.assertIn("-DrepositoryId=central-portal-snapshots", command)
            self.assertIn("-Durl=" + pub.URL, command)
            self.assertEqual(sum("deploy-file" in arg for arg in command), 1)
            if artifact in pub.ARTIFACTS:
                classifiers = next(a.split("=", 1)[1].split(",") for a in command if a.startswith("-Dclassifiers="))
                self.assertEqual(set(classifiers), set((*pub.HOSTS, "sources", "javadoc")))
            else:
                self.assertFalse(any(a.startswith("-Dclassifiers=") for a in command))

    def test_preflight_valid_namespace_and_all_touched_gavs(self):
        self.do_merge()
        with patch.object(pub, "fetch", side_effect=lambda url: metadata(url.split("/")[-3], namespace="http://maven.apache.org/METADATA/1.1.0")) as get:
            result = pub.preflight(self.output, pub.VERSION)
        self.assertEqual(set(result), set((*pub.ARTIFACTS, *pub.PARENTS)))
        self.assertEqual(get.call_count, 6)

    def test_remote_extra_any_gav_blocks_before_any_upload(self):
        self.do_merge()
        for artifact in (*pub.PARENTS, *pub.ARTIFACTS):
            with self.subTest(artifact=artifact):
                def fetch(url):
                    a = url.split("/")[-3]
                    if a == artifact:
                        return metadata(a, [("", "pom"), ("already-advertised-extra", "jar")])
                    return metadata(a)
                with patch.object(pub, "fetch", side_effect=fetch), patch.object(pub.subprocess, "run") as run:
                    with self.assertRaisesRegex(ValueError, "incomplete snapshot classifiers"):
                        pub.deploy(self.output, pub.VERSION)
                    run.assert_not_called()

    def test_advertised_sources_missing_is_failclosed(self):
        self.do_merge()
        path = self.output / pub.artifact_path("tokenizers-native", "sources")
        path.unlink()
        receipt_path = self.output / pub.RECEIPT
        receipt = json.loads(receipt_path.read_text())
        receipt["files"] = [r for r in receipt["files"] if r["path"] != path.relative_to(self.output).as_posix()]
        receipt_path.write_text(json.dumps(receipt))
        with patch.object(pub, "fetch", side_effect=self.fetch_metadata), patch.object(pub.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "incomplete snapshot classifiers"):
                pub.deploy(self.output, pub.VERSION)
            run.assert_not_called()

    def test_404_only_permitted_missing_metadata(self):
        self.do_merge()
        with patch.object(pub, "fetch", side_effect=FileNotFoundError):
            self.assertEqual(len(pub.preflight(self.output, pub.VERSION)), 6)
        for error in (RuntimeError("network error"), ValueError("malformed")):
            with patch.object(pub, "fetch", side_effect=error), patch.object(pub.subprocess, "run") as run:
                with self.assertRaises(type(error)):
                    pub.deploy(self.output, pub.VERSION)
                run.assert_not_called()

    def test_deploy_preflights_all_then_six_mocked_invocations(self):
        self.do_merge()
        events = []
        def fetch(url):
            events.append("metadata")
            return self.fetch_metadata(url)
        def run(command, **kwargs):
            events.append("upload")
            self.assertTrue(kwargs["check"])
            self.assertFalse("clean" in command or "install" in command)
        with patch.object(pub, "fetch", side_effect=fetch), patch.object(pub.subprocess, "run", side_effect=run):
            pub.deploy(self.output, pub.VERSION)
        self.assertEqual(events, ["metadata"] * 6 + ["upload"] * 6)
        pub.verify(self.output, **PROVENANCE)  # retained payload untouched

    def test_remote_verify_exact_receipted_bytes_and_mismatch(self):
        self.do_merge()
        def fetch(url):
            if url.endswith("maven-metadata.xml"):
                return self.fetch_metadata(url)
            relative = url[len(pub.URL):].replace(STAMP, pub.VERSION)
            return (self.output / relative).read_bytes()
        with patch.object(pub, "fetch", side_effect=fetch):
            pub.verify(self.output, **PROVENANCE, remote=True)
        with patch.object(pub, "fetch", side_effect=lambda url: fetch(url) if url.endswith(".xml") else b"wrong"):
            with self.assertRaisesRegex(ValueError, "remote artifact hash/size"):
                pub.verify(self.output, **PROVENANCE, remote=True)

    def test_no_overwrite_output(self):
        self.output.mkdir()
        sentinel = self.output / "unrelated"
        sentinel.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "already exist"):
            self.do_merge()
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_cli_dispatch_required_flags_without_execution(self):
        for action in ("collect", "merge", "verify", "deploy"):
            argv = [action, "--version", pub.VERSION]
            argv += ["--inputs", str(self.inputs)] if action == "merge" else ["--repository", str(self.output)]
            if action in ("collect", "merge"):
                argv += ["--output", str(self.output)]
            if action == "collect":
                argv += ["--classifier", pub.CANONICAL]
            if action != "deploy":
                argv += ["--commit", SHA, "--run-id", "12345", "--run-attempt", "2"]
            with patch.object(pub, action) as target, contextlib.redirect_stdout(io.StringIO()):
                pub.main(argv)
                target.assert_called_once()


class MetadataTests(unittest.TestCase):
    def test_strict_parser_rejects_malformed_ambiguous_xml(self):
        data = metadata("tokenizers-native")
        mutants = [b"not XML", b"<!DOCTYPE metadata [<!ENTITY x 'bad'>]>" + data,
                   ("<!DOCTYPE metadata [<!ENTITY x 'bad'>]>" + data.decode()).encode("utf-16"),
                   data.replace(b"<metadata>", b'<metadata xmlns="http://maven.apache.org/POM/4.0.0">'),
                   data.replace(pub.GROUP.encode(), b"wrong.group"),
                   data.replace(b"<version>1.0.0-SNAPSHOT</version>", b"<version>2-SNAPSHOT</version>"),
                   data.replace(b"<extension>jar</extension>", b"<extension>jar</extension><extension>jar</extension>", 1),
                   data.replace(b"<extension>jar</extension>", b"<extension><nested>jar</nested></extension>", 1),
                   data.replace(b"<updated>20260814123456</updated>", b"<updated>20261314123456</updated>", 1),
                   data.replace(STAMP.encode(), b"2.0.0-20260814.123456-1", 1),
                   data.replace(b"</snapshotVersions>", b"</snapshotVersions><snapshotVersions/>", 1),
                   data.replace(b"<snapshotVersions>", b"<snapshotVersions><unexpected/>", 1),
                   data.replace(b"<classifier>sources</classifier>", b"<classifier>../sources</classifier>", 1),
                   data.replace(b"<classifier>sources</classifier>", b"<classifier/>", 1),
                   data.replace(b"<value>" + STAMP.encode() + b"</value>", b"<value/>", 1)]
        for mutant in mutants:
            with self.subTest(xml=mutant[:60]):
                with self.assertRaises(ValueError):
                    pub.snapshot_entries(mutant, "tokenizers-native", pub.VERSION)

    def test_namespace_mixing_is_invalid(self):
        data = metadata("tokenizers-native", namespace="http://maven.apache.org/METADATA/1.1.0")
        data = data.replace(b"<extension>", b'<extension xmlns="">', 1)
        with self.assertRaisesRegex(ValueError, "mixed XML"):
            pub.snapshot_entries(data, "tokenizers-native", pub.VERSION)

    def test_duplicate_identity_and_missing_snapshot_versions(self):
        for data in (metadata("tokenizers-native", [("", "pom"), ("sources", "jar"), ("sources", "jar")]),
                     metadata("tokenizers-native", []),
                     metadata("tokenizers-native", [("", "jar")])):
            with self.assertRaises(ValueError):
                pub.snapshot_entries(data, "tokenizers-native", pub.VERSION)

    def test_safe_http_errors_never_leak_provider_material(self):
        for code in (401, 403, 500):
            error = urllib.error.HTTPError("https://secret@example.invalid", code, "password=secret", {}, None)
            with patch.object(pub.urllib.request, "urlopen", side_effect=error):
                with self.assertRaises(RuntimeError) as caught:
                    pub.fetch(pub.URL)
                self.assertNotIn("secret", str(caught.exception))
        with patch.object(pub.urllib.request, "urlopen", side_effect=urllib.error.HTTPError(pub.URL, 404, "missing", {}, None)):
            with self.assertRaises(FileNotFoundError):
                pub.fetch(pub.URL)


if __name__ == "__main__":
    unittest.main()
