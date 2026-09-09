"""Remote-only metadata recovery contracts; execute from platform-tests."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from release.github import metadata_recovery as metadata


class MetadataRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.fix, self.output = (self.root / p for p in ('source', 'fix', 'output'))
        self.output.mkdir()
        self.original = b'<project xmlns="http://maven.apache.org/POM/4.0.0"><profiles/></project>'
        self.fixed = self.original.replace(b'<profiles/>', b'<profiles><profile><id>central-release</id></profile></profiles>')
        for root, content in ((self.source, self.original), (self.fix, self.fixed)):
            module = root / metadata.MODULE
            module.mkdir(parents=True)
            (module / 'pom.xml').write_bytes(content)
        self.files = {'include/tokenizers_c.h': b'API', 'include/tokenizers_ffi.h': b'FFI',
                      'src/tokenizers_c.cpp': b'source', 'tokenizers-ffi/src/lib.rs': b'rust'}
        for name, content in self.files.items():
            path = self.source / metadata.MODULE / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def producer(self, command, **kwargs):
        self.assertEqual(['jar:jar@central-native-sources', 'jar:jar@central-native-javadoc'], command[-2:])
        self.assertNotIn('package', command)
        target = self.source / metadata.MODULE / 'target'
        target.mkdir()
        for kind in ('sources', 'javadoc'):
            with ZipFile(target / f'libtokenizers-1.0.0-SNAPSHOT-{kind}.jar', 'w') as archive:
                for name, content in self.files.items():
                    if kind == 'sources' or name.startswith('include/'):
                        archive.writestr(name, content)
                if kind == 'sources':
                    archive.writestr('pom.xml', self.fixed)

    def prepare(self):
        return metadata.prepare(self.source, self.fix, 'b' * 40, 'a' * 40, '1.0.0-M3', self.output)

    def test_documentation_packaging_overlay_is_exact_and_cumulative(self):
        original = b'<project>\n</project>\n'
        path = self.source / metadata.DOCUMENTATION_POM
        path.parent.mkdir(parents=True)
        path.write_bytes(original)
        fixed = metadata.documentation_pom(original)
        provenance = {'sourceCommit': metadata.SOURCE_COMMIT, 'packagingFixCommit': 'b' * 40,
                      'files': [{'path': 'prior-tokenizer-metadata'}]}
        with patch.object(metadata.subprocess, 'check_output', side_effect=[
                metadata.SOURCE_COMMIT, 'b' * 40, original, fixed]):
            metadata.prepare_documentation_config(self.source, self.fix, 'b' * 40,
                metadata.SOURCE_COMMIT, self.output, provenance)
        self.assertEqual(fixed, path.read_bytes())
        self.assertEqual([{'path': 'prior-tokenizer-metadata'}], provenance['files'])
        self.assertTrue(provenance['documentationPackaging']['compileSourceRootsUnchanged'])
        self.assertEqual(provenance, json.loads((self.output / 'metadata-recovery-provenance.json').read_text()))

    def test_documentation_packaging_rejects_extra_changes(self):
        original = b'<project>\n</project>\n'
        path = self.source / metadata.DOCUMENTATION_POM
        path.parent.mkdir(parents=True)
        path.write_bytes(original)
        for fixed in (metadata.documentation_pom(original).replace(b'false', b'true'),
                      metadata.documentation_pom(original) + b'<!-- extra -->', original):
            with self.subTest(fixed=fixed), patch.object(metadata.subprocess, 'check_output', side_effect=[
                    metadata.SOURCE_COMMIT, 'b' * 40, original, fixed]):
                with self.assertRaises(ValueError):
                    metadata.prepare_documentation_config(self.source, self.fix, 'b' * 40,
                        metadata.SOURCE_COMMIT, self.output, {})
            self.assertEqual(original, path.read_bytes())

    @unittest.skipUnless(os.environ.get('DOCUMENTATION_CONTRACT_SOURCE'), 'remote original checkout required')
    def test_real_documentation_pom_matches_audited_insertion(self):
        source = Path(os.environ['DOCUMENTATION_CONTRACT_SOURCE'])
        root = Path(__file__).resolve().parents[2]
        original = (source / metadata.DOCUMENTATION_POM).read_bytes()
        fixed = (root / metadata.DOCUMENTATION_POM).read_bytes()
        self.assertEqual(metadata.documentation_pom(original), fixed)
        profile = metadata.ET.fromstring(metadata.DOCUMENTATION_PROFILE)
        plugins = profile.find('profile/build/plugins')
        lombok, javadoc = list(plugins)
        config = lombok.find('executions/execution/configuration')
        self.assertEqual('false', config.findtext('addOutputDirectory'))
        self.assertEqual('process-classes', lombok.findtext('executions/execution/phase'))
        self.assertEqual('${project.build.directory}/delombok-javadoc', javadoc.findtext('configuration/sourcepath'))

    @unittest.skipUnless(os.environ.get('RELEASE_DELOMBOK_CONTRACT') == '1', 'remote Maven contract only')
    def test_real_delombok_and_javadoc_private_builder_signatures(self):
        # Exercise the exact audited profile on Java 21, without the native reactor.
        root = Path(__file__).resolve().parents[2]
        model = metadata.ET.parse(root / 'pom.xml').getroot()
        lombok_version = model.findtext('m:properties/m:lombok.version', namespaces=metadata.NS)
        pom = '''<project xmlns="http://maven.apache.org/POM/4.0.0">
<modelVersion>4.0.0</modelVersion><groupId>release.contract</groupId>
<artifactId>delombok-contract</artifactId><version>1</version>
<properties><lombok.version>LOMBOK_VERSION</lombok.version></properties>
<dependencies><dependency><groupId>org.projectlombok</groupId><artifactId>lombok</artifactId>
<version>${lombok.version}</version><scope>provided</scope></dependency></dependencies>
<build><plugins><plugin><groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-compiler-plugin</artifactId><version>3.13.0</version>
<configuration><release>21</release></configuration></plugin>
<plugin><groupId>org.apache.maven.plugins</groupId><artifactId>maven-javadoc-plugin</artifactId>
<version>3.10.1</version><configuration><failOnError>true</failOnError></configuration>
<executions><execution><id>central-javadoc</id><phase>package</phase>
<goals><goal>jar</goal></goals></execution></executions></plugin></plugins></build>
</project>
'''.replace('LOMBOK_VERSION', lombok_version).encode()
        (self.root / 'pom.xml').write_bytes(metadata.documentation_pom(pom))
        sources = self.root / 'src/main/java/example'
        sources.mkdir(parents=True)
        for name in ('TorchScriptGraph', 'TorchScriptMetadata', 'ArchitectureConfig'):
            (sources / (name + '.java')).write_text(
                'package example;\nimport lombok.Builder;\n/** Builder documentation contract. */\n'
                '@Builder public class ' + name + ' {\n'
                'private int value;\nprivate static void configure(' + name + 'Builder builder) {}\n}\n')
        (sources / 'Reader.java').write_text(
            'package example;\n/** External builder signature contract. */\npublic class Reader {\n'
            'private void extract(TorchScriptGraph.TorchScriptGraphBuilder builder) {}\n}\n')
        subprocess.run(['mvn', '--batch-mode', '--no-transfer-progress', '-f', str(self.root / 'pom.xml'),
                        '-Pcentral-release', 'install', '-DskipTests'], check=True)
        for name in ('TorchScriptGraph', 'TorchScriptMetadata', 'ArchitectureConfig'):
            generated = self.root / 'target/delombok-javadoc/example' / (name + '.java')
            self.assertIn('class ' + name + 'Builder', generated.read_text())
        archive = self.root / 'target/delombok-contract-1-javadoc.jar'
        with ZipFile(archive) as docs:
            self.assertIn('example/TorchScriptGraph.TorchScriptGraphBuilder.html', docs.namelist())

    def test_profile_only_changes_are_required(self):
        metadata.validate_pom(self.original, self.fixed)
        with self.assertRaises(ValueError):
            metadata.validate_pom(self.original, self.fixed.replace(b'<profiles>', b'<version>2</version><profiles>'))
        with self.assertRaises(ValueError):
            metadata.validate_pom(self.original, self.original)

    def test_recovery_records_both_commits_and_restores_original_pom(self):
        with patch.object(metadata, 'tree', return_value={'source': 'blob'}), \
             patch.object(metadata.subprocess, 'check_output', return_value='b' * 40), \
             patch.object(metadata.subprocess, 'run', side_effect=self.producer):
            supplements, provenance = self.prepare()
        self.assertEqual(2, len(supplements))
        self.assertEqual(self.original, (self.source / metadata.MODULE / 'pom.xml').read_bytes())
        self.assertEqual('a' * 40, provenance['sourceCommit'])
        self.assertEqual('b' * 40, provenance['packagingFixCommit'])
        self.assertEqual(provenance, json.loads((self.output / 'metadata-recovery-provenance.json').read_text()))
        self.assertTrue(all(path.exists() for path in supplements.values()))

    def test_failure_restores_original_source(self):
        with patch.object(metadata, 'tree', return_value={'source': 'blob'}), \
             patch.object(metadata.subprocess, 'check_output', return_value='b' * 40), \
             patch.object(metadata.subprocess, 'run', side_effect=RuntimeError('producer failed')):
            with self.assertRaises(RuntimeError):
                self.prepare()
        self.assertEqual(self.original, (self.source / metadata.MODULE / 'pom.xml').read_bytes())
        self.assertFalse((self.output / 'metadata-recovery-provenance.json').exists())

    def test_changed_native_source_is_rejected_before_packaging(self):
        with patch.object(metadata, 'tree', side_effect=[{'source': 'old'}, {'source': 'old'}, {'source': 'new'}]), \
             patch.object(metadata.subprocess, 'check_output', return_value='b' * 40), \
             patch.object(metadata.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                self.prepare()
            run.assert_not_called()

    def test_supplement_only_fills_missing_metadata_and_never_rewrites_workers(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('metadata_full_repository', root / 'release/github/full-repository.py')
        full = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(full)
        worker = self.root / 'worker'
        directory = full.component_path('libtokenizers', '1.0.0-M3')
        (worker / directory).mkdir(parents=True)
        for suffix, data in (('.pom', b'<project/>'), ('.jar', b'native')):
            (worker / directory / ('libtokenizers-1.0.0-M3' + suffix)).write_bytes(data)
        supplements = {}
        for kind in ('sources', 'javadoc'):
            path = self.root / (kind + '.jar')
            path.write_bytes(kind.encode())
            supplements[(directory / f'libtokenizers-1.0.0-M3-{kind}.jar').as_posix()] = path
        shards = {'cpu': {'build': {'variants': [{'name': 'base'}]}}}
        with patch.object(full, 'shared_owners', return_value={'libtokenizers': ('cpu', 'base')}), \
             patch.object(full.worker, 'plan_shards', return_value=shards), \
             patch.object(full, 'selected_contract', return_value=({}, {})), \
             patch.object(full.driver, 'attest_variant_classifier_artifacts'), \
             patch.object(full, 'classifier_components', return_value=()):
            ownership = full.seed_native({('cpu', 'base'): worker}, {}, self.root / 'local', '1.0.0-M3', supplements)
        for path in supplements:
            self.assertEqual('metadata-recovery', ownership[path])
            self.assertFalse((worker / path).exists())
            self.assertTrue(supplements[path].exists())

    def test_wrong_fix_revision_is_rejected(self):
        with patch.object(metadata.subprocess, 'check_output', return_value='c' * 40):
            with self.assertRaises(ValueError):
                self.prepare()
        with self.assertRaises(ValueError):
            metadata.prepare(self.source, self.fix, 'main', 'a' * 40, '1.0.0-M3', self.output)
