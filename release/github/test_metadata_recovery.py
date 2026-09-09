"""Remote-only metadata recovery contracts; execute from platform-tests."""
import importlib.util
import json
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
