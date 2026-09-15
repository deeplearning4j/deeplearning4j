import json
from pathlib import Path
import tempfile
import unittest
from release.central.repository import merge


class CanonicalWorkerMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def worker(self, name, payload):
        worker = self.root / name
        repository = worker / 'maven-repository'
        folder = repository / 'org/eclipse/deeplearning4j/libtokenizers/1.0.0-rewrite'
        folder.mkdir(parents=True)
        (worker / 'worker-success').touch()
        (worker / 'worker-config.json').write_text(json.dumps({
            'commit': 'a' * 40, 'releaseVersion': '1.0.0-rewrite',
            'shard': {'id': name, 'build': {'variants': [{'name': 'base'}]}}}))
        (worker / 'build-result.json').write_text(json.dumps({'completedVariants': ['base']}))
        (folder / 'libtokenizers-1.0.0-rewrite-javadoc.jar').write_bytes(payload)
        (folder / f'libtokenizers-1.0.0-rewrite-{name}.jar').write_bytes(payload)
        return repository

    def test_owner_is_independent_of_input_order(self):
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'canonical')
        for index, inputs in enumerate(([arm, x64], [x64, arm])):
            output = self.root / f'out-{index}'
            result = merge(inputs, output, self.root / f'manifest-{index}.json',
                           '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
            self.assertEqual(3, len(result['files']))
            self.assertEqual(b'canonical', next(output.rglob('*-javadoc.jar')).read_bytes())

    def test_build_only_aggregator_is_not_published(self):
        repo = self.worker('linux-x86_64-cpu', b'canonical')
        folder = repo / 'org/eclipse/deeplearning4j/libnd4j/1.0.0-rewrite'
        folder.mkdir(parents=True)
        (folder / 'libnd4j-1.0.0-rewrite.pom').write_text('<project/>')
        (repo / 'build-plugin.pom').write_text(
            '<project><build><plugins><plugin><dependencies><dependency>'
            '<groupId>org.eclipse.deeplearning4j</groupId><artifactId>libnd4j</artifactId>'
            '</dependency></dependencies></plugin></plugins></build></project>')
        output = self.root / 'out'
        merge([repo], output, self.root / 'manifest.json',
              '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
        self.assertFalse((output / 'org/eclipse/deeplearning4j/libnd4j').exists())
        consumer = repo / 'consumer.pom'
        consumer.write_text('<project><parent><groupId>org.eclipse.deeplearning4j</groupId>'
                            '<artifactId>libnd4j</artifactId></parent></project>')
        with self.assertRaisesRegex(ValueError, 'requires build-only'):
            merge([repo], self.root / 'out2', self.root / 'manifest2.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)

    def test_audited_launcher_pair_only(self):
        from release.central.source_identity import check, CUDA_SOURCE, CPU_SOURCE
        check(CUDA_SOURCE, CPU_SOURCE)
        check(CPU_SOURCE, CUDA_SOURCE)
        with self.assertRaises(ValueError):
            check(CUDA_SOURCE, 'b' * 40)

    def test_opt_in_source_zip_profile(self):
        repo = self.worker('linux-x86_64-cpu', b'canonical')
        folder = repo / 'org/eclipse/deeplearning4j/libnd4j/1.0.0-rewrite'
        folder.mkdir(parents=True)
        (folder / 'libnd4j-1.0.0-rewrite.pom').write_text('<project/>')
        profile = ('<project><profiles><profile><id>libnd4j-assembly</id>'
                   '<activation><property><name>libnd4j-assembly</name></property></activation>'
                   '<dependencies><dependency><groupId>org.eclipse.deeplearning4j</groupId>'
                   '<artifactId>libnd4j</artifactId><type>zip</type><classifier>native</classifier>'
                   '</dependency></dependencies></profile></profiles></project>')
        pom = repo / 'assembly.pom'
        pom.write_text(profile)
        merge([repo], self.root / 'out', self.root / 'manifest.json',
              '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
        pom.write_text(profile.replace('<activation>', '<activation><activeByDefault>true</activeByDefault>'))
        with self.assertRaisesRegex(ValueError, 'requires build-only'):
            merge([repo], self.root / 'out2', self.root / 'manifest2.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)

    def test_missing_owner_fails(self):
        arm = self.worker('linux-arm64-cpu', b'arm')
        with self.assertRaisesRegex(ValueError, 'Missing canonical owner'):
            merge([arm], self.root / 'out', self.root / 'manifest.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)

    def test_identity_mismatch_fails(self):
        x64 = self.worker('linux-x86_64-cpu', b'x')
        with self.assertRaisesRegex(ValueError, 'source/version mismatch'):
            merge([x64], self.root / 'out', self.root / 'manifest.json',
                  '1.0.0-rewrite', 'b' * 40, canonical_worker_owners=True)

    def test_unowned_component_attachments_deterministic_main_bytes_checked(self):
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'x64')
        output = self.root / 'out'
        for repository, javadoc in ((arm, b'doc-arm'), (x64, b'doc-x64')):
            path = repository / 'org/eclipse/deeplearning4j/nd4j-presets-common/1.0.0-rewrite/nd4j-presets-common-1.0.0-rewrite-javadoc.jar'
            path.parent.mkdir(parents=True)
            path.write_bytes(javadoc)
        for repository in (arm, x64):
            pom = repository / 'org/eclipse/deeplearning4j/nd4j-presets-common/1.0.0-rewrite/nd4j-presets-common-1.0.0-rewrite.pom'
            pom.write_bytes(b'<project/>')
        result = merge([arm, x64], output, self.root / 'manifest.json',
                       '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
        self.assertEqual(5, len(result['files']))
        javadoc = next(output.rglob('*-javadoc.jar'))
        self.assertEqual(b'doc-x64', javadoc.read_bytes())
        pom = next(output.rglob('*.pom'))
        self.assertEqual(b'<project/>', pom.read_bytes())
        doc_rows = [row for row in result['files'] if row['path'].endswith('-javadoc.jar')]
        self.assertEqual(['linux-x86_64-cpu'], doc_rows[0]['shards'])
        pom_rows = [row for row in result['files'] if row['path'].endswith('.pom')]
        self.assertEqual(['linux-arm64-cpu', 'linux-x86_64-cpu'], pom_rows[0]['shards'])

    def test_unowned_component_main_conflicts_fail(self):
        import zipfile
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'x64')
        for repository, data in ((arm, b'pom-one'), (x64, b'pom-two')):
            path = repository / 'org/eclipse/deeplearning4j/nd4j-presets-common/1.0.0-rewrite/nd4j-presets-common-1.0.0-rewrite.pom'
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Conflicting component main artifact'):
            merge([arm, x64], self.root / 'out2', self.root / 'manifest2.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)

    def test_reproducible_jar_timestamp_noise_is_ignored(self):
        import io
        import zipfile
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'x64')
        entries = {'org/nd4j/presets/OpExclusion.class': b'payload'}
        for repository, timestamp in ((arm, (2026, 9, 13, 1, 0, 0)), (x64, (2026, 9, 13, 3, 0, 0))):
            folder = repository / 'org/eclipse/deeplearning4j/nd4j-presets-common/1.0.0-rewrite'
            folder.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                for name, payload in entries.items():
                    info = zipfile.ZipInfo(name, date_time=timestamp)
                    archive.writestr(info, payload)
            (folder / 'nd4j-presets-common-1.0.0-rewrite.jar').write_bytes(buffer.getvalue())
        result = merge([arm, x64], self.root / 'out3', self.root / 'manifest3.json',
                       '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
        self.assertTrue(any(row['path'].endswith('nd4j-presets-common-1.0.0-rewrite.jar')
                            for row in result['files']))

    def test_real_jar_content_conflicts_still_fail(self):
        import io
        import zipfile
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'x64')
        for repository, payload in ((arm, b'version-one'), (x64, b'version-two')):
            folder = repository / 'org/eclipse/deeplearning4j/nd4j-presets-common/1.0.0-rewrite'
            folder.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
                info = zipfile.ZipInfo('org/nd4j/presets/OpExclusion.class')
                archive.writestr(info, payload)
            (folder / 'nd4j-presets-common-1.0.0-rewrite.jar').write_bytes(buffer.getvalue())
        with self.assertRaisesRegex(ValueError, 'Conflicting component main artifact'):
            merge([arm, x64], self.root / 'out4', self.root / 'manifest4.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)

    def test_unknown_component_conflicts_still_fail(self):
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'x64')
        for repository, data in ((arm, b'one'), (x64, b'two')):
            path = repository / 'org/eclipse/deeplearning4j/unknown/1.0.0-rewrite/unknown-1.0.0-rewrite.pom'
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'Conflicting component main artifact'):
            merge([arm, x64], self.root / 'out', self.root / 'manifest.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
