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

    def test_unknown_component_conflicts_still_fail(self):
        arm = self.worker('linux-arm64-cpu', b'arm')
        x64 = self.worker('linux-x86_64-cpu', b'x64')
        for repository, data in ((arm, b'one'), (x64, b'two')):
            path = repository / 'org/eclipse/deeplearning4j/unknown/1.0.0-rewrite/unknown-1.0.0-rewrite.pom'
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            merge([arm, x64], self.root / 'out', self.root / 'manifest.json',
                  '1.0.0-rewrite', 'a' * 40, canonical_worker_owners=True)
