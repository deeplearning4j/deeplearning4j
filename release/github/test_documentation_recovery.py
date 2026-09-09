"""Remote-only contracts for fail-closed, Javadoc-only release recovery."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from release.github import documentation_recovery as docs


class DocumentationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.fix, self.output = [self.root / name for name in ('source', 'fix', 'output')]
        self.output.mkdir()
        self.fix.mkdir()
        self.originals, self.fixed = {}, {}
        for name, repairs in docs.REPAIRS.items():
            lines = [b'\n'] * (max(repairs) + 1)
            for number, (before, after) in repairs.items():
                lines[number - 1] = before.encode()
            original = b''.join(lines)
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original)
            self.originals[name] = original
            self.fixed[name] = docs.repaired(original, repairs)

    def git(self, command, cwd, **kwargs):
        if command[1] == 'rev-parse':
            return docs.SOURCE_COMMIT if cwd == self.source else 'b' * 40
        name = command[2].split(':', 1)[1]
        return (self.originals if cwd == self.source else self.fixed)[name]

    def prepare(self):
        with patch.object(docs.subprocess, 'check_output', side_effect=self.git):
            return docs.prepare(self.source, self.fix, 'b' * 40, docs.SOURCE_COMMIT, self.output)

    def test_records_both_revisions_and_all_file_hashes(self):
        result = self.prepare()
        self.assertEqual(docs.SOURCE_COMMIT, result['sourceCommit'])
        self.assertEqual('b' * 40, result['documentationFixCommit'])
        self.assertEqual(27, sum(len(row['lines']) for row in result['files']))
        self.assertEqual({'recoveryRunId': '34381061422', 'errorCount': 14,
                          'repairedLineCount': 13}, result['datavecDiagnostics'])
        self.assertEqual('audited-javadoc-only-v2', result['policy'])
        for row in result['files']:
            self.assertEqual(docs.digest(self.originals[row['path']]), row['sourceSha256'])
            self.assertEqual(docs.digest(self.fixed[row['path']]), row['repairedSha256'])
        for name in docs.REPAIRS:
            self.assertEqual(self.fixed[name], (self.source / name).read_bytes())
        self.assertEqual(result, json.loads((self.output / 'documentation-recovery-provenance.json').read_text()))

    def test_code_change_rejected_before_any_write(self):
        name = list(docs.REPAIRS)[-1]
        self.fixed[name] += b'class Injected {}\n'
        with self.assertRaisesRegex(ValueError, 'compiled-code'):
            self.prepare()
        for name in docs.REPAIRS:
            self.assertEqual(self.originals[name], (self.source / name).read_bytes())

    def test_extra_documentation_or_line_number_changes_rejected(self):
        for suffix in (b'/** extra */', b'\n', b'\\u000a'):
            with self.subTest(suffix=suffix):
                name = next(iter(docs.REPAIRS))
                original = self.fixed[name]
                self.fixed[name] += suffix
                with self.assertRaises(ValueError):
                    self.prepare()
                self.fixed[name] = original

    def test_dirty_source_rejected(self):
        (self.source / next(iter(docs.REPAIRS))).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'pristine'):
            self.prepare()

    def test_wrong_revision_and_unaudited_source_rejected(self):
        with patch.object(docs.subprocess, 'check_output', return_value='c' * 40):
            with self.assertRaisesRegex(ValueError, 'revision'):
                docs.prepare(self.source, self.fix, 'b' * 40, docs.SOURCE_COMMIT, self.output)
        for source, fix in (('a' * 40, 'b' * 40), (docs.SOURCE_COMMIT, 'main')):
            with self.assertRaises(ValueError):
                docs.prepare(self.source, self.fix, fix, source, self.output)

    def test_heading_context_must_match(self):
        with self.assertRaisesRegex(ValueError, 'audited Javadoc'):
            docs.repaired(b'wrong\n', {1: ('     * <h3>Example Usage:</h3>\n',
                                             '     * <h4>Example Usage:</h4>\n')})

    def test_tensorflow_lite_link_repair_is_exact_and_required(self):
        name = 'nd4j/nd4j-tensorflow-lite/src/main/java/org/nd4j/tensorflowlite/runner/TensorFlowLiteRunner.java'
        self.assertEqual({96: ('     * Execute the {@link #session}\n',
                              '     * Execute the {@link Interpreter}\n')}, docs.REPAIRS[name])
        self.fixed[name] = self.originals[name]
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.prepare()
        for path in docs.REPAIRS:
            self.assertEqual(self.originals[path], (self.source / path).read_bytes())

    def test_each_vlm_repair_is_required_before_any_write(self):
        names = [name for name in docs.REPAIRS if name.startswith(docs.VLM)]
        self.assertEqual(4, len(names))
        self.assertEqual(23, sum(len(lines) for name, lines in docs.REPAIRS.items()
                                if not name.startswith(docs.VLM)))
        for name in names:
            with self.subTest(name=name):
                fixed = self.fixed[name]
                self.fixed[name] = self.originals[name]
                with self.assertRaisesRegex(ValueError, 'unaudited'):
                    self.prepare()
                for path in docs.REPAIRS:
                    self.assertEqual(self.originals[path], (self.source / path).read_bytes())
                self.fixed[name] = fixed

    def test_vlm_links_resolve_to_actual_source_classes(self):
        root = Path(__file__).resolve().parents[2]
        for relative, line, target in (
                ('model/projector/TemporalPatchEmbed.java', 64,
                 '../../preprocessing/VideoPreprocessor'),
                ('model/projector/ThreeDResampler.java', 68, '../VideoVisionLanguageModel')):
            with self.subTest(relative=relative):
                name = docs.VLM + relative
                label = target.rsplit('/', 1)[-1]
                self.assertEqual(f' * @see <a href="{target}.html">{label}</a>\n',
                                 docs.REPAIRS[name][line][1])
                source = (root / name).parent / (target + '.java')
                self.assertIn(f'public class {label}', source.read_text())

    def test_pipeline_config_names_real_external_module_without_unresolvable_link(self):
        root = Path(__file__).resolve().parents[2]
        name = docs.PIPELINE_CONFIG
        self.assertEqual({37: (
            ' * @deprecated Use {@link org.eclipse.deeplearning4j.llm.config.PreprocessorConfig} instead.\n',
            ' * @deprecated Use {@code PreprocessorConfig} in package {@code org.eclipse.deeplearning4j.llm.config} from the {@code samediff-llm} module instead.\n')},
            docs.REPAIRS[name])
        target = root / 'nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/config/PreprocessorConfig.java'
        content = target.read_text()
        self.assertIn('package org.eclipse.deeplearning4j.llm.config;', content)
        self.assertIn('public class PreprocessorConfig', content)
        self.assertIn('public static PreprocessorConfig fromJson(String json)', content)
        self.assertIn('<artifactId>samediff-llm</artifactId>',
                      (root / 'nd4j/samediff-llm/pom.xml').read_text())
        self.assertNotIn('<artifactId>samediff-llm</artifactId>',
                         (root / 'nd4j/samediff-pipeline-core/pom.xml').read_text())
        self.fixed[name] = self.originals[name]
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.prepare()
        for path in docs.REPAIRS:
            self.assertEqual(self.originals[path], (self.source / path).read_bytes())

    def test_tts_link_names_actual_boolean_getter_and_is_required(self):
        root = Path(__file__).resolve().parents[2]
        name = docs.TTS_PIPELINE
        self.assertEqual({55: (
            ' *       {@link TtsFineTuneConfig#isFreeze TextEncoder()}.</li>\n',
            ' *       {@link TtsFineTuneConfig#isFreezeTextEncoder()}.</li>\n')},
            docs.REPAIRS[name])
        config = (root / (docs.API + 'autodiff/samediff/config/TtsFineTuneConfig.java')).read_text()
        self.assertIn('import lombok.Data;', config)
        self.assertIn('@Data', config)
        self.assertIn('private boolean freezeTextEncoder = true;', config)
        pipeline = (root / name).read_text()
        self.assertIn('import org.nd4j.autodiff.samediff.config.TtsFineTuneConfig;', pipeline)
        self.assertIn('if (ttsConfig.isFreezeTextEncoder())', pipeline)
        self.fixed[name] = self.originals[name]
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.prepare()
        for path in docs.REPAIRS:
            self.assertEqual(self.originals[path], (self.source / path).read_bytes())

    def test_each_datavec_line_is_required_before_any_write(self):
        names = [name for name in docs.REPAIRS if name.startswith(docs.DATAVEC)]
        self.assertEqual(10, len(names))
        self.assertEqual(13, sum(len(docs.REPAIRS[name]) for name in names))
        for name in names:
            for number, (before, after) in docs.REPAIRS[name].items():
                with self.subTest(name=name, number=number):
                    fixed = self.fixed[name]
                    lines = fixed.splitlines(keepends=True)
                    lines[number - 1] = before.encode()
                    self.fixed[name] = b''.join(lines)
                    with self.assertRaisesRegex(ValueError, 'unaudited'):
                        self.prepare()
                    for path in docs.REPAIRS:
                        self.assertEqual(self.originals[path], (self.source / path).read_bytes())
                    self.fixed[name] = fixed

    def test_datavec_links_and_tags_match_source_signatures(self):
        root = Path(__file__).resolve().parents[2]
        comparator = (root / (docs.DATAVEC + 'io/WritableComparator.java')).read_text()
        self.assertIn('package org.datavec.api.io;', comparator)
        self.assertIn('public static int compareBytes(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2)', comparator)
        self.assertIn('public static int hashBytes(byte[] bytes, int length)', comparator)
        for name, repairs in docs.REPAIRS.items():
            if not name.startswith(docs.DATAVEC):
                continue
            text = (root / name).read_text()
            for number, (before, after) in repairs.items():
                self.assertEqual(after, text.splitlines(keepends=True)[number - 1])
                if '@see #resetSupported()' in after:
                    self.assertIn('void reset()', text)
                    self.assertIn('boolean resetSupported()', text)
        svm = (root / (docs.DATAVEC + 'records/reader/impl/misc/SVMLightRecordReader.java')).read_text()
        self.assertIn('public void setConf(Configuration conf) {', svm)
        set_conf = svm.split('public void setConf(Configuration conf) {', 1)[1].split('\n    }', 1)[0]
        self.assertEqual(2, set_conf.count('throw new UnsupportedOperationException('))
        reducer = (root / (docs.DATAVEC + 'transform/reduce/Reducer.java')).read_text()
        self.assertIn('String column, List<String> outputNames, List<ReduceOp> reductions,', reducer)
        self.assertIn('String column, String outputName, ReduceOp reduction, Condition condition)', reducer)

    def test_real_pinned_source_matches_only_audited_edits(self):
        original_root = os.environ.get('DOCUMENTATION_CONTRACT_SOURCE')
        if not original_root:
            self.skipTest('remote immutable source checkout required')
        current = Path(__file__).resolve().parents[2]
        for name, repairs in docs.REPAIRS.items():
            original = (Path(original_root) / name).read_bytes()
            self.assertEqual(docs.repaired(original, repairs), (current / name).read_bytes(), name)
