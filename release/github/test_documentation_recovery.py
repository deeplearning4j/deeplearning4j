"""Remote-only contracts for fail-closed, Javadoc-only release recovery."""
import json
import os
from pathlib import Path
import tempfile
import subprocess
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
        self.assertEqual(122, sum(len(row['lines']) for row in result['files']))
        self.assertEqual({'recoveryRunId': '34407671220', 'jobId': '102654524521',
                          'errorCount': 36, 'repairedLineCount': 32}, result['nnDiagnostics'])
        self.assertEqual({'recoveryRunId': '34381061422', 'errorCount': 14,
                          'repairedLineCount': 13}, result['datavecDiagnostics'])
        self.assertEqual({'recoveryRunId': '34383274542', 'errorCount': 1,
                          'repairedLineCount': 1}, result['resourcesDiagnostics'])
        self.assertEqual({'recoveryRunId': '34396560573', 'errorCount': 1,
                          'repairedLineCount': 1}, result['pythonDiagnostics'])
        self.assertEqual({'recoveryRunId': '34398348119', 'errorCount': 2,
                          'repairedLineCount': 1}, result['datavecLocalDiagnostics'])
        self.assertEqual({'recoveryRunId': '34400417264', 'errorCount': 4,
                          'repairedLineCount': 4}, result['lfwDiagnostics'])
        self.assertEqual({'recoveryRunId': '34404620379', 'errorCount': 6,
                          'repairedLineCount': 48}, result['utilityIteratorsDiagnostics'])
        self.assertEqual({'recoveryRunId': '34409908516', 'jobId': '102661762351',
                          'errorCount': 5, 'repairedLineCount': 8}, result['kerasDiagnostics'])
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
        self.assertEqual(118, sum(len(lines) for name, lines in docs.REPAIRS.items()
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

    def test_resources_repair_preserves_url_validation_and_is_required(self):
        root = Path(__file__).resolve().parents[2]
        name = docs.RESOURCES
        self.assertEqual({96: ('     * @throws MalformedURLException For bad URL\n',
                              '     * @see #getURL(String)\n')}, docs.REPAIRS[name])
        text = (root / name).read_text()
        self.assertEqual(docs.REPAIRS[name][96][1], text.splitlines(keepends=True)[95])
        self.assertEqual('     * @throws MalformedURLException For bad URL\n',
                         text.splitlines(keepends=True)[82])
        self.assertIn('public static URL getURL(String relativeToBase) throws MalformedURLException {', text)
        self.assertIn('return new URL(getURLString(relativeToBase));', text)
        method = text.split('public static String getURLString(String relativeToBase) {', 1)[1].split('\n    }', 1)[0]
        self.assertIn('return baseURL + relativeToBase;', method)
        self.assertNotIn('throw ', method)
        self.assertNotIn('new URL(', method)
        self.fixed[name] = self.originals[name]
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.prepare()
        for path in docs.REPAIRS:
            self.assertEqual(self.originals[path], (self.source / path).read_bytes())

    def test_python_inline_link_is_exact_resolvable_and_required(self):
        root = Path(__file__).resolve().parents[2]
        name = docs.PYTHON_EXECUTIONER
        self.assertEqual({46: (
            ' * @link {{@link PythonConstants#DEFAULT_PYTHON_PATH_PROPERTY}} : The default python path to be used by the executioner.\n',
            ' * {@link PythonConstants#DEFAULT_PYTHON_PATH_PROPERTY} : The default python path to be used by the executioner.\n')},
            docs.REPAIRS[name])
        self.assertEqual(121, sum(len(lines) for path, lines in docs.REPAIRS.items() if path != name))
        text = (root / name).read_text()
        self.assertEqual(docs.REPAIRS[name][46][1], text.splitlines(keepends=True)[45])
        constant = (root / name).with_name('PythonConstants.java').read_text()
        self.assertIn('package org.nd4j.python4j;', constant)
        self.assertIn('public final static String DEFAULT_PYTHON_PATH_PROPERTY = "org.eclipse.python4j.path";', constant)
        self.fixed[name] = self.originals[name]
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.prepare()
        for path in docs.REPAIRS:
            self.assertEqual(self.originals[path], (self.source / path).read_bytes())

    def test_datavec_local_italics_are_exact_and_required(self):
        root = Path(__file__).resolve().parents[2]
        name = docs.DATAVEC_LOCAL
        self.assertEqual({101: ('     * but returns <it>sequence</it>\n',
                               '     * but returns <i>sequence</i>\n')}, docs.REPAIRS[name])
        self.assertEqual(121, sum(len(lines) for path, lines in docs.REPAIRS.items() if path != name))
        text = (root / name).read_text()
        self.assertEqual(docs.REPAIRS[name][101][1], text.splitlines(keepends=True)[100])
        self.fixed[name] = self.originals[name]
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.prepare()
        for path in docs.REPAIRS:
            self.assertEqual(self.originals[path], (self.source / path).read_bytes())

    def test_each_lfw_entity_repair_is_exact_and_required(self):
        root = Path(__file__).resolve().parents[2]
        name = docs.LFW_ITERATOR
        self.assertEqual({60, 66, 72, 79}, set(docs.REPAIRS[name]))
        self.assertEqual(118, sum(len(lines) for path, lines in docs.REPAIRS.items() if path != name))
        text = (root / name).read_bytes().splitlines(keepends=True)
        for number, (before, after) in docs.REPAIRS[name].items():
            with self.subTest(number=number):
                self.assertTrue(before.startswith('    /** Loads images with given  '))
                self.assertTrue(before.endswith(' returned by the generator. */\n'))
                self.assertEqual(1, before.count(' & '))
                self.assertEqual(before.replace(' & ', ' &amp; '), after)
                self.assertEqual(after.encode(), text[number - 1])
                fixed = self.fixed[name]
                lines = fixed.splitlines(keepends=True)
                lines[number - 1] = before.encode()
                self.fixed[name] = b''.join(lines)
                with self.assertRaisesRegex(ValueError, 'unaudited'):
                    self.prepare()
                for path in docs.REPAIRS:
                    self.assertEqual(self.originals[path], (self.source / path).read_bytes())
                self.fixed[name] = fixed

    def test_utility_iterator_repairs_match_noop_overrides_and_are_required(self):
        root = Path(__file__).resolve().parents[2]
        names = [name for name in docs.REPAIRS if name.startswith(docs.UTILITY_ITERATORS)]
        self.assertEqual(6, len(names))
        self.assertEqual(74, sum(len(lines) for name, lines in docs.REPAIRS.items()
                                if name not in names))
        for name in names:
            text = (root / name).read_text()
            self.assertNotIn('@implSpec', text)
            body = text.split('public void remove() {', 1)[1].split('}', 1)[0]
            self.assertTrue(all(not line.strip() or line.strip().startswith('//')
                                for line in body.splitlines()))
            self.assertEqual(8, len(docs.REPAIRS[name]))
            for number, (before, after) in docs.REPAIRS[name].items():
                with self.subTest(name=name, number=number):
                    self.assertEqual(after, text.splitlines(keepends=True)[number - 1])
                    fixed = self.fixed[name]
                    lines = fixed.splitlines(keepends=True)
                    lines[number - 1] = before.encode()
                    self.fixed[name] = b''.join(lines)
                    with self.assertRaisesRegex(ValueError, 'unaudited'):
                        self.prepare()
                    for path in docs.REPAIRS:
                        self.assertEqual(self.originals[path], (self.source / path).read_bytes())
                    self.fixed[name] = fixed

    def test_nn_repairs_preserve_previous_82_lines_and_each_is_required(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(17, len(docs.NN_REPAIRS))
        self.assertEqual(32, sum(len(lines) for lines in docs.NN_REPAIRS.values()))
        self.assertEqual(82, sum(len(lines) for name, lines in docs.REPAIRS.items()
                                if not name.startswith((docs.NN, docs.KERAS))))
        for relative, repairs in docs.NN_REPAIRS.items():
            name = docs.NN + relative
            text = (root / name).read_bytes().splitlines(keepends=True)
            for number, (before, after) in repairs.items():
                with self.subTest(name=name, number=number):
                    self.assertEqual(after.encode(), text[number - 1])
                    self.assertTrue(before.lstrip().startswith('*'))
                    self.assertTrue(after.lstrip().startswith('*'))
                    fixed = self.fixed[name]
                    lines = fixed.splitlines(keepends=True)
                    lines[number - 1] = before.encode()
                    self.fixed[name] = b''.join(lines)
                    with self.assertRaisesRegex(ValueError, 'unaudited'):
                        self.prepare()
                    for path in docs.REPAIRS:
                        self.assertEqual(self.originals[path], (self.source / path).read_bytes())
                    self.fixed[name] = fixed

    def test_nn_lombok_links_preserve_real_accessors_and_source_targets(self):
        root = Path(__file__).resolve().parents[2]
        for config in ('MultiLayerConfiguration', 'ComputationGraphConfiguration'):
            text = (root / (docs.NN + 'nn/conf/' + config + '.java')).read_text()
            self.assertIn('@Data', text)
            self.assertIn('protected int iterationCount = 0;', text)
            self.assertNotIn('void setIterationCount(', text)
            self.assertIn('protected int epochCount = 0;', text)
            if config == 'MultiLayerConfiguration':
                self.assertIn('public void setEpochCount(int epochCount)', text)
            else:
                self.assertNotIn('void setEpochCount(', text)
        graph = (root / (docs.NN + 'nn/graph/ComputationGraph.java')).read_text()
        self.assertIn('@Getter\n    private int numOutputArrays;', graph)
        self.assertIn('this.numOutputArrays = configuration.getNetworkOutputs().size();', graph)
        self.assertNotIn('int getNumOutputArrays(', graph)
        config = (root / (docs.NN + 'nn/conf/ComputationGraphConfiguration.java')).read_text()
        self.assertIn('protected List<String> networkOutputs;', config)
        for relative in ('nn/multilayer/MultiLayerNetwork.java', 'util/NetworkUtils.java'):
            text = (root / (docs.NN + relative)).read_text()
            self.assertEqual(2, text.count('{@link MultiLayerConfiguration#setEpochCount(int)}'))
        for relative, repairs in docs.NN_REPAIRS.items():
            for before, after in repairs.values():
                if '#setIterationCount(int)' in before:
                    self.assertIn('Lombok-generated {@code setIterationCount(int)} setter for {@link ', after)
                    self.assertIn('#iterationCount}', after)
                if '#setEpochCount(int)' in before:
                    self.assertIn('{@code setEpochCount(int)}', after)
                    self.assertIn('{@link ComputationGraphConfiguration#epochCount}', after)
        # Neither the root nor this module supplies a delomboked Javadoc path.
        for pom in ('pom.xml', 'deeplearning4j/pom.xml', 'deeplearning4j/deeplearning4j-nn/pom.xml'):
            self.assertNotIn('delombok', (root / pom).read_text())

    def test_nn_corrected_links_match_declared_signatures(self):
        root = Path(__file__).resolve().parents[2]
        def source(relative):
            return (root / (docs.NN + relative)).read_text()
        self.assertIn('init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams)',
                      source('nn/api/ParamInitializer.java'))
        self.assertIn('public T dilation(long... dilation)', source('nn/conf/layers/ConvolutionLayer.java'))
        self.assertIn('public Builder dataFormat(CNN2DFormat dataFormat)', source('nn/conf/layers/LocalResponseNormalization.java'))
        self.assertIn('int batchSize, LayerWorkspaceMgr workspaceMgr)', source('optimize/api/ConvexOptimizer.java'))
        for shape in ('int[]', 'long[]'):
            self.assertIn('initWeights(double fanIn, double fanOut, ' + shape + ' shape, WeightInit initScheme,',
                          source('nn/weights/WeightInitUtil.java'))
        for type_name in ('Activation', 'IActivation'):
            self.assertIn('public Builder activation(' + type_name + ' ', source('nn/conf/layers/ActivationLayer.java'))
        self.assertIn('Strict, Truncate, Same, Causal;', source('nn/conf/ConvolutionMode.java'))
        self.assertIn('convolutionMode = ConvolutionMode.Truncate;', source('nn/conf/layers/ConvolutionLayer.java'))
        gcn = root / (docs.NN + 'nn/conf/layers/GcnLayer.java')
        self.assertIn('public class MultiLayerConfiguration', (gcn.parent / '../MultiLayerConfiguration.java').read_text())

    @unittest.skipUnless(os.environ.get('RELEASE_DELOMBOK_CONTRACT') == '1', 'remote Java 21 doclet only')
    def test_nn_generated_accessor_comments_in_source_doclet(self):
        # A source doclet has fields but not Lombok-generated methods. Exercise
        # the real before/after comment lines without adding synthetic setters.
        sources = self.root / 'doclet-source'
        sources.mkdir()
        for config in ('MultiLayerConfiguration', 'ComputationGraphConfiguration'):
            (sources / (config + '.java')).write_text(
                'import java.util.List;\n/** Configuration source view. */\npublic class ' + config + ' {\n'
                'protected int iterationCount; protected int epochCount;\n'
                'protected List<String> networkOutputs;\n'
                + ('public void setEpochCount(int epochCount) {}\n'
                   if config == 'MultiLayerConfiguration' else '') + '}\n')
        comments = [(before, after) for repairs in docs.NN_REPAIRS.values()
                    for before, after in repairs.values()
                    if '#setIterationCount(int)' in before or '#setEpochCount(int)' in before
                    or '#getNumOutputArrays()' in before]
        self.assertEqual(13, len(comments))
        reader = sources / 'Reader.java'
        for index, label in ((0, 'original'), (1, 'repaired')):
            reader.write_text('/** Reader source view. */\npublic class Reader {\n' + ''.join(
                '/**\n' + pair[index] + ' */\npublic void method' + str(i) + '() {}\n'
                for i, pair in enumerate(comments)) + '}\n')
            result = subprocess.run(['javadoc', '-quiet', '-Xdoclint:all', '-d',
                                     str(self.root / label), *map(str, sorted(sources.glob('*.java')))],
                                    capture_output=True, text=True)
            print(label + ' source-doclet diagnostics:\n' + result.stdout + result.stderr)
            if index == 0:
                self.assertNotEqual(0, result.returncode)
                self.assertEqual(13, result.stderr.count('error: reference not found'))
            else:
                self.assertEqual(0, result.returncode, result.stderr)
                html = (self.root / label / 'Reader.html').read_text()
                self.assertIn('MultiLayerConfiguration.html#iterationCount', html)
                self.assertIn('ComputationGraphConfiguration.html#epochCount', html)
                self.assertIn('ComputationGraphConfiguration.html#networkOutputs', html)
                self.assertIn('setIterationCount(int)', html)
                self.assertIn('getNumOutputArrays()', html)

    def test_keras_comments_match_literal_format_dispatch_and_optional_backend(self):
        root = Path(__file__).resolve().parents[2]
        conv = (root / docs.KERAS_CONV).read_text()
        for method, result in (('getCNN3DDataFormatFromConfig', 'Convolution3D.DataFormat'),
                               ('getDataFormatFromConfig', 'CNN2DFormat')):
            body = conv.split('public static ' + result + ' ' + method, 1)[1].split('\n    }', 1)[0]
            self.assertIn('dataFormat.equals("channels_last")', body)
            self.assertNotIn('getDIM_ORDERING_', body)
        self.assertIn('Convolution3D.DataFormat.NDHWC : Convolution3D.DataFormat.NCDHW', conv)
        config = (root / (docs.KERAS + 'config/KerasLayerConfiguration.java')).read_text()
        self.assertIn('@Data', config)
        self.assertNotIn('String getDIM_ORDERING_', config)
        model = (root / docs.KERAS_MODEL).read_text()
        body = model.split('public static String determineKerasBackend(', 1)[1].split('\n    }', 1)[0]
        self.assertIn('String kerasBackend = null;', body)
        self.assertIn('return kerasBackend;', body)
        self.assertNotIn('throw', body)
        for name in (docs.KERAS_CONV, docs.KERAS_MODEL):
            text = (root / name).read_bytes().splitlines(keepends=True)
            for number, (before, after) in docs.REPAIRS[name].items():
                with self.subTest(name=name, number=number):
                    self.assertEqual(after.encode(), text[number - 1])
                    fixed = self.fixed[name]
                    lines = fixed.splitlines(keepends=True)
                    lines[number - 1] = before.encode()
                    self.fixed[name] = b''.join(lines)
                    with self.assertRaisesRegex(ValueError, 'unaudited'):
                        self.prepare()
                    for path in docs.REPAIRS:
                        self.assertEqual(self.originals[path], (self.source / path).read_bytes())
                    self.fixed[name] = fixed

    def test_real_pinned_source_matches_only_audited_edits(self):
        original_root = os.environ.get('DOCUMENTATION_CONTRACT_SOURCE')
        if not original_root:
            self.skipTest('remote immutable source checkout required')
        current = Path(__file__).resolve().parents[2]
        for name, repairs in docs.REPAIRS.items():
            original = (Path(original_root) / name).read_bytes()
            self.assertEqual(docs.repaired(original, repairs), (current / name).read_bytes(), name)
