"""Audited Javadoc-only repair overlay for the pinned native release source.

Only the thirty-four reviewed Javadoc lines are eligible. No whole fix checkout is
merged: unrelated changes at that revision cannot enter the Java reactor.
Extending this table requires reviewing the original comment and source SHA.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

SOURCE_COMMIT = "5debc0e4ed3588748b8491c94c072df75a149834"
API = "nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/"
REPAIRS = {
    API + "autodiff/samediff/SameDiff.java": {2292: "Example Usage:"},
    API + "linalg/api/ndarray/SparseNDArray.java": {
        115: "CSR ({@link SparseFormat#CSR})", 121: "CSC ({@link SparseFormat#CSC})", 209: "BSR layout"},
    API + "linalg/api/ndarray/SparseSolvers.java": {212: "Algorithm", 385: "Steps"},
}
# Each entry is an exact original/replacement pair; preserve every other byte.
REPAIRS = {name: {number: ("     * <h3>" + heading + "</h3>\n",
                          "     * <h4>" + heading + "</h4>\n")
                  for number, heading in repairs.items()}
           for name, repairs in REPAIRS.items()}
REPAIRS["nd4j/nd4j-tensorflow/src/main/java/org/nd4j/tensorflow/conversion/TensorflowConversion.java"] = {
    356: ("     * @throws IOException\n",
          "     * @throws IllegalStateException if TensorFlow cannot import the graph\n"),
}

REPAIRS["nd4j/nd4j-tensorflow-lite/src/main/java/org/nd4j/tensorflowlite/runner/TensorFlowLiteRunner.java"] = {
    96: ("     * Execute the {@link #session}\n",
         "     * Execute the {@link Interpreter}\n"),
}

# Class documentation uses H2 after Javadoc's implicit H1. Relative HTML links
# retain the real cross-package targets without changing imports or source lines.
VLM = "nd4j/samediff-vlm/src/main/java/org/eclipse/deeplearning4j/vlm/"
REPAIRS.update({
    VLM + "eval/metrics/AnlsMetric.java": {
        32: (" * If NLS < threshold (default 0.5), the score is 0; otherwise it is the NLS value.\n",
             " * If NLS &lt; threshold (default 0.5), the score is 0; otherwise it is the NLS value.\n")},
    VLM + "model/encoder/VisionEncoderIOConfig.java": {
        41: (" * <h3>Usage:</h3>\n", " * <h2>Usage:</h2>\n")},
    VLM + "model/projector/TemporalPatchEmbed.java": {
        64: (" * @see VideoPreprocessor\n",
             ' * @see <a href="../../preprocessing/VideoPreprocessor.html">VideoPreprocessor</a>\n')},
    VLM + "model/projector/ThreeDResampler.java": {
        68: (" * @see VideoVisionLanguageModel\n",
             ' * @see <a href="../VideoVisionLanguageModel.html">VideoVisionLanguageModel</a>\n')},
})


# The canonical config is in a separate module, not on pipeline-core's
# dependency/source path. Name its actual package and artifact without claiming
# that module-local Javadoc can resolve it as a symbol or relative HTML page.
PIPELINE_CONFIG = "nd4j/samediff-pipeline-core/src/main/java/org/eclipse/deeplearning4j/pipeline/PreprocessorConfig.java"
REPAIRS[PIPELINE_CONFIG] = {
    37: (" * @deprecated Use {@link org.eclipse.deeplearning4j.llm.config.PreprocessorConfig} instead.\n",
         " * @deprecated Use {@code PreprocessorConfig} in package {@code org.eclipse.deeplearning4j.llm.config} from the {@code samediff-llm} module instead.\n"),
}


# Lombok @Data on TtsFineTuneConfig generates the boolean getter used by the
# pipeline constructor; the embedded space made Javadoc resolve #isFreeze.
TTS_PIPELINE = "nd4j/samediff-audio/src/main/java/org/eclipse/deeplearning4j/audio/training/TtsTrainingPipeline.java"
REPAIRS[TTS_PIPELINE] = {
    55: (" *       {@link TtsFineTuneConfig#isFreeze TextEncoder()}.</li>\n",
         " *       {@link TtsFineTuneConfig#isFreezeTextEncoder()}.</li>\n"),
}

# Recovery 34381061422 reported 14 DataVec errors on 13 distinct lines.
# Configuration's misspelled opening tag caused two diagnostics on one line.
DATAVEC = "datavec/datavec-api/src/main/java/org/datavec/api/"
DATAVEC_RECOVERY_RUN = "34381061422"
DATAVEC_ERROR_COUNT = 14
for relative, number in (
        ("records/reader/RecordReader.java", 106),
        ("records/reader/impl/inmemory/InMemoryRecordReader.java", 108),
        ("records/reader/impl/inmemory/InMemorySequenceRecordReader.java", 189),
        ("records/reader/impl/transform/TransformProcessRecordReader.java", 149),
        ("records/reader/impl/transform/TransformProcessSequenceRecordReader.java", 199)):
    REPAIRS[DATAVEC + relative] = {
        number: ("     * @return\n", "     * @see #resetSupported()\n")}
REPAIRS.update({
    DATAVEC + "io/BinaryComparable.java": {
        37: ("     * @see org.apache.hadoop.io.WritableComparator#compareBytes(byte[],int,int,byte[],int,int)\n",
             "     * @see WritableComparator#compareBytes(byte[],int,int,byte[],int,int)\n"),
        66: ("     * @see org.apache.hadoop.io.WritableComparator#hashBytes(byte[],int)\n",
             "     * @see WritableComparator#hashBytes(byte[],int)\n")},
    DATAVEC + "conf/Configuration.java": {
        559: ("     * Get the value of the <code>name</code> property as a <ocde>Pattern</code>.\n",
              "     * Get the value of the <code>name</code> property as a <code>Pattern</code>.\n")},
    DATAVEC + "records/reader/impl/misc/SVMLightRecordReader.java": {
        99: ("     * @throws IOException\n",
             "     * @throws UnsupportedOperationException if the number of features is not configured,\n"),
        100: ("     * @throws InterruptedException\n",
              "     *         or if multilabel mode is enabled without a configured number of labels\n")},
    DATAVEC + "split/NumberedFileInputSplit.java": {
        46: ("     *                        @see {NumberedFileInputSplitTest}\n",
             '     * @see "NumberedFileInputSplitTest"\n')},
    DATAVEC + "transform/reduce/Reducer.java": {
        485: ("         * @param outputName Name of the column, after the reduction has been executed\n",
              "         * @param outputNames Names of the output columns, one for each reduction\n"),
        505: ("         * @param reductions  Reductions to execute\n",
              "         * @param reduction  Reduction to execute\n")},
})


# Recovery 34383274542: getURLString only concatenates strings; getURL performs
# URL construction and declares MalformedURLException. Keep its valid throws tag.
RESOURCES = "resources/src/main/java/org/deeplearning4j/common/resources/DL4JResources.java"
RESOURCES_RECOVERY_RUN = "34383274542"
REPAIRS[RESOURCES] = {
    96: ("     * @throws MalformedURLException For bad URL\n",
         "     * @see #getURL(String)\n"),
}


# Recovery 34396560573: @link is an inline tag, not a block tag; use one
# balanced inline link to the existing constant in the same package.
PYTHON_EXECUTIONER = "python4j/python4j-core/src/main/java/org/nd4j/python4j/PythonExecutioner.java"
PYTHON_RECOVERY_RUN = "34396560573"
REPAIRS[PYTHON_EXECUTIONER] = {
    46: (" * @link {{@link PythonConstants#DEFAULT_PYTHON_PATH_PROPERTY}} : The default python path to be used by the executioner.\n",
         " * {@link PythonConstants#DEFAULT_PYTHON_PATH_PROPERTY} : The default python path to be used by the executioner.\n"),
}


# Recovery 34398348119: invalid opening/closing it tags produce two diagnostics
# on one line. Preserve the intended italics using the standard HTML i element.
DATAVEC_LOCAL = "datavec/datavec-local/src/main/java/org/datavec/local/transforms/LocalTransformExecutor.java"
DATAVEC_LOCAL_RECOVERY_RUN = "34398348119"
REPAIRS[DATAVEC_LOCAL] = {
    101: ("     * but returns <it>sequence</it>\n",
          "     * but returns <i>sequence</i>\n"),
}


# Recovery 34400417264: four constructor comments contain bare ampersands.
LFW_ITERATOR = "deeplearning4j/deeplearning4j-data/deeplearning4j-datasets/src/main/java/org/deeplearning4j/datasets/iterator/impl/LFWDataSetIterator.java"
LFW_RECOVERY_RUN = "34400417264"
REPAIRS[LFW_ITERATOR] = {
    60: ("    /** Loads images with given  batchSize, numExamples, imgDim, train, & splitTrainTest returned by the generator. */\n",
         "    /** Loads images with given  batchSize, numExamples, imgDim, train, &amp; splitTrainTest returned by the generator. */\n"),
    66: ("    /** Loads images with given  batchSize, numExamples, numLabels, train, & splitTrainTest returned by the generator. */\n",
         "    /** Loads images with given  batchSize, numExamples, numLabels, train, &amp; splitTrainTest returned by the generator. */\n"),
    72: ("    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, useSubset, train, splitTrainTest & Random returned by the generator. */\n",
         "    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, useSubset, train, splitTrainTest &amp; Random returned by the generator. */\n"),
    79: ("    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, useSubset, train, splitTrainTest & Random returned by the generator. */\n",
         "    /** Loads images with given  batchSize, numExamples, imgDim, numLabels, useSubset, train, splitTrainTest &amp; Random returned by the generator. */\n"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def repaired(original, repairs):
    lines = original.splitlines(keepends=True)
    for number, (before, after) in repairs.items():
        expected = before.encode()
        if number > len(lines) or lines[number - 1] != expected:
            raise ValueError(f"audited Javadoc line {number} does not match")
        lines[number - 1] = after.encode()
    return b"".join(lines)


def prepare(source, fix_source, fix_commit, commit, output):
    if commit != SOURCE_COMMIT or not re.fullmatch(r"[0-9a-f]{40}", fix_commit):
        raise ValueError("documentation repair requires the audited source and immutable fix SHA")
    for root, sha in ((source, commit), (fix_source, fix_commit)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if actual != sha:
            raise ValueError("documentation checkout revision mismatch")
    pending, evidence = [], []
    for name, repairs in REPAIRS.items():
        original = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=source)
        fixed = subprocess.check_output(["git", "show", f"{fix_commit}:{name}"], cwd=fix_source)
        path = source / name
        if path.is_symlink() or path.read_bytes() != original:
            raise ValueError(f"documentation source is not pristine: {name}")
        expected = repaired(original, repairs)
        if fixed != expected:
            raise ValueError(f"fix contains unaudited or compiled-code changes: {name}")
        pending.append((path, fixed))
        evidence.append({"path": name, "sourceSha256": digest(original), "repairedSha256": digest(fixed),
                         "lines": sorted(repairs), "changes": [
                             {"line": number, "before": before, "after": after}
                             for number, (before, after) in sorted(repairs.items())],
                         "change": "Audited Javadoc lines only; all other bytes unchanged"})
    provenance = {"schemaVersion": 1, "sourceCommit": commit, "documentationFixCommit": fix_commit,
                  "policy": "audited-javadoc-only-v2", "files": evidence,
                  "datavecDiagnostics": {"recoveryRunId": DATAVEC_RECOVERY_RUN,
                                         "errorCount": DATAVEC_ERROR_COUNT,
                                         "repairedLineCount": 13},
                  "resourcesDiagnostics": {"recoveryRunId": RESOURCES_RECOVERY_RUN,
                                           "errorCount": 1, "repairedLineCount": 1},
                  "pythonDiagnostics": {"recoveryRunId": PYTHON_RECOVERY_RUN,
                                        "errorCount": 1, "repairedLineCount": 1},
                  "datavecLocalDiagnostics": {"recoveryRunId": DATAVEC_LOCAL_RECOVERY_RUN,
                                              "errorCount": 2, "repairedLineCount": 1},
                  "lfwDiagnostics": {"recoveryRunId": LFW_RECOVERY_RUN,
                                     "errorCount": 4, "repairedLineCount": 4}}
    # Validate every file before writing any overlay. Preserve line numbers and
    # every non-repaired byte, including all compiled code and source positions.
    for path, fixed in pending:
        path.write_bytes(fixed)
    (output / "documentation-recovery-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
