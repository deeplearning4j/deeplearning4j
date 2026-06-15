/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.dl4jcore.datasets.datavec;

import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for two bugs in the DL4J data pipeline:
 *
 * Bug A: DataSetUtil.tailor3d2d() produced garbage output due to
 *        permute(0,2,1).reshape('f',...) creating a view with incompatible strides.
 *
 * Bug B: RecordReaderMultiDataSetIterator ALIGN_END used per-example
 *        longestSequence[i] instead of batch-wide maxTSLength for the offset,
 *        causing data to not be truly right-aligned.
 *
 * Both bugs caused NormalizerStandardize.fit() to produce NaN/Inf statistics
 * or throw "No data was added" on variable-length masked sequence data.
 */
@Slf4j
@DisplayName("Test tailor3d2d and ALIGN_END bugs")
@NativeTag
@Tag(TagNames.NDARRAY_ETL)
@Tag(TagNames.FILE_IO)
class TestTailor3d2dAndAlignEndBugs extends BaseDL4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @TempDir
    public Path temporaryFolder;

    // =========================================================================
    //  tailor3d2d tests
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("tailor3d2d: no mask returns all rows")
    void testTailor3d2dNoMask(Nd4jBackend backend) {
        // Shape [2, 3, 4]: 2 examples, 3 features, 4 time steps
        INDArray data3d = Nd4j.create(2, 3, 4);
        for (int ex = 0; ex < 2; ex++) {
            for (int t = 0; t < 4; t++) {
                for (int f = 0; f < 3; f++) {
                    data3d.putScalar(ex, f, t, (ex + 1) * 100.0 + (t + 1) * 10.0 + (f + 1));
                }
            }
        }

        INDArray result = DataSetUtil.tailor3d2d(data3d, null);

        assertNotNull(result);
        assertEquals(2, result.rank());
        // 2 examples * 4 timesteps = 8 rows
        assertEquals(8, result.rows());
        assertEquals(3, result.columns());

        // Verify every value is present and correct
        // Time-major ordering: (t0,ex0), (t0,ex1), (t1,ex0), (t1,ex1), ...
        int row = 0;
        for (int t = 0; t < 4; t++) {
            for (int ex = 0; ex < 2; ex++) {
                for (int f = 0; f < 3; f++) {
                    double expected = (ex + 1) * 100.0 + (t + 1) * 10.0 + (f + 1);
                    assertEquals(expected, result.getDouble(row, f), 1e-5,
                            "Mismatch at row=" + row + " col=" + f);
                }
                row++;
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("tailor3d2d: all-ones mask returns all rows")
    void testTailor3d2dAllOnesMask(Nd4jBackend backend) {
        INDArray data3d = Nd4j.create(2, 3, 4);
        for (int ex = 0; ex < 2; ex++) {
            for (int t = 0; t < 4; t++) {
                for (int f = 0; f < 3; f++) {
                    data3d.putScalar(ex, f, t, ex * 10.0 + t + f * 0.1);
                }
            }
        }

        INDArray mask = Nd4j.ones(2, 4);
        INDArray result = DataSetUtil.tailor3d2d(data3d, mask);

        assertNotNull(result);
        assertEquals(8, result.rows());
        assertEquals(3, result.columns());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("tailor3d2d: partial mask filters correctly")
    void testTailor3d2dPartialMask(Nd4jBackend backend) {
        // 2 examples, 3 features, 4 timesteps
        // ex0: all 4 timesteps valid, ex1: only timesteps 2,3 valid (ALIGN_END style)
        INDArray data3d = Nd4j.create(2, 3, 4);
        // ex0: values (t+1)*10 + (f+1)
        for (int t = 0; t < 4; t++) {
            for (int f = 0; f < 3; f++) {
                data3d.putScalar(0, f, t, (t + 1) * 10.0 + (f + 1));
            }
        }
        // ex1: values (t+1)*100 + (f+1) for t=2,3; zeros for t=0,1 (padding)
        for (int t = 2; t < 4; t++) {
            for (int f = 0; f < 3; f++) {
                data3d.putScalar(1, f, t, (t + 1) * 100.0 + (f + 1));
            }
        }

        INDArray mask = Nd4j.create(new double[][]{{1, 1, 1, 1}, {0, 0, 1, 1}});

        INDArray result = DataSetUtil.tailor3d2d(data3d, mask);
        assertNotNull(result);
        // 4 from ex0 + 2 from ex1 = 6 rows
        assertEquals(6, result.rows());
        assertEquals(3, result.columns());

        // Time-major with mask: t0(ex0), t1(ex0), t2(ex0,ex1), t3(ex0,ex1)
        double[][] expectedRows = {
                {11, 12, 13},       // t0/ex0
                {21, 22, 23},       // t1/ex0
                {31, 32, 33},       // t2/ex0
                {301, 302, 303},    // t2/ex1
                {41, 42, 43},       // t3/ex0
                {401, 402, 403}     // t3/ex1
        };

        for (int r = 0; r < expectedRows.length; r++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(expectedRows[r][c], result.getDouble(r, c), 1e-5,
                        "Mismatch at row=" + r + " col=" + c);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("tailor3d2d: all-zeros mask returns null")
    void testTailor3d2dAllZerosMask(Nd4jBackend backend) {
        INDArray data3d = Nd4j.create(2, 3, 4);
        INDArray mask = Nd4j.zeros(2, 4);

        INDArray result = DataSetUtil.tailor3d2d(data3d, mask);
        assertNull(result, "All-zeros mask should return null (no valid data)");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("tailor3d2d: f-order data produces same result as c-order")
    void testTailor3d2dFOrder(Nd4jBackend backend) {
        INDArray dataCOrder = Nd4j.create(2, 3, 4);
        INDArray dataFOrder = Nd4j.create(new int[]{2, 3, 4}, 'f');

        // Fill both with identical values
        for (int ex = 0; ex < 2; ex++) {
            for (int f = 0; f < 3; f++) {
                for (int t = 0; t < 4; t++) {
                    double val = ex * 100 + f * 10 + t;
                    dataCOrder.putScalar(ex, f, t, val);
                    dataFOrder.putScalar(ex, f, t, val);
                }
            }
        }

        INDArray mask = Nd4j.create(new double[][]{{1, 1, 1, 1}, {0, 0, 1, 1}});

        INDArray resultC = DataSetUtil.tailor3d2d(dataCOrder, mask);
        INDArray resultF = DataSetUtil.tailor3d2d(dataFOrder, mask);

        assertNotNull(resultC);
        assertNotNull(resultF);
        assertEquals(resultC, resultF, "c-order and f-order should produce identical tailored output");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("tailor3d2d: no NaN or Inf in result")
    void testTailor3d2dNoNanInf(Nd4jBackend backend) {
        Random rng = new Random(42);
        INDArray data3d = Nd4j.create(4, 5, 10);
        for (int ex = 0; ex < 4; ex++) {
            for (int f = 0; f < 5; f++) {
                for (int t = 0; t < 10; t++) {
                    data3d.putScalar(ex, f, t, rng.nextGaussian() * 10 + 5);
                }
            }
        }

        // Variable-length mask
        INDArray mask = Nd4j.ones(4, 10);
        mask.putScalar(0, 0, 0); // ex0 missing first timestep
        mask.putScalar(1, 0, 0);
        mask.putScalar(1, 1, 0); // ex1 missing first two timesteps
        mask.putScalar(2, 0, 0);
        mask.putScalar(2, 1, 0);
        mask.putScalar(2, 2, 0); // ex2 missing first three timesteps
        // ex3 all present

        INDArray result = DataSetUtil.tailor3d2d(data3d, mask);
        assertNotNull(result);

        // Check no NaN or Inf
        for (int r = 0; r < result.rows(); r++) {
            for (int c = 0; c < result.columns(); c++) {
                double val = result.getDouble(r, c);
                assertFalse(Double.isNaN(val), "NaN at row=" + r + " col=" + c);
                assertFalse(Double.isInfinite(val), "Inf at row=" + r + " col=" + c);
            }
        }

        // Row count: (10-1) + (10-2) + (10-3) + 10 = 9+8+7+10 = 34
        assertEquals(34, result.rows());
    }

    // =========================================================================
    //  ALIGN_END mask direction tests
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ALIGN_END: mask has padding at start, not end")
    void testAlignEndMaskDirection(Nd4jBackend backend) throws Exception {
        // Create variable-length sequences using CollectionSequenceRecordReader
        // Sequence 1: length 4 (features), length 2 (labels)
        // Sequence 2: length 2 (features), length 4 (labels)
        // In a batch, maxTSLength=4 for both features and labels

        List<List<Writable>> featSeq1 = new ArrayList<>();
        for (int t = 0; t < 4; t++) {
            featSeq1.add(Arrays.asList(
                    (Writable) new DoubleWritable(t + 1),
                    new DoubleWritable(t + 10),
                    new DoubleWritable(t + 100)));
        }
        List<List<Writable>> featSeq2 = new ArrayList<>();
        for (int t = 0; t < 2; t++) {
            featSeq2.add(Arrays.asList(
                    (Writable) new DoubleWritable(t + 201),
                    new DoubleWritable(t + 210),
                    new DoubleWritable(t + 300)));
        }

        List<List<Writable>> labelSeq1 = new ArrayList<>();
        for (int t = 0; t < 2; t++) {
            labelSeq1.add(Arrays.asList((Writable) new IntWritable(t % 2)));
        }
        List<List<Writable>> labelSeq2 = new ArrayList<>();
        for (int t = 0; t < 4; t++) {
            labelSeq2.add(Arrays.asList((Writable) new IntWritable(t % 2)));
        }

        SequenceRecordReader featReader = new CollectionSequenceRecordReader(
                Arrays.asList(featSeq1, featSeq2));
        SequenceRecordReader labelReader = new CollectionSequenceRecordReader(
                Arrays.asList(labelSeq1, labelSeq2));

        DataSetIterator iter = new SequenceRecordReaderDataSetIterator(
                featReader, labelReader, 2, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        DataSet ds = iter.next();

        // Features mask: seq1 is length 4 (all present), seq2 is length 2 (padded at start)
        INDArray featuresMask = ds.getFeaturesMaskArray();
        assertNotNull(featuresMask);

        // Seq2 (example index 1): should have zeros at START for ALIGN_END
        // mask[1, 0] and mask[1, 1] should be 0 (padding)
        // mask[1, 2] and mask[1, 3] should be 1 (real data)
        assertEquals(0.0, featuresMask.getDouble(1, 0), 1e-5, "ALIGN_END: padding should be at position 0");
        assertEquals(0.0, featuresMask.getDouble(1, 1), 1e-5, "ALIGN_END: padding should be at position 1");
        assertEquals(1.0, featuresMask.getDouble(1, 2), 1e-5, "ALIGN_END: real data at position 2");
        assertEquals(1.0, featuresMask.getDouble(1, 3), 1e-5, "ALIGN_END: real data at position 3");

        // Seq1 (example index 0): all 4 timesteps present
        for (int t = 0; t < 4; t++) {
            assertEquals(1.0, featuresMask.getDouble(0, t), 1e-5, "Seq1 should have all ones in mask");
        }

        // Labels mask: seq1 has 2 labels (padded at start), seq2 has 4 (all present)
        INDArray labelsMask = ds.getLabelsMaskArray();
        assertNotNull(labelsMask);
        assertEquals(0.0, labelsMask.getDouble(0, 0), 1e-5, "ALIGN_END labels: padding at start");
        assertEquals(0.0, labelsMask.getDouble(0, 1), 1e-5, "ALIGN_END labels: padding at start");
        assertEquals(1.0, labelsMask.getDouble(0, 2), 1e-5, "ALIGN_END labels: real data");
        assertEquals(1.0, labelsMask.getDouble(0, 3), 1e-5, "ALIGN_END labels: real data");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ALIGN_END: data is right-aligned to batch-wide max")
    void testAlignEndDataPosition(Nd4jBackend backend) throws Exception {
        // Seq1: 3 timesteps of features [1,2,3], [4,5,6], [7,8,9]
        // Seq2: 1 timestep of features [10,11,12]
        // With ALIGN_END and batch-wide maxTS=3:
        //   Seq1: data at t=0,1,2 (no padding)
        //   Seq2: padding at t=0,1; data at t=2

        List<List<Writable>> featSeq1 = new ArrayList<>();
        featSeq1.add(Arrays.asList((Writable) new DoubleWritable(1), new DoubleWritable(2), new DoubleWritable(3)));
        featSeq1.add(Arrays.asList((Writable) new DoubleWritable(4), new DoubleWritable(5), new DoubleWritable(6)));
        featSeq1.add(Arrays.asList((Writable) new DoubleWritable(7), new DoubleWritable(8), new DoubleWritable(9)));

        List<List<Writable>> featSeq2 = new ArrayList<>();
        featSeq2.add(Arrays.asList((Writable) new DoubleWritable(10), new DoubleWritable(11), new DoubleWritable(12)));

        // Labels: same length as features
        List<List<Writable>> labelSeq1 = new ArrayList<>();
        labelSeq1.add(Arrays.asList((Writable) new IntWritable(0)));
        labelSeq1.add(Arrays.asList((Writable) new IntWritable(1)));
        labelSeq1.add(Arrays.asList((Writable) new IntWritable(0)));

        List<List<Writable>> labelSeq2 = new ArrayList<>();
        labelSeq2.add(Arrays.asList((Writable) new IntWritable(1)));

        SequenceRecordReader featReader = new CollectionSequenceRecordReader(
                Arrays.asList(featSeq1, featSeq2));
        SequenceRecordReader labelReader = new CollectionSequenceRecordReader(
                Arrays.asList(labelSeq1, labelSeq2));

        DataSetIterator iter = new SequenceRecordReaderDataSetIterator(
                featReader, labelReader, 2, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        DataSet ds = iter.next();
        INDArray features = ds.getFeatures();

        // Shape should be [2, 3, 3] (2 examples, 3 features, 3 timesteps)
        assertEquals(2, features.size(0));
        assertEquals(3, features.size(1));
        assertEquals(3, features.size(2));

        // Seq2's data should be at the END (t=2), not at the start
        // features[1, :, 2] should be [10, 11, 12]
        assertEquals(10.0, features.getDouble(1, 0, 2), 1e-5, "Seq2 data should be at last timestep");
        assertEquals(11.0, features.getDouble(1, 1, 2), 1e-5);
        assertEquals(12.0, features.getDouble(1, 2, 2), 1e-5);

        // features[1, :, 0] and features[1, :, 1] should be 0 (padding)
        for (int f = 0; f < 3; f++) {
            assertEquals(0.0, features.getDouble(1, f, 0), 1e-5,
                    "Seq2 should have zero padding at t=0, f=" + f);
            assertEquals(0.0, features.getDouble(1, f, 1), 1e-5,
                    "Seq2 should have zero padding at t=1, f=" + f);
        }
    }

    // =========================================================================
    //  NormalizerStandardize with variable-length masked sequences
    // =========================================================================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("NormalizerStandardize.fit(iterator) with ALIGN_END produces valid stats")
    void testNormalizerWithAlignEndVariableLength(Nd4jBackend backend) throws Exception {
        File baseDir = temporaryFolder.toFile();
        File featuresDir = new File(baseDir, "features");
        File labelsDir = new File(baseDir, "labels");
        featuresDir.mkdirs();
        labelsDir.mkdirs();

        Random rng = new Random(12345);
        int numExamples = 20;
        int numFeatures = 3;

        // Generate variable-length CSV sequences
        int[] featureLengths = new int[numExamples];
        for (int i = 0; i < numExamples; i++) {
            int seqLen = 5 + rng.nextInt(6); // length 5-10
            featureLengths[i] = seqLen;
            generateFeatureCSV(new File(featuresDir, i + ".csv"), seqLen, numFeatures, rng);
            // Labels shorter than features to trigger mask
            int labelLen = Math.max(1, seqLen - 2 - rng.nextInt(3));
            generateLabelCSV(new File(labelsDir, i + ".csv"), labelLen, 2, rng);
        }

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(
                featuresDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(
                labelsDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));

        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(
                trainFeatures, trainLabels, 4, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);

        // This should not throw and should produce valid statistics
        normalizer.fit(trainData);

        INDArray mean = normalizer.getMean();
        INDArray std = normalizer.getStd();

        assertNotNull(mean, "Mean should not be null");
        assertNotNull(std, "Std should not be null");

        // No NaN or Inf
        for (int f = 0; f < numFeatures; f++) {
            assertFalse(Double.isNaN(mean.getDouble(f)), "Mean contains NaN at feature " + f);
            assertFalse(Double.isInfinite(mean.getDouble(f)), "Mean contains Inf at feature " + f);
            assertFalse(Double.isNaN(std.getDouble(f)), "Std contains NaN at feature " + f);
            assertFalse(Double.isInfinite(std.getDouble(f)), "Std contains Inf at feature " + f);
            assertTrue(std.getDouble(f) > 0, "Std should be positive at feature " + f);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("NormalizerStandardize.fit(iterator) vs fit(DataSet) loop gives different results")
    void testNormalizerFitIteratorVsFitDataSetLoop(Nd4jBackend backend) throws Exception {
        File baseDir = temporaryFolder.toFile();
        File featuresDir = new File(baseDir, "features2");
        File labelsDir = new File(baseDir, "labels2");
        featuresDir.mkdirs();
        labelsDir.mkdirs();

        Random rng = new Random(99);
        int numExamples = 16;
        int numFeatures = 3;
        int seqLen = 8;

        for (int i = 0; i < numExamples; i++) {
            generateFeatureCSV(new File(featuresDir, i + ".csv"), seqLen, numFeatures, rng);
            generateLabelCSV(new File(labelsDir, i + ".csv"), seqLen, 2, rng);
        }

        // Path 1: fit(iterator) - correct
        SequenceRecordReader feat1 = new CSVSequenceRecordReader();
        feat1.initialize(new NumberedFileInputSplit(
                featuresDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        SequenceRecordReader lab1 = new CSVSequenceRecordReader();
        lab1.initialize(new NumberedFileInputSplit(
                labelsDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        DataSetIterator iter1 = new SequenceRecordReaderDataSetIterator(
                feat1, lab1, 4, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        NormalizerStandardize normCorrect = new NormalizerStandardize();
        normCorrect.fitLabel(true);
        normCorrect.fit(iter1);

        INDArray correctMean = normCorrect.getMean().dup();
        INDArray correctStd = normCorrect.getStd().dup();

        // Path 2: fit(DataSet) in loop - buggy (only keeps last batch)
        SequenceRecordReader feat2 = new CSVSequenceRecordReader();
        feat2.initialize(new NumberedFileInputSplit(
                featuresDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        SequenceRecordReader lab2 = new CSVSequenceRecordReader();
        lab2.initialize(new NumberedFileInputSplit(
                labelsDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        DataSetIterator iter2 = new SequenceRecordReaderDataSetIterator(
                feat2, lab2, 4, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        NormalizerStandardize normBuggy = new NormalizerStandardize();
        normBuggy.fitLabel(true);
        int batchCount = 0;
        while (iter2.hasNext()) {
            normBuggy.fit(iter2.next()); // each call OVERWRITES previous
            batchCount++;
        }
        assertTrue(batchCount > 1, "Need multiple batches to demonstrate the bug");

        INDArray buggyMean = normBuggy.getMean().dup();

        // The buggy approach should NOT match the correct approach
        // (unless by coincidence, which is astronomically unlikely with random data)
        assertFalse(correctMean.equalsWithEps(buggyMean, 1e-3),
                "fit(DataSet) loop should produce different mean than fit(iterator) - "
                + "this demonstrates the accumulation bug");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("NormalizerStandardize: ALIGN_END and ALIGN_START produce same stats on same data")
    void testNormalizerAlignEndVsAlignStartSameStats(Nd4jBackend backend) throws Exception {
        File baseDir = temporaryFolder.toFile();
        File featuresDir = new File(baseDir, "features3");
        File labelsDir = new File(baseDir, "labels3");
        featuresDir.mkdirs();
        labelsDir.mkdirs();

        Random rng = new Random(7777);
        int numExamples = 20;
        int numFeatures = 3;

        for (int i = 0; i < numExamples; i++) {
            int seqLen = 5 + rng.nextInt(6);
            generateFeatureCSV(new File(featuresDir, i + ".csv"), seqLen, numFeatures, rng);
            int labelLen = Math.max(1, seqLen - rng.nextInt(3));
            generateLabelCSV(new File(labelsDir, i + ".csv"), labelLen, 2, rng);
        }

        // ALIGN_END
        SequenceRecordReader featEnd = new CSVSequenceRecordReader();
        featEnd.initialize(new NumberedFileInputSplit(
                featuresDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        SequenceRecordReader labEnd = new CSVSequenceRecordReader();
        labEnd.initialize(new NumberedFileInputSplit(
                labelsDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        DataSetIterator iterEnd = new SequenceRecordReaderDataSetIterator(
                featEnd, labEnd, 4, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        NormalizerStandardize normEnd = new NormalizerStandardize();
        normEnd.fitLabel(true);
        normEnd.fit(iterEnd);

        // ALIGN_START
        SequenceRecordReader featStart = new CSVSequenceRecordReader();
        featStart.initialize(new NumberedFileInputSplit(
                featuresDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        SequenceRecordReader labStart = new CSVSequenceRecordReader();
        labStart.initialize(new NumberedFileInputSplit(
                labelsDir.getAbsolutePath() + "/%d.csv", 0, numExamples - 1));
        DataSetIterator iterStart = new SequenceRecordReaderDataSetIterator(
                featStart, labStart, 4, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_START);

        NormalizerStandardize normStart = new NormalizerStandardize();
        normStart.fitLabel(true);
        normStart.fit(iterStart);

        // Same underlying data -> same statistics regardless of alignment mode
        assertTrue(normEnd.getMean().equalsWithEps(normStart.getMean(), 1e-3),
                "ALIGN_END and ALIGN_START should produce same mean. "
                + "END=" + normEnd.getMean() + " START=" + normStart.getMean());
        assertTrue(normEnd.getStd().equalsWithEps(normStart.getStd(), 1e-3),
                "ALIGN_END and ALIGN_START should produce same std. "
                + "END=" + normEnd.getStd() + " START=" + normStart.getStd());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ALIGN_END: variable-length batch mask has correct structure")
    void testAlignEndVariableLengthBatchMask(Nd4jBackend backend) throws Exception {
        // 4 sequences with different lengths: 3, 5, 2, 4
        // maxTS = 5
        // With ALIGN_END, shorter sequences should be padded at the START
        List<List<List<Writable>>> featureSequences = new ArrayList<>();
        List<List<List<Writable>>> labelSequences = new ArrayList<>();
        int[] seqLengths = {3, 5, 2, 4};

        for (int s = 0; s < 4; s++) {
            List<List<Writable>> featSeq = new ArrayList<>();
            List<List<Writable>> labSeq = new ArrayList<>();
            for (int t = 0; t < seqLengths[s]; t++) {
                featSeq.add(Arrays.asList(
                        (Writable) new DoubleWritable(s * 100 + t),
                        new DoubleWritable(s * 100 + t + 10)));
                labSeq.add(Arrays.asList((Writable) new IntWritable(t % 2)));
            }
            featureSequences.add(featSeq);
            labelSequences.add(labSeq);
        }

        SequenceRecordReader featReader = new CollectionSequenceRecordReader(featureSequences);
        SequenceRecordReader labelReader = new CollectionSequenceRecordReader(labelSequences);

        DataSetIterator iter = new SequenceRecordReaderDataSetIterator(
                featReader, labelReader, 4, 2, false,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        DataSet ds = iter.next();
        INDArray featuresMask = ds.getFeaturesMaskArray();
        assertNotNull(featuresMask);

        // maxTS across all = 5
        assertEquals(4, featuresMask.size(0));
        assertEquals(5, featuresMask.size(1));

        // For each sequence, verify padding is at START
        for (int s = 0; s < 4; s++) {
            int padLen = 5 - seqLengths[s];
            // First padLen positions should be 0
            for (int t = 0; t < padLen; t++) {
                assertEquals(0.0, featuresMask.getDouble(s, t), 1e-5,
                        "Seq " + s + " should have padding at t=" + t);
            }
            // Remaining positions should be 1
            for (int t = padLen; t < 5; t++) {
                assertEquals(1.0, featuresMask.getDouble(s, t), 1e-5,
                        "Seq " + s + " should have real data at t=" + t);
            }
        }

        // Verify that features data is placed at the end
        INDArray features = ds.getFeatures();
        // Seq 0 (length 3): data at t=2,3,4
        assertEquals(0.0, features.getDouble(0, 0, 2), 1e-5);  // first feature, first data timestep
        assertEquals(10.0, features.getDouble(0, 1, 2), 1e-5);  // second feature, first data timestep

        // Seq 2 (length 2): data at t=3,4
        assertEquals(200.0, features.getDouble(2, 0, 3), 1e-5);
        assertEquals(210.0, features.getDouble(2, 1, 3), 1e-5);
    }

    // =========================================================================
    //  Helper methods
    // =========================================================================

    private static void generateFeatureCSV(File file, int seqLen, int numFeatures, Random rng) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(file))) {
            for (int t = 0; t < seqLen; t++) {
                StringBuilder sb = new StringBuilder();
                for (int f = 0; f < numFeatures; f++) {
                    if (f > 0) sb.append(",");
                    sb.append(String.format("%.4f", rng.nextGaussian() * 10 + 5));
                }
                pw.println(sb.toString());
            }
        }
    }

    private static void generateLabelCSV(File file, int seqLen, int numClasses, Random rng) throws Exception {
        try (PrintWriter pw = new PrintWriter(new FileWriter(file))) {
            for (int t = 0; t < seqLen; t++) {
                pw.println(rng.nextInt(numClasses));
            }
        }
    }
}
