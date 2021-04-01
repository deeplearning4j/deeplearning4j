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

package org.nd4j.linalg.dataset;

import lombok.Getter;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.dataset.api.preprocessor.AbstractDataSetNormalizer;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MinMaxStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerHybrid;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.CustomSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import static java.util.Arrays.asList;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * @author Ede Meijer
 */
@Tag(TagNames.NDARRAY_ETL)
@NativeTag
@Tag(TagNames.FILE_IO)
public class NormalizerSerializerTest extends BaseNd4jTestWithBackends {
    @TempDir  File tmpFile;
    private static NormalizerSerializer SUT;


    @BeforeAll
    public static void setUp() throws IOException {
        SUT = NormalizerSerializer.getDefault();
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testImagePreProcessingScaler(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0,1);
        SUT.write(imagePreProcessingScaler,normalizerFile);

        ImagePreProcessingScaler restored = SUT.restore(normalizerFile);
        assertEquals(imagePreProcessingScaler,restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizerStandardizeNotFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        NormalizerStandardize original = new NormalizerStandardize(Nd4j.create(new double[] {0.5, 1.5}).reshape(1, -1),
                Nd4j.create(new double[] {2.5, 3.5}).reshape(1, -1));

        SUT.write(original, normalizerFile);
        NormalizerStandardize restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizerStandardizeFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        NormalizerStandardize original = new NormalizerStandardize(Nd4j.create(new double[] {0.5, 1.5}).reshape(1, -1),
                Nd4j.create(new double[] {2.5, 3.5}).reshape(1, -1), Nd4j.create(new double[] {4.5, 5.5}).reshape(1, -1),
                Nd4j.create(new double[] {6.5, 7.5}).reshape(1, -1));
        original.fitLabel(true);

        SUT.write(original, normalizerFile);
        NormalizerStandardize restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizerMinMaxScalerNotFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        NormalizerMinMaxScaler original = new NormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(Nd4j.create(new double[] {0.5, 1.5}).reshape(1, -1), Nd4j.create(new double[] {2.5, 3.5}).reshape(1, -1));

        SUT.write(original, normalizerFile);
        NormalizerMinMaxScaler restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testNormalizerMinMaxScalerFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        NormalizerMinMaxScaler original = new NormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5}));
        original.setLabelStats(Nd4j.create(new double[] {4.5, 5.5}), Nd4j.create(new double[] {6.5, 7.5}));
        original.fitLabel(true);

        SUT.write(original, normalizerFile);
        NormalizerMinMaxScaler restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerStandardizeNotFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerStandardize original = new MultiNormalizerStandardize();
        original.setFeatureStats(asList(
                new DistributionStats(Nd4j.create(new double[] {0.5, 1.5}).reshape(1, -1),
                        Nd4j.create(new double[] {2.5, 3.5}).reshape(1, -1)),
                new DistributionStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}).reshape(1, -1),
                        Nd4j.create(new double[] {7.5, 8.5, 9.5}).reshape(1, -1))));

        SUT.write(original, normalizerFile);
        MultiNormalizerStandardize restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerStandardizeFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerStandardize original = new MultiNormalizerStandardize();
        original.setFeatureStats(asList(
                new DistributionStats(Nd4j.create(new double[] {0.5, 1.5}).reshape(1, -1),
                        Nd4j.create(new double[] {2.5, 3.5}).reshape(1, -1)),
                new DistributionStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}).reshape(1, -1),
                        Nd4j.create(new double[] {7.5, 8.5, 9.5}).reshape(1, -1))));
        original.setLabelStats(asList(
                new DistributionStats(Nd4j.create(new double[] {0.5, 1.5}).reshape(1, -1),
                        Nd4j.create(new double[] {2.5, 3.5}).reshape(1, -1)),
                new DistributionStats(Nd4j.create(new double[] {4.5}).reshape(1, -1), Nd4j.create(new double[] {7.5}).reshape(1, -1)),
                new DistributionStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}).reshape(1, -1),
                        Nd4j.create(new double[] {7.5, 8.5, 9.5}).reshape(1, -1))));
        original.fitLabel(true);

        SUT.write(original, normalizerFile);
        MultiNormalizerStandardize restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerMinMaxScalerNotFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerMinMaxScaler original = new MultiNormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(asList(
                new MinMaxStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5})),
                new MinMaxStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));

        SUT.write(original, normalizerFile);
        MultiNormalizerMinMaxScaler restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerMinMaxScalerFitLabels(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerMinMaxScaler original = new MultiNormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(asList(
                new MinMaxStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5})),
                new MinMaxStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));
        original.setLabelStats(asList(
                new MinMaxStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5})),
                new MinMaxStats(Nd4j.create(new double[] {4.5}), Nd4j.create(new double[] {7.5})),
                new MinMaxStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));
        original.fitLabel(true);

        SUT.write(original, normalizerFile);
        MultiNormalizerMinMaxScaler restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerHybridEmpty(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerHybrid original = new MultiNormalizerHybrid();
        original.setInputStats(new HashMap<>());
        original.setOutputStats(new HashMap<>());

        SUT.write(original, normalizerFile);
        MultiNormalizerHybrid restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerHybridGlobalStats(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerHybrid original = new MultiNormalizerHybrid().minMaxScaleAllInputs().standardizeAllOutputs();

        Map<Integer, NormalizerStats> inputStats = new HashMap<>();
        inputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {1, 2}).reshape(1, -1), Nd4j.create(new float[] {3, 4}).reshape(1, -1)));
        inputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {5, 6}).reshape(1, -1), Nd4j.create(new float[] {7, 8}).reshape(1, -1)));

        Map<Integer, NormalizerStats> outputStats = new HashMap<>();
        outputStats.put(0, new DistributionStats(Nd4j.create(new float[] {9, 10}).reshape(1, -1), Nd4j.create(new float[] {11, 12}).reshape(1, -1)));
        outputStats.put(0, new DistributionStats(Nd4j.create(new float[] {13, 14}).reshape(1, -1), Nd4j.create(new float[] {15, 16}).reshape(1, -1)));

        original.setInputStats(inputStats);
        original.setOutputStats(outputStats);

        SUT.write(original, normalizerFile);
        MultiNormalizerHybrid restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testMultiNormalizerHybridGlobalAndSpecificStats(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MultiNormalizerHybrid original = new MultiNormalizerHybrid().standardizeAllInputs().minMaxScaleInput(0, -5, 5)
                .minMaxScaleAllOutputs(-10, 10).standardizeOutput(1);

        Map<Integer, NormalizerStats> inputStats = new HashMap<>();
        inputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {1, 2}).reshape(1, -1), Nd4j.create(new float[] {3, 4}).reshape(1, -1)));
        inputStats.put(1, new DistributionStats(Nd4j.create(new float[] {5, 6}).reshape(1, -1), Nd4j.create(new float[] {7, 8}).reshape(1, -1)));

        Map<Integer, NormalizerStats> outputStats = new HashMap<>();
        outputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {9, 10}).reshape(1, -1), Nd4j.create(new float[] {11, 12}).reshape(1, -1)));
        outputStats.put(1, new DistributionStats(Nd4j.create(new float[] {13, 14}).reshape(1, -1), Nd4j.create(new float[] {15, 16}).reshape(1, -1)));

        original.setInputStats(inputStats);
        original.setOutputStats(outputStats);

        SUT.write(original, normalizerFile);
        MultiNormalizerHybrid restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Execution(ExecutionMode.SAME_THREAD)
    public void testCustomNormalizerWithoutRegisteredStrategy(Nd4jBackend backend) throws Exception {
        assertThrows(RuntimeException.class, () -> {
            File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();
            SUT.write(new MyNormalizer(123), normalizerFile);

        });
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCustomNormalizer(Nd4jBackend backend) throws Exception {
        File normalizerFile = Files.createTempFile(tmpFile.toPath(),"pre-process-" + UUID.randomUUID().toString(),"bin").toFile();

        MyNormalizer original = new MyNormalizer(42);

        SUT.addStrategy(new MyNormalizerSerializerStrategy());

        SUT.write(original, normalizerFile);
        MyNormalizer restored = SUT.restore(normalizerFile);

        assertEquals(original, restored);
    }

    public static class MyNormalizer extends AbstractDataSetNormalizer<MinMaxStats> {
        @Getter
        private final int foo;

        public MyNormalizer(int foo) {
            super(new MinMaxStrategy());
            this.foo = foo;
            setFeatureStats(new MinMaxStats(Nd4j.zeros(1), Nd4j.ones(1)));
        }

        @Override
        public NormalizerType getType() {
            return NormalizerType.CUSTOM;
        }

        @Override
        protected NormalizerStats.Builder newBuilder() {
            return new MinMaxStats.Builder();
        }
    }

    public static class MyNormalizerSerializerStrategy extends CustomSerializerStrategy<MyNormalizer> {
        @Override
        public Class<MyNormalizer> getSupportedClass() {
            return MyNormalizer.class;
        }

        @Override
        public void write(MyNormalizer normalizer, OutputStream stream) throws IOException {
            new DataOutputStream(stream).writeInt(normalizer.getFoo());
        }

        @Override
        public MyNormalizer restore(InputStream stream) throws IOException {
            return new MyNormalizer(new DataInputStream(stream).readInt());
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
