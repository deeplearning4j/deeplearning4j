package org.nd4j.linalg.dataset;

import lombok.Getter;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.api.preprocessor.*;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.*;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertEquals;

/**
 * @author Ede Meijer
 */
@RunWith(Parameterized.class)
public class NormalizerSerializerTest extends BaseNd4jTest {
    private File tmpFile;
    private NormalizerSerializer SUT;

    public NormalizerSerializerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Before
    public void setUp() throws IOException {
        tmpFile = File.createTempFile("test", "preProcessor");
        tmpFile.deleteOnExit();

        SUT = NormalizerSerializer.getDefault();
    }

    @Test
    public void testNormalizerStandardizeNotFitLabels() throws Exception {
        NormalizerStandardize original = new NormalizerStandardize(Nd4j.create(new double[] {0.5, 1.5}),
                        Nd4j.create(new double[] {2.5, 3.5}));

        SUT.write(original, tmpFile);
        NormalizerStandardize restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testNormalizerStandardizeFitLabels() throws Exception {
        NormalizerStandardize original = new NormalizerStandardize(Nd4j.create(new double[] {0.5, 1.5}),
                        Nd4j.create(new double[] {2.5, 3.5}), Nd4j.create(new double[] {4.5, 5.5}),
                        Nd4j.create(new double[] {6.5, 7.5}));
        original.fitLabel(true);

        SUT.write(original, tmpFile);
        NormalizerStandardize restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testNormalizerMinMaxScalerNotFitLabels() throws Exception {
        NormalizerMinMaxScaler original = new NormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5}));

        SUT.write(original, tmpFile);
        NormalizerMinMaxScaler restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testNormalizerMinMaxScalerFitLabels() throws Exception {
        NormalizerMinMaxScaler original = new NormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5}));
        original.setLabelStats(Nd4j.create(new double[] {4.5, 5.5}), Nd4j.create(new double[] {6.5, 7.5}));
        original.fitLabel(true);

        SUT.write(original, tmpFile);
        NormalizerMinMaxScaler restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerStandardizeNotFitLabels() throws Exception {
        MultiNormalizerStandardize original = new MultiNormalizerStandardize();
        original.setFeatureStats(asList(
                        new DistributionStats(Nd4j.create(new double[] {0.5, 1.5}),
                                        Nd4j.create(new double[] {2.5, 3.5})),
                        new DistributionStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));

        SUT.write(original, tmpFile);
        MultiNormalizerStandardize restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerStandardizeFitLabels() throws Exception {
        MultiNormalizerStandardize original = new MultiNormalizerStandardize();
        original.setFeatureStats(asList(
                        new DistributionStats(Nd4j.create(new double[] {0.5, 1.5}),
                                        Nd4j.create(new double[] {2.5, 3.5})),
                        new DistributionStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));
        original.setLabelStats(asList(
                        new DistributionStats(Nd4j.create(new double[] {0.5, 1.5}),
                                        Nd4j.create(new double[] {2.5, 3.5})),
                        new DistributionStats(Nd4j.create(new double[] {4.5}), Nd4j.create(new double[] {7.5})),
                        new DistributionStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));
        original.fitLabel(true);

        SUT.write(original, tmpFile);
        MultiNormalizerStandardize restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerMinMaxScalerNotFitLabels() throws Exception {
        MultiNormalizerMinMaxScaler original = new MultiNormalizerMinMaxScaler(0.1, 0.9);
        original.setFeatureStats(asList(
                        new MinMaxStats(Nd4j.create(new double[] {0.5, 1.5}), Nd4j.create(new double[] {2.5, 3.5})),
                        new MinMaxStats(Nd4j.create(new double[] {4.5, 5.5, 6.5}),
                                        Nd4j.create(new double[] {7.5, 8.5, 9.5}))));

        SUT.write(original, tmpFile);
        MultiNormalizerMinMaxScaler restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerMinMaxScalerFitLabels() throws Exception {
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

        SUT.write(original, tmpFile);
        MultiNormalizerMinMaxScaler restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerHybridEmpty() throws Exception {
        MultiNormalizerHybrid original = new MultiNormalizerHybrid();
        original.setInputStats(new HashMap<Integer, NormalizerStats>());
        original.setOutputStats(new HashMap<Integer, NormalizerStats>());

        SUT.write(original, tmpFile);
        MultiNormalizerHybrid restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerHybridGlobalStats() throws Exception {
        MultiNormalizerHybrid original = new MultiNormalizerHybrid().minMaxScaleAllInputs().standardizeAllOutputs();

        Map<Integer, NormalizerStats> inputStats = new HashMap<>();
        inputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {1, 2}), Nd4j.create(new float[] {3, 4})));
        inputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {5, 6}), Nd4j.create(new float[] {7, 8})));

        Map<Integer, NormalizerStats> outputStats = new HashMap<>();
        outputStats.put(0, new DistributionStats(Nd4j.create(new float[] {9, 10}), Nd4j.create(new float[] {11, 12})));
        outputStats.put(0, new DistributionStats(Nd4j.create(new float[] {13, 14}), Nd4j.create(new float[] {15, 16})));

        original.setInputStats(inputStats);
        original.setOutputStats(outputStats);

        SUT.write(original, tmpFile);
        MultiNormalizerHybrid restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test
    public void testMultiNormalizerHybridGlobalAndSpecificStats() throws Exception {
        MultiNormalizerHybrid original = new MultiNormalizerHybrid().standardizeAllInputs().minMaxScaleInput(0, -5, 5)
                        .minMaxScaleAllOutputs(-10, 10).standardizeOutput(1);

        Map<Integer, NormalizerStats> inputStats = new HashMap<>();
        inputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {1, 2}), Nd4j.create(new float[] {3, 4})));
        inputStats.put(1, new DistributionStats(Nd4j.create(new float[] {5, 6}), Nd4j.create(new float[] {7, 8})));

        Map<Integer, NormalizerStats> outputStats = new HashMap<>();
        outputStats.put(0, new MinMaxStats(Nd4j.create(new float[] {9, 10}), Nd4j.create(new float[] {11, 12})));
        outputStats.put(1, new DistributionStats(Nd4j.create(new float[] {13, 14}), Nd4j.create(new float[] {15, 16})));

        original.setInputStats(inputStats);
        original.setOutputStats(outputStats);

        SUT.write(original, tmpFile);
        MultiNormalizerHybrid restored = SUT.restore(tmpFile);

        assertEquals(original, restored);
    }

    @Test(expected = RuntimeException.class)
    public void testCustomNormalizerWithoutRegisteredStrategy() throws Exception {
        SUT.write(new MyNormalizer(123), tmpFile);
    }

    @Test
    public void testCustomNormalizer() throws Exception {
        MyNormalizer original = new MyNormalizer(42);

        SUT.addStrategy(new MyNormalizerSerializerStrategy());

        SUT.write(original, tmpFile);
        MyNormalizer restored = SUT.restore(tmpFile);

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
