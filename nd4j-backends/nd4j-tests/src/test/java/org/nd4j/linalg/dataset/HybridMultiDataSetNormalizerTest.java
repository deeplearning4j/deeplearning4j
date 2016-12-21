package org.nd4j.linalg.dataset;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.TestMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.HybridMultiDataSetNormalizer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.HybridMultiDataSetNormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.*;

@RunWith(Parameterized.class)
public class HybridMultiDataSetNormalizerTest extends BaseNd4jTest {
    private static final double TOLERANCE_PERC = 0.01; // 0.01% of correct value
    private static final int N_SAMPLES = 5120, INPUT1_SCALE = 1, INPUT2_SCALE = 2, OUTPUT1_SCALE = 3, OUTPUT2_SCALE = 4;

    private HybridMultiDataSetNormalizer SUT;
    private MultiDataSet data;
    private double meanNaturalNums;
    private double stdNaturalNums;

    @Before
    public void setUp() {
        SUT = new HybridMultiDataSetNormalizer().standardizeAllInputs().minMaxScaleInput(1).standardizeAllOutputs();

        // Prepare test data
        INDArray values = Nd4j.linspace(1, N_SAMPLES, N_SAMPLES).transpose();
        INDArray input1 = values.mul(INPUT1_SCALE);
        INDArray input2 = values.mul(INPUT2_SCALE);
        INDArray output1 = values.mul(OUTPUT1_SCALE);
        INDArray output2 = values.mul(OUTPUT2_SCALE);

        data = new MultiDataSet(
            new INDArray[]{input1, input2},
            new INDArray[]{output1, output2}
        );

        meanNaturalNums = (N_SAMPLES + 1) / 2.0;
        stdNaturalNums = Math.sqrt((N_SAMPLES * N_SAMPLES - 1) / 12.0);
    }

    public HybridMultiDataSetNormalizerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testMultipleInputsAndOutputsWithDataSet() {
        SUT.fit(data);
        assertExpectedMeanStd();
    }

    @Test
    public void testMultipleInputsAndOutputsWithIterator() {
        MultiDataSetIterator iter = new TestMultiDataSetIterator(1, data);
        SUT.fit(iter);
        assertExpectedMeanStd();
    }

    @Test
    public void testRevertFeaturesINDArray() {
        SUT.fit(data);

        MultiDataSet transformed = data.copy();
        SUT.preProcess(transformed);

        INDArray reverted = transformed.getFeatures(0).dup();
        SUT.revertFeatures(reverted, null, 0);

        assertNotEquals(reverted, transformed.getFeatures(0));

        SUT.revert(transformed);
        assertEquals(reverted, transformed.getFeatures(0));
    }

    @Test
    public void testRevertLabelsINDArray() {
        SUT.fit(data);

        MultiDataSet transformed = data.copy();
        SUT.preProcess(transformed);

        INDArray reverted = transformed.getLabels(0).dup();
        SUT.revertLabels(reverted, null, 0);

        assertNotEquals(reverted, transformed.getLabels(0));

        SUT.revert(transformed);
        assertEquals(reverted, transformed.getLabels(0));
    }

    @Test
    public void testRevertMultiDataSet() {
        SUT.fit(data);

        MultiDataSet transformed = data.copy();
        SUT.preProcess(transformed);

        double diffBeforeRevert = getMaxRelativeDifference(data, transformed);
        assertTrue(diffBeforeRevert > TOLERANCE_PERC);

        SUT.revert(transformed);

        double diffAfterRevert = getMaxRelativeDifference(data, transformed);
        assertTrue(diffAfterRevert < TOLERANCE_PERC);
    }

    @Test
    public void testFullyMaskedData() {
        MultiDataSetIterator iter = new TestMultiDataSetIterator(
            1,
            new MultiDataSet(
                new INDArray[]{Nd4j.create(new float[]{1}).reshape(1, 1, 1)},
                new INDArray[]{Nd4j.create(new float[]{2}).reshape(1, 1, 1)}
            ),
            new MultiDataSet(
                new INDArray[]{Nd4j.create(new float[]{2}).reshape(1, 1, 1)},
                new INDArray[]{Nd4j.create(new float[]{4}).reshape(1, 1, 1)},
                null,
                new INDArray[]{Nd4j.create(new float[]{0}).reshape(1, 1)}
            )
        );

        SUT.fit(iter);

        // The label mean should be 2, as the second row with 4 is masked.
        assertEquals(2, getLabelMean(0), 1e-6);
    }

    @Test
    public void testSerializer() throws IOException {
        SUT.fit(data);

        File tmp = File.createTempFile("test", "hybrid-multi-norm");
        tmp.deleteOnExit();
        HybridMultiDataSetNormalizerSerializer.write(SUT, tmp);

        HybridMultiDataSetNormalizer restored = HybridMultiDataSetNormalizerSerializer.restore(tmp);

        assertEquals(SUT, restored);
    }

    private double getMaxRelativeDifference(MultiDataSet a, MultiDataSet b) {
        double max = 0;
        for (int i = 0; i < a.getFeatures().length; i++) {
            INDArray inputA = a.getFeatures()[i];
            INDArray inputB = b.getFeatures()[i];
            INDArray delta = Transforms.abs(inputA.sub(inputB)).div(inputB);
            double maxdeltaPerc = delta.max(0, 1).mul(100).getDouble(0, 0);
            if (maxdeltaPerc > max) {
                max = maxdeltaPerc;
            }
        }
        return max;
    }

    private void assertExpectedMeanStd() {
        assertSmallDifference(meanNaturalNums * INPUT1_SCALE, getFeatureMean(0));
        assertSmallDifference(stdNaturalNums * INPUT1_SCALE, getFeatureStd(0));

        assertSmallDifference(INPUT2_SCALE, getFeatureMin(1));
        assertSmallDifference(INPUT2_SCALE * N_SAMPLES, getFeatureMax(1));

        assertSmallDifference(meanNaturalNums * OUTPUT1_SCALE, getLabelMean(0));
        assertSmallDifference(stdNaturalNums * OUTPUT1_SCALE, getLabelStd(0));

        assertSmallDifference(meanNaturalNums * OUTPUT2_SCALE, getLabelMean(1));
        assertSmallDifference(stdNaturalNums * OUTPUT2_SCALE, getLabelStd(1));
    }

    private double getFeatureMean(int input) {
        return ((DistributionStats) SUT.getInputStats(input)).getMean().getDouble(0);
    }

    private double getFeatureStd(int input) {
        return ((DistributionStats) SUT.getInputStats(input)).getStd().getDouble(0);
    }

    private double getLabelMean(int output) {
        return ((DistributionStats) SUT.getOutputStats(output)).getMean().getDouble(0);
    }

    private double getLabelStd(int output) {
        return ((DistributionStats) SUT.getOutputStats(output)).getStd().getDouble(0);
    }

    private double getFeatureMin(int input) {
        return ((MinMaxStats) SUT.getInputStats(input)).getLower().getDouble(0);
    }

    private double getFeatureMax(int input) {
        return ((MinMaxStats) SUT.getInputStats(input)).getUpper().getDouble(0);
    }

    private void assertSmallDifference(double expected, double actual) {
        double delta = Math.abs(expected - actual);
        double deltaPerc = (delta / expected) * 100;
        assertTrue(
            String.format("Failed to assert that expected value %f is close to actual value %f", expected, actual),
            deltaPerc < TOLERANCE_PERC
        );
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
