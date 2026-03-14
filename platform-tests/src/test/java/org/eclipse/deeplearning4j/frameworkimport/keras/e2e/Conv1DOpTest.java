package org.eclipse.deeplearning4j.frameworkimport.keras.e2e;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal test for Conv1D op correctness.
 */
public class Conv1DOpTest {

    @Test
    void testConv1DK2Valid() {
        INDArray input = Nd4j.create(new double[]{1, 2, 3, 4}, new int[]{1, 1, 4});
        INDArray weights = Nd4j.create(new double[]{1, 1}, new int[]{2, 1, 1});
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(2).s(1).d(1).p(0)
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(PaddingMode.VALID)
                .build();
        Conv1D op = new Conv1D(new INDArray[]{input, weights}, null, conf);
        Nd4j.exec(op);
        INDArray output = op.getOutputArgument(0);
        INDArray expected = Nd4j.create(new double[]{3, 5, 7}, new int[]{1, 1, 3});
        assertTrue(expected.equalsWithEps(output, 1e-5),
            "Expected " + expected + " but got " + output);
    }

    @Test
    void testConv1DK2MultiChannel() {
        INDArray input = Nd4j.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8
        }, new int[]{1, 2, 4});
        INDArray weights = Nd4j.create(new double[]{
            1, 0, 0, 1, 1, 0, 0, 1
        }, new int[]{2, 2, 2});
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(2).s(1).d(1).p(0)
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(PaddingMode.VALID)
                .build();
        Conv1D op = new Conv1D(new INDArray[]{input, weights}, null, conf);
        Nd4j.exec(op);
        INDArray output = op.getOutputArgument(0);
        INDArray expected = Nd4j.create(new double[]{3, 5, 7, 11, 13, 15}, new int[]{1, 2, 3});
        assertTrue(expected.equalsWithEps(output, 1e-5),
            "Expected " + expected + " but got " + output);
    }

    @Test
    void testConv1DK3Valid() {
        INDArray input = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{1, 1, 6});
        INDArray weights = Nd4j.create(new double[]{1, 1, 1}, new int[]{3, 1, 1});
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(3).s(1).d(1).p(0)
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(PaddingMode.VALID)
                .build();
        Conv1D op = new Conv1D(new INDArray[]{input, weights}, null, conf);
        Nd4j.exec(op);
        INDArray output = op.getOutputArgument(0);
        INDArray expected = Nd4j.create(new double[]{6, 9, 12, 15}, new int[]{1, 1, 4});
        assertTrue(expected.equalsWithEps(output, 1e-5),
            "Expected " + expected + " but got " + output);
    }

    @Test
    void testConv1DK3MultiChannel() {
        INDArray input = Nd4j.create(new double[]{
            1, 2, 3, 4, 5, 6, 7, 8
        }, new int[]{1, 2, 4});
        INDArray weights = Nd4j.create(new double[]{
            1, 1, 1, 1, 1, 1
        }, new int[]{3, 2, 1});
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(3).s(1).d(1).p(0)
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(PaddingMode.VALID)
                .build();
        Conv1D op = new Conv1D(new INDArray[]{input, weights}, null, conf);
        Nd4j.exec(op);
        INDArray output = op.getOutputArgument(0);
        INDArray expected = Nd4j.create(new double[]{24, 30}, new int[]{1, 1, 2});
        assertTrue(expected.equalsWithEps(output, 1e-5),
            "Expected " + expected + " but got " + output);
    }

    @Test
    void testConv1DDilation2() {
        INDArray input = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{1, 1, 6});
        INDArray weights = Nd4j.create(new double[]{1, 1}, new int[]{2, 1, 1});
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(2).s(1).d(2).p(0)
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(PaddingMode.VALID)
                .build();
        Conv1D op = new Conv1D(new INDArray[]{input, weights}, null, conf);
        Nd4j.exec(op);
        INDArray output = op.getOutputArgument(0);
        INDArray expected = Nd4j.create(new double[]{4, 6, 8, 10}, new int[]{1, 1, 4});
        assertTrue(expected.equalsWithEps(output, 1e-5),
            "Expected " + expected + " but got " + output);
    }
}
