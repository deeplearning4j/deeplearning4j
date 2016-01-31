package org.deeplearning4j.ui.weights;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class HistogramTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGetBins() throws Exception {
        INDArray array = Nd4j.create(new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0});

        HistogramBin histogram = new HistogramBin.Builder(array)
                .setBinCount(10)
                .build();

        assertEquals(0.1, histogram.getMin(), 0.001);
        assertEquals(1.0, histogram.getMax(), 0.001);

        System.out.println("Result: " + histogram.getBins());

        assertEquals(2, histogram.getBins().getDouble(9), 0.001);
    }

    @Test
    public void testGetData() throws Exception {
        INDArray array = Nd4j.create(new double[] {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0});

        HistogramBin histogram = new HistogramBin.Builder(array)
                .setBinCount(10)
                .build();

        assertEquals(0.1, histogram.getMin(), 0.001);
        assertEquals(1.0, histogram.getMax(), 0.001);

        System.out.println("Result: " + histogram.getData());

        assertEquals(10, histogram.getData().size()

        );
    }
}