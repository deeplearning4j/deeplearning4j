package org.nd4j.linalg.dataset.api.preprocessor;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class RGBtoGrayscaleDataSetPreProcessorTest {

    @Test(expected = NullPointerException.class)
    public void when_dataSetIsNull_expect_NullPointerException() {
        // Assemble
        RGBtoGrayscaleDataSetPreProcessor sut = new RGBtoGrayscaleDataSetPreProcessor();

        // Act
        sut.preProcess(null);
    }

    @Test
    public void when_dataSetIsEmpty_expect_EmptyDataSet() {
        // Assemble
        RGBtoGrayscaleDataSetPreProcessor sut = new RGBtoGrayscaleDataSetPreProcessor();
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(ds.isEmpty());
    }

    @Test
    public void when_colorsAreConverted_expect_grayScaleResult() {
        // Assign
        int numChannels = 3;
        int height = 1;
        int width = 5;

        RGBtoGrayscaleDataSetPreProcessor sut = new RGBtoGrayscaleDataSetPreProcessor();
        INDArray input = Nd4j.create(2, numChannels, height, width);

        // Black, Example 1
        input.putScalar(0, 0, 0, 0, 0.0 );
        input.putScalar(0, 1, 0, 0, 0.0 );
        input.putScalar(0, 2, 0, 0, 0.0 );

        // White, Example 1
        input.putScalar(0, 0, 0, 1, 255.0 );
        input.putScalar(0, 1, 0, 1, 255.0 );
        input.putScalar(0, 2, 0, 1, 255.0 );

        // Red, Example 1
        input.putScalar(0, 0, 0, 2, 255.0 );
        input.putScalar(0, 1, 0, 2, 0.0 );
        input.putScalar(0, 2, 0, 2, 0.0 );

        // Green, Example 1
        input.putScalar(0, 0, 0, 3, 0.0 );
        input.putScalar(0, 1, 0, 3, 255.0 );
        input.putScalar(0, 2, 0, 3, 0.0 );

        // Blue, Example 1
        input.putScalar(0, 0, 0, 4, 0.0 );
        input.putScalar(0, 1, 0, 4, 0.0 );
        input.putScalar(0, 2, 0, 4, 255.0 );


        // Black, Example 2
        input.putScalar(1, 0, 0, 4, 0.0 );
        input.putScalar(1, 1, 0, 4, 0.0 );
        input.putScalar(1, 2, 0, 4, 0.0 );

        // White, Example 2
        input.putScalar(1, 0, 0, 3, 255.0 );
        input.putScalar(1, 1, 0, 3, 255.0 );
        input.putScalar(1, 2, 0, 3, 255.0 );

        // Red, Example 2
        input.putScalar(1, 0, 0, 2, 255.0 );
        input.putScalar(1, 1, 0, 2, 0.0 );
        input.putScalar(1, 2, 0, 2, 0.0 );

        // Green, Example 2
        input.putScalar(1, 0, 0, 1, 0.0 );
        input.putScalar(1, 1, 0, 1, 255.0 );
        input.putScalar(1, 2, 0, 1, 0.0 );

        // Blue, Example 2
        input.putScalar(1, 0, 0, 0, 0.0 );
        input.putScalar(1, 1, 0, 0, 0.0 );
        input.putScalar(1, 2, 0, 0, 255.0 );

        DataSet ds = new DataSet(input, null);

        // Act
        sut.preProcess(ds);

        // Assert
        INDArray result = ds.getFeatures();
        long[] shape = result.shape();

        assertEquals(3, shape.length);
        assertEquals(2, shape[0]);
        assertEquals(1, shape[1]);
        assertEquals(5, shape[2]);

        assertEquals(0.0, result.getDouble(0, 0, 0), 0.05);
        assertEquals(255.0, result.getDouble(0, 0, 1), 0.05);
        assertEquals(255.0 * 0.3, result.getDouble(0, 0, 2), 0.05);
        assertEquals(255.0 * 0.59, result.getDouble(0, 0, 3), 0.05);
        assertEquals(255.0 * 0.11, result.getDouble(0, 0, 4), 0.05);

        assertEquals(0.0, result.getDouble(1, 0, 4), 0.05);
        assertEquals(255.0, result.getDouble(1, 0, 3), 0.05);
        assertEquals(255.0 * 0.3, result.getDouble(1, 0, 2), 0.05);
        assertEquals(255.0 * 0.59, result.getDouble(1, 0, 1), 0.05);
        assertEquals(255.0 * 0.11, result.getDouble(1, 0, 0), 0.05);

    }
}
