package org.deeplearning4j.rl4j.observation.preprocessor;

import org.deeplearning4j.datasets.iterator.CombinedPreProcessor;
import org.deeplearning4j.rl4j.observation.preprocessors.*;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class PreProcessorIntegrationTest {

    private static final int HEIGHT = 10;
    private static final int WIDTH = 15;

    private INDArray createRawObservation() {
        CropAndResizeDataSetPreProcessor sut = new CropAndResizeDataSetPreProcessor(HEIGHT, WIDTH, 5, 5, 4, 3, 3, CropAndResizeDataSetPreProcessor.ResizeMethod.NearestNeighbor);
        INDArray rgb = Nd4j.create(new long[] { 3, HEIGHT, WIDTH })
                .putSlice(0, Nd4j.rand(new long[] { HEIGHT, WIDTH }, 0.0, 255.0, Nd4j.getRandom()).castTo(DataType.FLOAT)) // Red
                .putSlice(0, Nd4j.rand(new long[] { HEIGHT, WIDTH }, 0.0, 255.0, Nd4j.getRandom()).castTo(DataType.FLOAT)) // Green
                .putSlice(0, Nd4j.rand(new long[] { HEIGHT, WIDTH }, 0.0, 255.0, Nd4j.getRandom()).castTo(DataType.FLOAT)); // Blue
        return Nd4j.create(new long[] { 1, 3, HEIGHT, WIDTH }).putSlice(0, rgb);

    }

    @Test
    public void test_preProcessors() {

        // Arrange
        SkippingDataSetPreProcessor skip = new SkippingDataSetPreProcessor(2);
        PermuteDataSetPreProcessor nchwTonhwC = new PermuteDataSetPreProcessor(PermuteDataSetPreProcessor.PermutationTypes.NCHWtoNHWC);
        CropAndResizeDataSetPreProcessor crop = new CropAndResizeDataSetPreProcessor(HEIGHT, WIDTH, 1, 4, 8, 8, 3, CropAndResizeDataSetPreProcessor.ResizeMethod.NearestNeighbor);
        PermuteDataSetPreProcessor nhwcTonchw = new PermuteDataSetPreProcessor(PermuteDataSetPreProcessor.PermutationTypes.NHWCtoNCHW);
        RGBtoGrayscaleDataSetPreProcessor rgbToGray = new RGBtoGrayscaleDataSetPreProcessor();
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler();
        PoolingDataSetPreProcessor pool = PoolingDataSetPreProcessor.builder().build();
        PipelineDataSetPreProcessor preProcessor = new PipelineDataSetPreProcessor.Builder()
                .addPreProcessor(skip)
                .addPreProcessor(nchwTonhwC)
                .addPreProcessor(crop)
                .addPreProcessor(nhwcTonchw)
                .addPreProcessor(rgbToGray)
                .addPreProcessor(scaler)
                .addPreProcessor(pool)
                .build();

        INDArray input = createRawObservation();
        DataSet ds = new DataSet(input, null);
        preProcessor.preProcess(ds);
        assertTrue(ds.isEmpty()); // pool element 1 of 4

        input = createRawObservation();
        ds = new DataSet(input, null);
        preProcessor.preProcess(ds);
        assertTrue(ds.isEmpty()); // skipped

        input = createRawObservation();
        ds = new DataSet(input, null);
        preProcessor.preProcess(ds);
        assertTrue(ds.isEmpty()); // pool element 2 of 4

        input = createRawObservation();
        ds = new DataSet(input, null);
        preProcessor.preProcess(ds);
        assertTrue(ds.isEmpty()); // skipped

        input = createRawObservation();
        ds = new DataSet(input, null);
        preProcessor.preProcess(ds);
        assertTrue(ds.isEmpty()); // pool element 3 of 4

        input = createRawObservation();
        ds = new DataSet(input, null);
        preProcessor.preProcess(ds);
        assertTrue(ds.isEmpty()); // skipped

        input = createRawObservation();
        ds = new DataSet(input, null);
        preProcessor.preProcess(ds);

        INDArray results = ds.getFeatures();
        long[] shape = results.shape();

        assertEquals(4, shape.length);
        assertEquals(1, shape[0]);
        assertEquals(4, shape[1]);
        assertEquals(8, shape[2]);
        assertEquals(8, shape[3]);
    }
}
