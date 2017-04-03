package org.deeplearning4j.parallelism;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * @author raver119@gmail.com
 */
public class ParallelInferenceTests {
    private static MultiLayerNetwork model;
    private static DataSetIterator iterator;

    @Before
    public void setUp() throws Exception {
        if (model == null) {
            File file = new ClassPathResource("models/LenetMnistMLN.zip").getFile();
            model = ModelSerializer.restoreMultiLayerNetwork(file, false);

            iterator = new MnistDataSetIterator(32,false,12345);
        }
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testInferenceSequential1() throws Exception {
        ParallelInference inf = new ParallelInference.Builder(model)
                .inferenceMode(InferenceMode.SEQUENTIAL)
                .workers(2)
                .build();

        INDArray array = inf.output(iterator.next().getFeatureMatrix());
    }
}