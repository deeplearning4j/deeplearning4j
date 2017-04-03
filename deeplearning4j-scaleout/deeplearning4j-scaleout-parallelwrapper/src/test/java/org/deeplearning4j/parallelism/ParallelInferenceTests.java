package org.deeplearning4j.parallelism;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class ParallelInferenceTests {
    private static MultiLayerNetwork model;
    private static DataSetIterator iterator;

    @Before
    public void setUp() throws Exception {
        if (model == null) {
            File file = new ClassPathResource("models/LenetMnistMLN.zip").getFile();
            model = ModelSerializer.restoreMultiLayerNetwork(file, true);

            iterator = new MnistDataSetIterator(1, false, 12345);
        }
    }

    @After
    public void tearDown() throws Exception {
        iterator.reset();
    }

    @Test
    public void testInferenceSequential1() throws Exception {
        ParallelInference inf = new ParallelInference.Builder(model)
                .inferenceMode(InferenceMode.SEQUENTIAL)
                .workers(2)
                .build();



        log.info("Features shape: {}", Arrays.toString(iterator.next().getFeatureMatrix().shapeInfoDataBuffer().asInt()));

        INDArray array1 = inf.output(iterator.next().getFeatureMatrix());
        INDArray array2 = inf.output(iterator.next().getFeatureMatrix());

        assertFalse(array1.isAttached());
        assertFalse(array2.isAttached());

        INDArray array3 = inf.output(iterator.next().getFeatureMatrix());
        assertFalse(array3.isAttached());

        iterator.reset();

        evalClassifcationSingleThread(inf, iterator);

        // both workers threads should have non-zero
        assertTrue( inf.getWorkerCounter(0) > 100L);
        assertTrue( inf.getWorkerCounter(1) > 100L);
    }


    protected void evalClassifcationSingleThread(@NonNull ParallelInference inf, @NonNull DataSetIterator iterator) {
        DataSet ds = iterator.next();
        log.info("NumColumns: {}", ds.getLabels().columns());
        iterator.reset();
        Evaluation eval = new Evaluation(ds.getLabels().columns());
        while (iterator.hasNext()) {
            ds = iterator.next();
            INDArray output = inf.output(ds.getFeatureMatrix());
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
    }
}