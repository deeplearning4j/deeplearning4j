package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by susaneraly on 6/17/17.
 */
public class CombinedPreProcessorTests extends BaseDL4JTest {

    @Test
    public void somePreProcessorsCombined() {

        INDArray[] featureArr = new INDArray[] {Nd4j.linspace(100, 200, 20).reshape(10, 2)};
        org.nd4j.linalg.dataset.MultiDataSet multiDataSet =
                        new org.nd4j.linalg.dataset.MultiDataSet(featureArr, null, null, null);

        MultiNormalizerMinMaxScaler minMaxScaler = new MultiNormalizerMinMaxScaler();
        minMaxScaler.fit(multiDataSet);
        CombinedMultiDataSetPreProcessor multiDataSetPreProcessor = new CombinedMultiDataSetPreProcessor.Builder()
                        .addPreProcessor(minMaxScaler).addPreProcessor(1, new addFivePreProcessor()).build();

        multiDataSetPreProcessor.preProcess(multiDataSet);
        assertEquals(Nd4j.zeros(10, 2).addColumnVector(Nd4j.linspace(0, 1, 10).reshape(10, 1)).addi(5),
                        multiDataSet.getFeatures(0));

    }

    /*
        Adds five to the features - assumes multidataset here is one feature and one label
     */
    public final class addFivePreProcessor implements MultiDataSetPreProcessor {

        @Override
        public void preProcess(MultiDataSet multiDataSet) {
            multiDataSet.getFeatures(0).addi(5);
        }
    }
}
