package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;

/**
 * Created by agibsonccc on 9/12/15.
 */
public class StandardScalerTest {

    @Test
    public void testScale() {
        StandardScaler scaler = new StandardScaler();
        DataSetIterator iter = new IrisDataSetIterator(10,150);
        scaler.fit(iter);
        INDArray featureMatrix = new IrisDataSetIterator(150,150).next().getFeatureMatrix();
        INDArray mean = featureMatrix.mean(0);
        INDArray std = featureMatrix.std(0);
        System.out.println(mean);
    }

}
