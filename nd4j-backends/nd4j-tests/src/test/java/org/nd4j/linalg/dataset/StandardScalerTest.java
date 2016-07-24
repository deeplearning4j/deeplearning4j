package org.nd4j.linalg.dataset;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * Created by agibsonccc on 9/12/15.
 */
@RunWith(Parameterized.class)
public class StandardScalerTest extends BaseNd4jTest {
    public StandardScalerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Ignore
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

    @Override
    public char ordering() {
        return 'c';
    }
}
